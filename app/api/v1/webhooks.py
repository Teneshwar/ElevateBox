"""
Vapi and Exotel webhook handlers — the operational core of the system.

Vapi webhooks received:
  assistant-request       — Vapi needs assistant config (dynamic assistant)
  call-start              — Call connected, update state to IN_PROGRESS
  transcript              — Real-time transcript turns
  tool-calls              — LLM called one of our 4 functions (main action point)
  call-end                — Call ended, trigger post-call sequence
  status-update           — General call status changes

Exotel webhooks:
  /exotel/bridge/{call_id}    — Returns ExoML SIP forward XML
  /exotel/status/{call_id}    — Exotel terminal status callback

All Vapi webhooks are HMAC-verified before processing.
Tool call handlers are async and non-blocking — they return immediately
to Vapi with an ack, then do the heavy work in background tasks.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.context import set_call_id
from app.core.exceptions import StateError, WebhookAuthError
from app.core.logging import get_logger
from app.core.security import normalise_phone, sanitise_webhook_payload, verify_vapi_signature
from app.models.call import CallStatus, Language, LeadTier
from app.services.followup import generate_followup_message
from app.services.qualifier import qualify_lead
from app.services.scheduler import schedule_callback
from app.services.state import get_state_manager
from app.services.whatsapp import get_whatsapp_client

router = APIRouter(tags=["webhooks"])
logger = get_logger(__name__)

# ─── Language code mapping ────────────────────────────────────────────────────

_LANG_MAP: dict[str, Language] = {
    "te": Language.TELUGU,
    "hi": Language.HINDI,
    "en": Language.ENGLISH,
    "te-IN": Language.TELUGU,
    "hi-IN": Language.HINDI,
    "en-IN": Language.ENGLISH,
}


def _parse_language(lang_str: str | None) -> Language:
    if not lang_str:
        return Language.UNKNOWN
    return _LANG_MAP.get(lang_str.lower(), Language.UNKNOWN)


# ─── Main Vapi webhook dispatcher ────────────────────────────────────────────


@router.post("/vapi", summary="Main Vapi event webhook")
async def vapi_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Central Vapi webhook receiver.
    Verifies signature, dispatches to the correct handler.
    Returns quickly (< 200ms) so Vapi doesn't time out.
    """
    try:
        raw_body = await verify_vapi_signature(request)
    except WebhookAuthError as exc:
        logger.warning("vapi_webhook_auth_failed", error=str(exc))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    try:
        payload: dict[str, Any] = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    payload = sanitise_webhook_payload(payload)
    event_type: str = payload.get("type", "unknown")
    call_data: dict[str, Any] = payload.get("call", {})
    call_id: str = call_data.get("id", payload.get("callId", ""))

    if call_id:
        set_call_id(call_id)

    logger.info("vapi_webhook_received", event_type=event_type, call_id=call_id)

    # Dispatch
    if event_type == "call-start":
        background_tasks.add_task(_handle_call_start, call_id, call_data)
        return JSONResponse({"status": "ok"})

    elif event_type == "transcript":
        background_tasks.add_task(_handle_transcript, call_id, payload)
        return JSONResponse({"status": "ok"})

    elif event_type == "tool-calls":
        # Tool calls must return a result synchronously within a reasonable time
        return await _handle_tool_calls(call_id, payload, background_tasks)

    elif event_type == "call-end":
        background_tasks.add_task(_handle_call_end, call_id, payload)
        return JSONResponse({"status": "ok"})

    elif event_type == "status-update":
        background_tasks.add_task(_handle_status_update, call_id, payload)
        return JSONResponse({"status": "ok"})

    elif event_type == "assistant-request":
        # Vapi requesting assistant config dynamically — return our config
        return await _handle_assistant_request(payload)

    else:
        logger.debug("unhandled_vapi_event", event_type=event_type)
        return JSONResponse({"status": "ok"})


# ─── Event handlers ───────────────────────────────────────────────────────────


async def _handle_call_start(call_id: str, call_data: dict[str, Any]) -> None:
    """Update call state to IN_PROGRESS when call connects."""
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
        state.status = CallStatus.IN_PROGRESS
        state.vapi_call_object = call_data
        await state_mgr.save(state)
        logger.info("call_started", call_id=call_id)
    except StateError:
        logger.warning("call_start_state_not_found", call_id=call_id)


async def _handle_transcript(call_id: str, payload: dict[str, Any]) -> None:
    """Record transcript turns and update detected language."""
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        return

    transcript = payload.get("transcript", {})
    role: str = transcript.get("role", "unknown")
    text: str = transcript.get("transcript", "").strip()
    lang_tag: str = transcript.get("language", "")

    if not text:
        return

    detected = _parse_language(lang_tag)

    # Lock detected language on first user turn
    if role == "user" and state.detected_language == Language.UNKNOWN and detected != Language.UNKNOWN:
        state.detected_language = detected
        logger.info("language_detected", language=detected, call_id=call_id)

    state.add_turn(role=role, content=text, language=detected)
    await state_mgr.save(state)


async def _handle_tool_calls(
    call_id: str,
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Handle LLM function calls from Vapi.

    CRITICAL: Vapi expects a tool result response within ~5 seconds.
    Heavy work (WhatsApp sends, Calendar creates) is kicked off as
    background tasks and we return an immediate ack to Vapi.
    """
    tool_calls: list[dict[str, Any]] = payload.get("toolCallList", [])
    results: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        fn_name: str = tool_call.get("function", {}).get("name", "")
        tool_call_id: str = tool_call.get("id", "")
        raw_args: str = tool_call.get("function", {}).get("arguments", "{}")

        try:
            args: dict[str, Any] = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        logger.info("tool_call_received", function=fn_name, call_id=call_id)

        if fn_name == "classify_lead":
            result, bg_task = await _tool_classify_lead(call_id, args)
            if bg_task:
                background_tasks.add_task(bg_task)

        elif fn_name == "send_whatsapp_mid_call":
            result = {"status": "triggered", "message": "WhatsApp is being sent"}
            background_tasks.add_task(_bg_send_mid_call_whatsapp, call_id, args)

        elif fn_name == "book_callback":
            result = {"status": "processing", "message": "Booking your callback now"}
            background_tasks.add_task(_bg_book_callback, call_id, args)

        elif fn_name == "end_call_summary":
            result = {"status": "received"}
            background_tasks.add_task(_bg_end_call_summary, call_id, args)

        else:
            result = {"error": f"Unknown function: {fn_name}"}

        results.append({
            "toolCallId": tool_call_id,
            "result": json.dumps(result),
        })

    return JSONResponse({"results": results})


async def _handle_call_end(call_id: str, payload: dict[str, Any]) -> None:
    """
    Called when Vapi signals the call has ended.
    The end_call_summary tool should have already fired, but we use this
    as a safety net to ensure post-call actions run even if the tool didn't fire.
    """
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        logger.warning("call_end_state_not_found", call_id=call_id)
        return

    # Get final transcript from Vapi call object if available
    call_obj = payload.get("call", {})
    artifact = call_obj.get("artifact", {})
    full_transcript: str = artifact.get("transcript", "") or state.full_transcript

    if full_transcript and not state.full_transcript:
        state.full_transcript = full_transcript

    # Safety net: if post-call WhatsApp hasn't been sent, send it now
    if not state.post_call_whatsapp_sent:
        logger.info("post_call_safety_net_triggered", call_id=call_id)
        await _run_post_call_sequence(state)

    await state_mgr.mark_completed(call_id)


async def _handle_status_update(call_id: str, payload: dict[str, Any]) -> None:
    """Handle general status updates from Vapi."""
    vapi_status: str = payload.get("status", "")
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
        if vapi_status == "ringing":
            state.status = CallStatus.RINGING
            await state_mgr.save(state)
    except StateError:
        pass


async def _handle_assistant_request(payload: dict[str, Any]) -> JSONResponse:
    """
    Vapi's dynamic assistant request — return assistant ID.
    We use a static assistant ID registered at startup.
    """
    from app.core.config import get_settings
    settings = get_settings()
    # The assistant was created at startup; Vapi just needs its ID
    # In a full dynamic setup you'd return the full config here
    return JSONResponse({"assistantId": payload.get("assistantId", "")})


# ─── Tool background tasks ────────────────────────────────────────────────────


async def _tool_classify_lead(
    call_id: str,
    args: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """
    Synchronous part of classify_lead — updates state and decides if
    WhatsApp should fire. Returns (result_dict, optional_bg_task).
    """
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        return {"status": "error", "message": "Call state not found"}, None

    tier_str: str = args.get("tier", "unclassified")
    llm_score: int = int(args.get("score", 0))
    signals: list[str] = args.get("signals", [])
    rationale: str = args.get("rationale", "")

    discovery_data = {
        "budget_mentioned": args.get("budget_mentioned"),
        "product_type": args.get("product_type"),
        "product_count": args.get("product_count"),
        "timeline": args.get("timeline"),
        "features_requested": args.get("features_requested", []),
        "decision_maker": args.get("decision_maker"),
        "barrier": args.get("barrier"),
        "caller_name": args.get("caller_name"),
    }

    updated_score, should_fire_whatsapp = qualify_lead(
        call_state=state,
        llm_tier=tier_str,
        llm_score=llm_score,
        signals=signals,
        rationale=rationale,
        discovery_data=discovery_data,
    )

    state.score = updated_score
    await state_mgr.save(state)

    bg_task = None
    if should_fire_whatsapp:
        bg_task = lambda: _bg_send_mid_call_whatsapp(  # noqa: E731
            call_id,
            {
                "trigger_reason": "auto_hot_classification",
                "context_summary": rationale,
                "detected_language": state.detected_language.value,
            },
        )

    return {
        "tier": updated_score.tier.value,
        "score": updated_score.score,
        "acknowledged": True,
    }, bg_task


async def _bg_send_mid_call_whatsapp(call_id: str, args: dict[str, Any]) -> None:
    """
    Background task: send the mid-call WhatsApp message.
    This runs concurrently while the voice conversation continues.
    """
    state_mgr = await get_state_manager()
    whatsapp = get_whatsapp_client()

    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        logger.error("mid_call_whatsapp_state_not_found", call_id=call_id)
        return

    # Idempotency guard
    if state.mid_call_whatsapp_sent:
        logger.info("mid_call_whatsapp_already_sent", call_id=call_id)
        return

    lang_str: str = args.get("detected_language", state.detected_language.value)
    language = _parse_language(lang_str)
    context_summary: str = args.get("context_summary", "")

    logger.info(
        "sending_mid_call_whatsapp",
        call_id=call_id,
        language=language,
        trigger=args.get("trigger_reason"),
    )

    try:
        message_id = await whatsapp.send_mid_call_message(
            to_number=state.phone_number,
            language=language,
            context_summary=context_summary,
        )
        state.mid_call_whatsapp_sent = True
        state.mid_call_whatsapp_message_id = message_id
        state.status = CallStatus.WHATSAPP_SENT
        await state_mgr.save(state)

        logger.info(
            "mid_call_whatsapp_sent_successfully",
            call_id=call_id,
            message_id=message_id,
        )
    except Exception as exc:
        logger.error("mid_call_whatsapp_failed", call_id=call_id, error=str(exc))


async def _bg_book_callback(call_id: str, args: dict[str, Any]) -> None:
    """
    Background task: parse the time expression and create a Google Calendar event.
    """
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        logger.error("book_callback_state_not_found", call_id=call_id)
        return

    time_expression: str = args.get("time_expression", "tomorrow morning")
    customer_name: str | None = args.get("customer_name") or state.discovery.caller_name

    # Build a brief discovery summary for the calendar description
    disc = state.discovery
    discovery_summary = (
        f"Product: {disc.product_type or 'not specified'}\n"
        f"Budget: {disc.budget_mentioned or 'not mentioned'}\n"
        f"Timeline: {disc.timeline or 'not specified'}\n"
        f"Features: {', '.join(disc.features_requested) or 'not specified'}\n"
        f"Lead tier: {state.score.tier.value.upper()}"
    )

    logger.info(
        "booking_callback",
        call_id=call_id,
        time_expression=time_expression,
        customer_name=customer_name,
    )

    try:
        result = await schedule_callback(
            time_expression=time_expression,
            customer_name=customer_name,
            phone_number=state.phone_number,
            discovery_summary=discovery_summary,
        )
        state.calendar_event_id = result.get("event_id")
        state.calendar_event_link = result.get("event_link")
        state.discovery.callback_requested = True
        state.discovery.callback_time_raw = time_expression
        state.status = CallStatus.CALLBACK_BOOKED
        await state_mgr.save(state)

        logger.info(
            "callback_booked_successfully",
            call_id=call_id,
            event_id=result.get("event_id"),
            callback_dt=result.get("callback_datetime_ist"),
        )
    except Exception as exc:
        logger.error("callback_booking_failed", call_id=call_id, error=str(exc))


async def _bg_end_call_summary(call_id: str, args: dict[str, Any]) -> None:
    """
    Background task triggered by the end_call_summary tool call.
    Updates state with final data and runs the post-call sequence.
    """
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        logger.error("end_call_summary_state_not_found", call_id=call_id)
        return

    # Update state with final LLM-extracted data
    disc = state.discovery
    disc.caller_name = args.get("caller_name") or disc.caller_name
    disc.budget_mentioned = args.get("budget_mentioned") or disc.budget_mentioned
    disc.product_type = args.get("product_type") or disc.product_type
    disc.product_count = args.get("product_count") or disc.product_count
    disc.timeline = args.get("timeline") or disc.timeline
    disc.barrier = args.get("barrier") or disc.barrier

    features = args.get("features_requested", [])
    if features:
        existing = set(disc.features_requested)
        disc.features_requested = list(existing | set(features))

    # Store key quotes in signals for follow-up generation
    key_quotes: list[str] = args.get("key_quotes", [])
    if key_quotes:
        state.score.signals_detected = list(
            set(state.score.signals_detected + key_quotes)
        )

    lang_str: str = args.get("detected_language", state.detected_language.value)
    if state.detected_language == Language.UNKNOWN:
        state.detected_language = _parse_language(lang_str)

    # Final tier override from the LLM if confident
    final_tier_str: str = args.get("final_tier", state.score.tier.value)
    from app.services.qualifier import determine_tier
    # Only override if LLM is more confident (call is ending)
    tier_map = {"hot": LeadTier.HOT, "warm": LeadTier.WARM, "cold": LeadTier.COLD}
    if final_tier_str in tier_map:
        state.score.tier = tier_map[final_tier_str]

    await state_mgr.save(state)

    # Run the full post-call messaging sequence
    await _run_post_call_sequence(state)
    await state_mgr.save(state)


async def _run_post_call_sequence(state: "CallState") -> None:  # type: ignore[name-defined]
    """
    Execute the full post-call WhatsApp sequence:
    1. Generate personalised follow-up message using GPT-4o
    2. Send personalised text
    3. Send resume PDF
    4. Send architecture image
    """
    from app.models.call import CallState as CS  # avoid circular
    whatsapp = get_whatsapp_client()

    logger.info("post_call_sequence_starting", call_id=state.call_id)

    # Generate personalised message
    try:
        personalised_message = await generate_followup_message(state)
    except Exception as exc:
        logger.error("followup_generation_error", call_id=state.call_id, error=str(exc))
        personalised_message = (
            f"Hi! It was great speaking with you. "
            f"Please find our details below. Looking forward to working with you!"
        )

    # Send all three messages
    try:
        results = await whatsapp.send_full_post_call_sequence(
            to_number=state.phone_number,
            personalised_summary=personalised_message,
            language=state.detected_language,
        )
        state.post_call_whatsapp_sent = True
        state.post_call_whatsapp_message_id = results.get("summary", "")
        logger.info(
            "post_call_sequence_complete",
            call_id=state.call_id,
            messages_sent=len(results),
        )
    except Exception as exc:
        logger.error("post_call_sequence_failed", call_id=state.call_id, error=str(exc))


# ─── Exotel bridge endpoint ───────────────────────────────────────────────────


@router.get(
    "/exotel/bridge/{call_id}",
    response_class=PlainTextResponse,
    summary="ExoML: Bridge Exotel call to Vapi SIP",
)
async def exotel_bridge(call_id: str) -> PlainTextResponse:
    """
    Exotel fetches this URL when the callee answers.
    Returns ExoML instructing Exotel to connect the audio to Vapi's SIP endpoint.

    Vapi SIP format: sip:{assistant_id}@sip.vapi.ai
    """
    from app.core.config import get_settings
    from app.services.telephony import get_exotel_client

    settings = get_settings()
    # The Vapi SIP URI for our assistant
    # When Exotel connects to this URI, Vapi's voice agent takes over
    vapi_sip_uri = f"sip:{settings.vapi_phone_number_id}@sip.vapi.ai"

    exo_client = get_exotel_client()
    exoml = exo_client.build_sip_bridge_exoml(vapi_sip_uri)

    logger.info("exotel_bridge_xml_served", call_id=call_id, sip_uri=vapi_sip_uri)
    return PlainTextResponse(content=exoml, media_type="application/xml")


@router.post(
    "/exotel/status/{call_id}",
    summary="Exotel terminal status callback",
)
async def exotel_status_callback(
    call_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Exotel calls this when the call reaches a terminal state.
    We use it to clean up state for calls that Exotel ended independently.
    """
    from app.core.security import verify_exotel_signature
    raw_body = await verify_exotel_signature(request)

    form_data = await request.form()
    call_status: str = str(form_data.get("Status", ""))
    exotel_sid: str = str(form_data.get("CallSid", ""))

    logger.info(
        "exotel_status_callback",
        call_id=call_id,
        exotel_sid=exotel_sid,
        status=call_status,
    )

    if call_status in ("completed", "failed", "no-answer", "busy"):
        state_mgr = await get_state_manager()
        try:
            state = await state_mgr.get(call_id)
            if state:
                state.exotel_call_sid = exotel_sid
                await state_mgr.save(state)
        except Exception as exc:
            logger.warning("exotel_status_state_update_failed", error=str(exc))

    return JSONResponse({"status": "ok"})


# ─── Custom STT/TTS endpoints (used by Vapi custom providers) ─────────────────


@router.post("/stt", summary="Sarvam STT endpoint for Vapi custom transcriber")
async def custom_stt(request: Request) -> JSONResponse:
    """
    Vapi calls this with audio chunks for transcription.
    Returns transcript in Vapi's expected format.
    """
    from app.services.stt import get_stt_client

    body = await request.json()
    audio_b64: str = body.get("audio", "")
    call_id: str = body.get("callId", "")
    language_hint_str: str = body.get("language", "")

    language_hint = _parse_language(language_hint_str) if language_hint_str else None

    # Get current call language from state if known
    if not language_hint and call_id:
        state_mgr = await get_state_manager()
        state = await state_mgr.get(call_id)
        if state and state.detected_language != Language.UNKNOWN:
            language_hint = state.detected_language

    stt = get_stt_client()
    transcript, detected_lang = await stt.transcribe_base64(
        audio_b64,
        language_hint=language_hint,
    )

    return JSONResponse({
        "transcript": transcript,
        "confidence": 0.95,
        "language": detected_lang.value,
    })


@router.post("/tts", summary="Sarvam TTS endpoint for Vapi custom voice")
async def custom_tts(request: Request) -> Response:
    """
    Vapi calls this with text to synthesize.
    Returns raw audio bytes.
    """
    from app.services.tts import get_tts_service

    body = await request.json()
    text: str = body.get("text", "")
    language_str: str = body.get("language", "en")
    call_id: str = body.get("callId", "")

    language = _parse_language(language_str)

    # Get language from call state if available
    if language == Language.UNKNOWN and call_id:
        state_mgr = await get_state_manager()
        state = await state_mgr.get(call_id)
        if state and state.detected_language != Language.UNKNOWN:
            language = state.detected_language

    tts = get_tts_service()
    audio_bytes = await tts.synthesize(text, language)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"X-Sample-Rate": "8000"},
    )


# ─── Vapi tool-specific endpoints (alternative routing) ──────────────────────
# These are referenced in the Vapi tool definitions as individual URLs.
# They parse the Vapi tool payload format and delegate to handlers above.


@router.post("/vapi/tool/classify_lead")
async def tool_classify_lead(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    payload = await request.json()
    call_id: str = payload.get("call", {}).get("id", "")
    message = payload.get("message", {})
    tool_calls = message.get("toolCallList", [])

    if tool_calls:
        tc = tool_calls[0]
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        result, bg = await _tool_classify_lead(call_id, args)
        if bg:
            background_tasks.add_task(bg)
        return JSONResponse({"results": [{"toolCallId": tc.get("id"), "result": json.dumps(result)}]})

    return JSONResponse({"results": []})


@router.post("/vapi/tool/send_whatsapp_mid_call")
async def tool_send_whatsapp(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    payload = await request.json()
    call_id: str = payload.get("call", {}).get("id", "")
    message = payload.get("message", {})
    tool_calls = message.get("toolCallList", [])

    if tool_calls:
        tc = tool_calls[0]
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        background_tasks.add_task(_bg_send_mid_call_whatsapp, call_id, args)
        result = {"status": "triggered", "message": "WhatsApp is being sent now"}
        return JSONResponse({"results": [{"toolCallId": tc.get("id"), "result": json.dumps(result)}]})

    return JSONResponse({"results": []})


@router.post("/vapi/tool/book_callback")
async def tool_book_callback(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    payload = await request.json()
    call_id: str = payload.get("call", {}).get("id", "")
    message = payload.get("message", {})
    tool_calls = message.get("toolCallList", [])

    if tool_calls:
        tc = tool_calls[0]
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        background_tasks.add_task(_bg_book_callback, call_id, args)
        result = {"status": "processing", "message": "Booking callback, will confirm shortly"}
        return JSONResponse({"results": [{"toolCallId": tc.get("id"), "result": json.dumps(result)}]})

    return JSONResponse({"results": []})


@router.post("/vapi/tool/end_call_summary")
async def tool_end_call_summary(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    payload = await request.json()
    call_id: str = payload.get("call", {}).get("id", "")
    message = payload.get("message", {})
    tool_calls = message.get("toolCallList", [])

    if tool_calls:
        tc = tool_calls[0]
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        background_tasks.add_task(_bg_end_call_summary, call_id, args)
        result = {"status": "received", "message": "Post-call sequence initiated"}
        return JSONResponse({"results": [{"toolCallId": tc.get("id"), "result": json.dumps(result)}]})

    return JSONResponse({"results": []})
