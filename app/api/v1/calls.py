"""
Call management endpoints.

POST /api/v1/calls/initiate   — Place the outbound call (protected)
GET  /api/v1/calls/{call_id}  — Get call status
GET  /api/v1/calls/           — List recent calls
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.context import set_call_id
from app.core.exceptions import StateError, VoiceAgentError
from app.core.logging import get_logger
from app.core.security import limiter, require_api_key
from app.models.call import (
    CallState,
    CallStatus,
    CallStatusResponse,
    InitiateCallRequest,
    InitiateCallResponse,
    Language,
)
from app.services.state import get_state_manager
from app.services.vapi_agent import get_vapi_client

router = APIRouter(prefix="/calls", tags=["calls"])
logger = get_logger(__name__)
_bearer = HTTPBearer(auto_error=False)


@router.post(
    "/initiate",
    response_model=InitiateCallResponse,
    summary="Place outbound call to the target number",
)
@limiter.limit("5/minute")
async def initiate_call(
    request: Request,
    body: InitiateCallRequest | None = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> InitiateCallResponse:
    """
    Trigger the outbound AI sales call.

    Protected — requires Bearer token (WEBHOOK_SECRET).
    Rate limited to 5 calls per minute per IP.
    """
    await require_api_key(request, credentials)

    from app.core.config import get_settings
    settings = get_settings()

    to_number = settings.target_phone_number
    if body and body.phone_number:
        from app.core.security import normalise_phone
        to_number = normalise_phone(body.phone_number)

    vapi = get_vapi_client()
    state_mgr = await get_state_manager()

    # Generate a provisional call_id (Vapi will return the real one)
    provisional_id = str(uuid.uuid4())
    set_call_id(provisional_id)

    logger.info("call_initiation_requested", to=to_number)

    try:
        # Ensure assistant is created/updated in Vapi
        assistant_id = await vapi.create_or_update_assistant()

        # Place the outbound call
        call_data = await vapi.initiate_outbound_call(
            assistant_id=assistant_id,
            to_phone_number=to_number,
            call_metadata={
                "language_hint": body.language_hint.value if (body and body.language_hint) else None,
            },
        )

        real_call_id: str = call_data.get("id", provisional_id)
        set_call_id(real_call_id)

        # Create initial call state in Redis
        initial_state = CallState(
            call_id=real_call_id,
            phone_number=to_number,
            status=CallStatus.INITIATED,
            detected_language=Language.UNKNOWN,
            vapi_call_object=call_data,
        )
        await state_mgr.create(initial_state)

        logger.info(
            "call_initiated_successfully",
            call_id=real_call_id,
            to=to_number,
        )

        return InitiateCallResponse(
            success=True,
            call_id=real_call_id,
            message=f"Call initiated to {to_number}",
        )

    except VoiceAgentError as exc:
        logger.error("call_initiation_failed", error=str(exc), detail=exc.detail)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to initiate call: {exc.message}",
        ) from exc
    except Exception as exc:
        logger.exception("unexpected_error_initiating_call")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.get(
    "/{call_id}",
    response_model=CallStatusResponse,
    summary="Get call status and lead classification",
)
async def get_call_status(call_id: str) -> CallStatusResponse:
    state_mgr = await get_state_manager()
    try:
        state = await state_mgr.get_or_raise(call_id)
    except StateError:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    return CallStatusResponse(
        call_id=state.call_id,
        status=state.status,
        lead_tier=state.score.tier,
        score=state.score.score,
        detected_language=state.detected_language,
        turn_count=state.turn_count,
        mid_call_whatsapp_sent=state.mid_call_whatsapp_sent,
        post_call_whatsapp_sent=state.post_call_whatsapp_sent,
        calendar_event_id=state.calendar_event_id,
        created_at=state.created_at,
    )
