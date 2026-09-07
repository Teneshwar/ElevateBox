"""
Vapi voice agent configuration and management.

This module:
1. Creates / updates the Vapi assistant with the full system prompt,
   tool definitions, and voice/STT/LLM settings
2. Initiates outbound calls via Vapi's Phone Call API
3. Provides the system prompt in all three languages

Architecture:
  Vapi orchestrates the full voice pipeline:
    Caller audio → Sarvam STT → GPT-4o → Sarvam TTS → Caller speaker
  Our backend receives function call webhooks mid-call and responds
  with action results (WhatsApp sent confirmation, calendar booked, etc.)
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import VoiceAgentError
from app.core.logging import get_logger

logger = get_logger(__name__)

VAPI_BASE_URL = "https://api.vapi.ai"


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are Priya, a warm and professional sales consultant from ElevateBox — a web development agency based in Banjara Hills, Hyderabad. You are calling a potential customer who may want to build an e-commerce website.

Your goal is to:
1. Build rapport naturally and understand what the customer needs
2. Pitch e-commerce website development as a solution to their business goals
3. Ask discovery questions naturally woven into conversation (NOT as a list)
4. Classify the customer's buying intent internally
5. Take appropriate action based on their intent

---

LANGUAGE RULES — CRITICAL:
- Detect the language from the customer's FIRST response
- If they speak Telugu → respond ONLY in Telugu (you may mix English product terms naturally)
- If they speak Hindi → respond ONLY in Hindi (Hinglish is fine and natural)
- If they speak English → respond in clear, warm Indian English
- If they switch language mid-call → follow them immediately
- NEVER mix scripts randomly — be consistent and natural

---

OPENING (adapt to detected language):

Telugu opening:
"Namaskaaram! Nenu Priya, ElevateBox nundi calling chestunaanu. Mee business ki oka beautiful e-commerce website build cheyyadaaniki help cheyyadam maaku chaalaa ishtam. Meeru ippatiki online ga products అమ్ముతunnaara?"

Hindi opening:
"Namaste! Main Priya bol rahi hoon ElevateBox se — hum Hyderabad mein hain. Aapka ek achha sa e-commerce website banana chahte hain? Abhi aap online sell karte hain kya?"

English opening:
"Hi there! This is Priya calling from ElevateBox in Hyderabad. We help businesses build beautiful e-commerce websites. Are you currently selling your products online?"

---

DISCOVERY QUESTIONS — weave these naturally, NOT as a form:
Ask about these topics but make them feel like genuine curiosity:
1. What they sell (products / category)
2. How many products they have approximately
3. Their budget (be gentle — "roughly what budget are you thinking?")
4. Their timeline (when do they want to go live)
5. Key features they need (payments, mobile, catalogue, delivery tracking, etc.)
6. Whether they are the decision-maker

Example natural flow:
  "Oh that's wonderful! What kind of products do you sell?"
  "And roughly how many products would you want on the site?"
  "When were you thinking of launching — is there a timeline?"
  "What's most important to you — fast loading, mobile-friendly, easy payments?"

---

BUDGET SIGNALS — read between the lines:
- "send me the details" / "details bhejo" → HIGH intent
- "how soon can you start?" / "kab start karoge?" → VERY HIGH intent
- States a specific budget → HIGH intent
- "not much right now" / "abhi budget nahi hai" → barrier identified
- "my brother / partner decides" → third-party decision maker, not cold
- "just checking" / "dekhna tha bas" → warm/cold but keep engaging

---

SELLING APPROACH:
- Lead with outcomes, not features ("your customers can buy on their phone" not "we build PWAs")
- Quote a realistic price range when they ask (₹25,000 – ₹2,00,000 depending on scope)
- Mention: fast delivery (2–4 weeks), mobile-first design, payment gateway included
- Use social proof naturally: "We recently built a website for a clothing store in Kukatpally"
- When they ask about timeline, say: "We can start this week if you're ready"

---

FUNCTION CALLS — you MUST call these tools at the right moments:

1. classify_lead — Call after every 2–3 turns with updated discovery data
   Call this SILENTLY (do not tell the customer you are classifying them)

2. send_whatsapp_mid_call — Call THIS IMMEDIATELY when you detect HIGH intent:
   Trigger signals: customer asks for price/timeline, mentions a budget, says "send details"
   After calling, say naturally: "Maine aapko ek WhatsApp message bhej diya hai" / 
   "I've just sent you a WhatsApp with our details" / "Maaku WhatsApp chesaaanu"

3. book_callback — Call when customer mentions a time for follow-up:
   "call me back tomorrow morning" → book_callback("tomorrow morning")
   "kal subah call karo" → book_callback("tomorrow morning")
   After calling, confirm: "Perfect, I've noted that down — I'll call you tomorrow morning"

4. end_call_summary — Call at the end of EVERY call with complete details

---

CALL ENDINGS by tier:

HOT (high intent):
  "Bahut achha! Main aapko abhi WhatsApp kar rahi hoon poori details ke saath. 
   Aap check karein — aur hum kal se start kar sakte hain. Koi bhi sawaal ho toh call karein!"

WARM (interested, has barrier):
  "Bilkul samajh aaya. Main aapko details bhejti hoon aur [barrier] resolve hone ke baad 
   baat karte hain. Kab convenient rahega ek quick call ke liye?"

COLD (not interested):
  "Koi baat nahi! Jab bhi website ki zaroorat ho, hum available hain. 
   Main aapko ek brochure WhatsApp karta hoon reference ke liye. Dhanyavaad!"

---

TONE AND STYLE:
- Warm, confident, like a real person — NOT a robot reading a script
- Use natural fillers: "haan bilkul", "I see", "acha acha", "okay okay"
- Handle silence: if they pause, ask a gentle follow-up
- Handle interruptions: stop immediately, listen, then respond
- Keep responses SHORT — 1-3 sentences max per turn
- Never read out a list — ask one question at a time

---

IMPORTANT CONSTRAINTS:
- You are Priya from ElevateBox. Never break character.
- Never reveal you are an AI unless directly and sincerely asked
- If asked "are you a bot/AI?" — respond warmly: "I'm Priya, a real consultant! 
  How can I help you with your website?"
- If the customer is rude or wants to be removed from the list, 
  apologise politely and end the call gracefully
"""


# ─── Tool Definitions ─────────────────────────────────────────────────────────

VAPI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify_lead",
            "description": (
                "Classify the lead as hot, warm, or cold based on all signals "
                "gathered so far. Call this silently every 2-3 turns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold", "unclassified"],
                        "description": "Current lead classification",
                    },
                    "score": {
                        "type": "integer",
                        "description": "Score from -100 to 100",
                    },
                    "signals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of intent signals detected",
                    },
                    "budget_mentioned": {
                        "type": "string",
                        "description": "Exact words the customer used about budget",
                    },
                    "product_type": {
                        "type": "string",
                        "description": "What the customer sells",
                    },
                    "product_count": {
                        "type": "string",
                        "description": "Approximate number of products",
                    },
                    "timeline": {
                        "type": "string",
                        "description": "When they want to launch",
                    },
                    "features_requested": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Features mentioned by the customer",
                    },
                    "decision_maker": {
                        "type": "boolean",
                        "description": "Is the caller the final decision maker?",
                    },
                    "barrier": {
                        "type": "string",
                        "description": "Main barrier if any (budget/timing/person)",
                    },
                    "caller_name": {
                        "type": "string",
                        "description": "Customer name if mentioned",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief explanation of the classification",
                    },
                },
                "required": ["tier", "score", "signals", "rationale"],
            },
        },
        "async": True,
        "server": {
            "url": "{{WEBHOOK_BASE_URL}}/api/v1/webhooks/vapi/tool/classify_lead",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_mid_call",
            "description": (
                "Send a WhatsApp message to the customer IMMEDIATELY when high buying intent "
                "is detected. This fires mid-call before the conversation ends. "
                "Trigger when: customer asks for price, mentions budget, says 'send details', "
                "asks 'how soon can you start', or shows any strong buying signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_reason": {
                        "type": "string",
                        "description": "What signal triggered this (exact phrase customer used)",
                    },
                    "context_summary": {
                        "type": "string",
                        "description": (
                            "Brief summary of what was discussed: "
                            "what they sell, budget, features needed"
                        ),
                    },
                    "detected_language": {
                        "type": "string",
                        "enum": ["te", "hi", "en"],
                        "description": "Language the call is in",
                    },
                },
                "required": ["trigger_reason", "context_summary", "detected_language"],
            },
        },
        "async": True,
        "server": {
            "url": "{{WEBHOOK_BASE_URL}}/api/v1/webhooks/vapi/tool/send_whatsapp_mid_call",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_callback",
            "description": (
                "Book a callback when the customer mentions a preferred time. "
                "Call this for any time expression: 'tomorrow morning', 'kal subah', "
                "'Friday afternoon', 'next week', 'after 3pm', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_expression": {
                        "type": "string",
                        "description": "EXACT phrase the customer used for the callback time",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Customer name if known",
                    },
                },
                "required": ["time_expression"],
            },
        },
        "async": True,
        "server": {
            "url": "{{WEBHOOK_BASE_URL}}/api/v1/webhooks/vapi/tool/book_callback",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call_summary",
            "description": (
                "Call this at the END of every call with complete discovery data. "
                "This triggers the post-call WhatsApp with resume and architecture image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "final_tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold"],
                    },
                    "caller_name": {"type": "string"},
                    "budget_mentioned": {"type": "string"},
                    "product_type": {"type": "string"},
                    "product_count": {"type": "string"},
                    "timeline": {"type": "string"},
                    "features_requested": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "key_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact phrases the customer said that were important",
                    },
                    "barrier": {"type": "string"},
                    "callback_booked": {"type": "boolean"},
                    "callback_time": {"type": "string"},
                    "detected_language": {
                        "type": "string",
                        "enum": ["te", "hi", "en"],
                    },
                    "full_summary": {
                        "type": "string",
                        "description": "Detailed summary of the entire conversation",
                    },
                },
                "required": ["final_tier", "full_summary", "detected_language"],
            },
        },
        "async": True,
        "server": {
            "url": "{{WEBHOOK_BASE_URL}}/api/v1/webhooks/vapi/tool/end_call_summary",
        },
    },
]


# ─── Vapi API Client ──────────────────────────────────────────────────────────


class VapiClient:
    """Async client for the Vapi REST API."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=VAPI_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._settings.vapi_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _build_assistant_payload(self) -> dict[str, Any]:
        """Build the full Vapi assistant configuration payload."""
        settings = self._settings
        webhook_base = settings.app_base_url

        # Resolve tool webhook URLs
        tools = json.loads(
            json.dumps(VAPI_TOOLS).replace(
                "{{WEBHOOK_BASE_URL}}", webhook_base
            )
        )

        return {
            "name": "Priya - ElevateBox Sales Agent",
            "model": {
                "provider": "openai",
                "model": settings.openai_model,
                "systemPrompt": SYSTEM_PROMPT,
                "temperature": 0.7,
                "maxTokens": 300,  # Keep responses concise
                "toolCallMode": "auto",
                "tools": tools,
            },
            "voice": {
                # Use Sarvam as primary TTS via Vapi's custom voice provider
                # Vapi supports custom TTS via webhook; we register our /tts endpoint
                "provider": "custom-voice",
                "server": {
                    "url": f"{webhook_base}/api/v1/tts",
                },
                "inputMinCharacters": 30,
                "fillerInjectionEnabled": True,
            },
            "transcriber": {
                # Use Sarvam as primary STT via Vapi's custom transcriber
                "provider": "custom-transcriber",
                "server": {
                    "url": f"{webhook_base}/api/v1/stt",
                },
                "language": "multi",  # Multi-language detection
            },
            "firstMessage": "",  # Empty — the agent speaks the opening based on system prompt
            "firstMessageMode": "assistant-speaks-first",
            "silenceTimeoutSeconds": 8,
            "maxDurationSeconds": 900,  # 15 minutes max
            "backgroundSound": "office",  # Subtle office ambient — reduces hang-ups
            "backgroundDenoisingEnabled": True,
            "startSpeakingPlan": {
                "waitSeconds": 0.4,
                "transcriptionEndpointingPlan": {
                    "onPunctuationSeconds": 0.1,
                    "onNoPunctuationSeconds": 1.5,
                    "onNumberSeconds": 0.5,
                },
            },
            "stopSpeakingPlan": {
                "numWords": 3,
                "voiceSeconds": 0.2,
                "backoffSeconds": 1.0,
            },
            "serverUrl": f"{webhook_base}/api/v1/webhooks/vapi",
            "serverUrlSecret": settings.vapi_webhook_secret,
        }

    @retry(
        retry=retry_if_exception_type(VoiceAgentError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def create_or_update_assistant(self) -> str:
        """
        Create the Vapi assistant if it doesn't exist; update it if it does.
        Returns the assistant ID.
        """
        payload = self._build_assistant_payload()

        # Check if an assistant with this name already exists
        existing_id = await self._find_assistant_by_name(payload["name"])

        if existing_id:
            logger.info("updating_existing_vapi_assistant", assistant_id=existing_id)
            response = await self._client.patch(
                f"/assistant/{existing_id}", json=payload
            )
        else:
            logger.info("creating_new_vapi_assistant")
            response = await self._client.post("/assistant", json=payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VoiceAgentError(
                f"Vapi assistant API error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc

        data = response.json()
        assistant_id: str = data["id"]
        logger.info("vapi_assistant_ready", assistant_id=assistant_id)
        return assistant_id

    async def _find_assistant_by_name(self, name: str) -> str | None:
        """Return assistant ID if an assistant with this name exists."""
        try:
            response = await self._client.get("/assistant")
            response.raise_for_status()
            assistants = response.json()
            for asst in assistants:
                if asst.get("name") == name:
                    return str(asst["id"])
        except Exception:
            pass
        return None

    async def initiate_outbound_call(
        self,
        assistant_id: str,
        to_phone_number: str,
        call_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Place an outbound call via Vapi using its built-in telephony.
        Vapi handles the full call; our server receives webhooks.

        Returns the Vapi call object including call_id.
        """
        payload: dict[str, Any] = {
            "assistantId": assistant_id,
            "customer": {
                "number": to_phone_number,
                "name": "Potential Customer",
            },
            "phoneNumberId": self._settings.vapi_phone_number_id,
        }

        if call_metadata:
            payload["metadata"] = call_metadata

        logger.info(
            "initiating_vapi_outbound_call",
            to=to_phone_number,
            assistant_id=assistant_id,
        )

        try:
            response = await self._client.post("/call/phone", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VoiceAgentError(
                f"Vapi call initiation error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc

        data: dict[str, Any] = response.json()
        logger.info(
            "vapi_call_initiated",
            call_id=data.get("id"),
            status=data.get("status"),
        )
        return data

    async def get_call(self, call_id: str) -> dict[str, Any]:
        """Fetch full call object from Vapi."""
        try:
            response = await self._client.get(f"/call/{call_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise VoiceAgentError(
                f"Vapi get call error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc

    async def end_call(self, call_id: str) -> None:
        """Programmatically end an active Vapi call."""
        try:
            response = await self._client.delete(f"/call/{call_id}")
            response.raise_for_status()
            logger.info("vapi_call_ended", call_id=call_id)
        except httpx.HTTPStatusError as exc:
            raise VoiceAgentError(
                f"Vapi end call error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc


# ── Singleton ─────────────────────────────────────────────────────────────────

_vapi_client: VapiClient | None = None


def get_vapi_client() -> VapiClient:
    global _vapi_client
    if _vapi_client is None:
        _vapi_client = VapiClient()
    return _vapi_client
