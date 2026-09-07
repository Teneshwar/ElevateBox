"""
Core data models for call state, lead qualification, and conversation tracking.
All models are Pydantic v2 with strict validation.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────


class Language(str, Enum):
    TELUGU = "te"
    HINDI = "hi"
    ENGLISH = "en"
    UNKNOWN = "unknown"


class LeadTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    UNCLASSIFIED = "unclassified"


class CallStatus(str, Enum):
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    WHATSAPP_SENT = "whatsapp_sent"
    CALLBACK_BOOKED = "callback_booked"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── Lead Discovery ───────────────────────────────────────────────────────────


class LeadDiscovery(BaseModel):
    """Extracted discovery information from the conversation."""

    budget_mentioned: str | None = Field(None, description="Exact words the caller used for budget")
    budget_numeric_inr: float | None = Field(None, description="Parsed numeric budget in INR")
    product_type: str | None = Field(None, description="What the caller sells")
    product_count: str | None = Field(None, description="Approximate number of products")
    timeline: str | None = Field(None, description="When they want the website")
    features_requested: list[str] = Field(
        default_factory=list,
        description="Features mentioned: payment, mobile, catalogue, etc.",
    )
    decision_maker: bool | None = Field(
        None, description="Is the caller the final decision maker?"
    )
    barrier: str | None = Field(None, description="Main blocker if any (budget/timing/person)")
    callback_requested: bool = False
    callback_time_raw: str | None = Field(None, description="Exact phrase used for callback time")
    callback_datetime_ist: datetime | None = Field(None, description="Parsed callback datetime IST")
    caller_name: str | None = None


# ─── Lead Score ───────────────────────────────────────────────────────────────


class LeadScore(BaseModel):
    """Running score and classification with rationale."""

    score: int = Field(0, ge=-100, le=100)
    tier: LeadTier = LeadTier.UNCLASSIFIED
    signals_detected: list[str] = Field(default_factory=list)
    rationale: str = ""
    whatsapp_triggered: bool = False
    classified_at_turn: int = 0


# ─── Conversation Turn ────────────────────────────────────────────────────────


class ConversationTurn(BaseModel):
    """A single turn in the conversation (caller or agent)."""

    role: str  # "user" | "assistant"
    content: str
    language: Language = Language.UNKNOWN
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    turn_number: int = 0


# ─── Call State ───────────────────────────────────────────────────────────────


class CallState(BaseModel):
    """
    Full state of an active or completed call.
    Stored in Redis; serialised as JSON.
    """

    # Identity
    call_id: str = Field(..., description="Vapi call ID")
    exotel_call_sid: str | None = None
    phone_number: str = Field(..., description="Caller's phone number")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Status
    status: CallStatus = CallStatus.INITIATED
    detected_language: Language = Language.UNKNOWN
    turn_count: int = 0

    # Conversation
    conversation: list[ConversationTurn] = Field(default_factory=list)
    full_transcript: str = ""

    # Discovery & scoring
    discovery: LeadDiscovery = Field(default_factory=LeadDiscovery)
    score: LeadScore = Field(default_factory=LeadScore)

    # Actions taken
    mid_call_whatsapp_sent: bool = False
    mid_call_whatsapp_message_id: str | None = None
    post_call_whatsapp_sent: bool = False
    post_call_whatsapp_message_id: str | None = None
    calendar_event_id: str | None = None
    calendar_event_link: str | None = None

    # Raw Vapi data
    vapi_call_object: dict[str, Any] | None = None

    def add_turn(self, role: str, content: str, language: Language = Language.UNKNOWN) -> None:
        self.turn_count += 1
        self.conversation.append(
            ConversationTurn(
                role=role,
                content=content,
                language=language,
                turn_number=self.turn_count,
            )
        )
        self.full_transcript += f"\n[{role.upper()}]: {content}"
        self.updated_at = datetime.utcnow()

    def get_recent_transcript(self, last_n: int = 10) -> str:
        """Return last N turns formatted for LLM context."""
        recent = self.conversation[-last_n:] if len(self.conversation) > last_n else self.conversation
        return "\n".join(f"[{t.role.upper()}]: {t.content}" for t in recent)


# ─── Vapi Webhook Payloads ────────────────────────────────────────────────────


class VapiToolCallPayload(BaseModel):
    """Vapi sends this when the LLM calls a tool function."""

    call_id: str
    tool_call_id: str
    function_name: str
    parameters: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VapiCallEndPayload(BaseModel):
    """Vapi sends this when a call ends."""

    call_id: str
    ended_reason: str | None = None
    duration_seconds: float | None = None
    transcript: str | None = None
    summary: str | None = None
    call_object: dict[str, Any] | None = None


# ─── API Request/Response ─────────────────────────────────────────────────────


class InitiateCallRequest(BaseModel):
    """Request body for POST /api/v1/calls/initiate."""

    phone_number: str | None = Field(
        None, description="Override target phone number (optional)"
    )
    language_hint: Language | None = Field(
        None, description="Hint for initial language (optional)"
    )


class InitiateCallResponse(BaseModel):
    success: bool
    call_id: str | None = None
    exotel_sid: str | None = None
    message: str = ""


class CallStatusResponse(BaseModel):
    call_id: str
    status: CallStatus
    lead_tier: LeadTier
    score: int
    detected_language: Language
    turn_count: int
    mid_call_whatsapp_sent: bool
    post_call_whatsapp_sent: bool
    calendar_event_id: str | None
    created_at: datetime
