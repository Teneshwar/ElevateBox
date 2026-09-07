"""
Application-level exceptions.
All business logic raises these; the FastAPI error handlers convert them to HTTP responses.
"""
from __future__ import annotations


class VoiceAgentError(Exception):
    """Base exception for all voice agent errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class TelephonyError(VoiceAgentError):
    """Exotel call placement or SIP errors."""


class STTError(VoiceAgentError):
    """Speech-to-text transcription errors."""


class TTSError(VoiceAgentError):
    """Text-to-speech synthesis errors."""


class LLMError(VoiceAgentError):
    """LLM inference or function calling errors."""


class WhatsAppError(VoiceAgentError):
    """WhatsApp message delivery errors."""


class CalendarError(VoiceAgentError):
    """Google Calendar event creation errors."""


class StateError(VoiceAgentError):
    """Redis call state errors."""


class WebhookAuthError(VoiceAgentError):
    """Webhook signature verification failure."""


class ConfigurationError(VoiceAgentError):
    """Missing or invalid configuration."""
