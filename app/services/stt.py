"""
Speech-to-Text service — Sarvam AI (primary), with language detection.

Sarvam's saaras:v2 model handles:
  - Telugu, Hindi, English natively
  - Code-mixed speech (Tenglish, Hinglish) in the same utterance
  - 8kHz telephony-grade audio (exactly what Exotel/Vapi delivers)
  - Returns language tag alongside transcript

This module is used by Vapi as a custom STT provider via webhook,
and can also be called standalone for testing.
"""
from __future__ import annotations

import base64
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import STTError
from app.core.logging import get_logger
from app.models.call import Language

logger = get_logger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Map Sarvam language codes to our internal Language enum
SARVAM_LANG_MAP: dict[str, Language] = {
    "te-IN": Language.TELUGU,
    "hi-IN": Language.HINDI,
    "en-IN": Language.ENGLISH,
    "en-US": Language.ENGLISH,
    "en-GB": Language.ENGLISH,
}


class SarvamSTTClient:
    """
    Async Sarvam AI speech-to-text client.
    Sarvam's API accepts raw audio files (WAV/MP3/FLAC/OGG).
    For real-time streaming via Vapi, audio chunks are accumulated
    and transcribed per utterance.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            headers={
                "api-subscription-key": self._settings.sarvam_api_key,
            },
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(STTError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        reraise=True,
    )
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Language | None = None,
        audio_format: str = "wav",
    ) -> tuple[str, Language]:
        """
        Transcribe audio bytes to text.

        Returns:
            (transcript: str, detected_language: Language)

        audio_format: "wav" | "mp3" | "flac" | "ogg"
        language_hint: If we already know the language, pass it to improve accuracy.
                       If None, Sarvam auto-detects.
        """
        if not audio_bytes:
            return "", Language.UNKNOWN

        # Build language code for Sarvam
        lang_code = _language_to_sarvam_code(language_hint) if language_hint else None

        files: dict[str, Any] = {
            "file": (f"audio.{audio_format}", audio_bytes, f"audio/{audio_format}"),
        }
        data: dict[str, str] = {
            "model": self._settings.sarvam_stt_model,
            "with_diarization": "false",
        }
        if lang_code:
            data["language_code"] = lang_code

        logger.debug(
            "transcribing_audio",
            audio_size_bytes=len(audio_bytes),
            language_hint=lang_code,
        )

        try:
            response = await self._client.post(
                SARVAM_STT_URL,
                files=files,
                data=data,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise STTError(
                f"Sarvam STT API error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise STTError(f"Sarvam STT connection error: {exc}") from exc

        result = response.json()
        transcript: str = result.get("transcript", "").strip()
        lang_tag: str = result.get("language_code", "")
        detected = SARVAM_LANG_MAP.get(lang_tag, Language.UNKNOWN)

        logger.debug(
            "transcription_complete",
            transcript_length=len(transcript),
            detected_language=detected,
        )

        return transcript, detected

    async def transcribe_base64(
        self,
        audio_b64: str,
        language_hint: Language | None = None,
        audio_format: str = "wav",
    ) -> tuple[str, Language]:
        """Convenience wrapper for base64-encoded audio (common in webhooks)."""
        audio_bytes = base64.b64decode(audio_b64)
        return await self.transcribe(audio_bytes, language_hint, audio_format)


def _language_to_sarvam_code(lang: Language) -> str:
    """Map our internal Language enum to Sarvam's BCP-47 codes."""
    mapping = {
        Language.TELUGU: "te-IN",
        Language.HINDI: "hi-IN",
        Language.ENGLISH: "en-IN",
    }
    return mapping.get(lang, "en-IN")


# ── Singleton ─────────────────────────────────────────────────────────────────

_stt_client: SarvamSTTClient | None = None


def get_stt_client() -> SarvamSTTClient:
    global _stt_client
    if _stt_client is None:
        _stt_client = SarvamSTTClient()
    return _stt_client
