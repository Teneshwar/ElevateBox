"""
Text-to-Speech service.

Primary:  Sarvam AI bulbul:v1  — Indian languages, ₹30/10K chars, sub-250ms
Fallback: ElevenLabs Turbo v2.5 — 75ms latency, high expressiveness

Language routing:
  Telugu  → Sarvam (te-IN female voice) / ElevenLabs Telugu voice
  Hindi   → Sarvam (hi-IN female voice) / ElevenLabs Hindi voice
  English → Sarvam (en-IN female voice) / ElevenLabs English (Indian accent)

Voice selection note:
  The task document explicitly notes a female voice reduces hang-ups on
  Indian outbound calls. All voices are configured as female.
"""
from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import TTSError
from app.core.logging import get_logger
from app.models.call import Language

logger = get_logger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"


# ─── Sarvam TTS ───────────────────────────────────────────────────────────────


class SarvamTTSClient:
    """
    Sarvam AI TTS — primary provider for all Indian languages.
    Returns raw audio bytes (WAV/MP3) ready to stream into the call.
    """

    # Female speaker IDs per language (Sarvam's built-in speakers)
    SPEAKER_MAP: dict[Language, str] = {
        Language.TELUGU: "anushka",   # Female Telugu speaker
        Language.HINDI: "anushka",    # Female Hindi speaker
        Language.ENGLISH: "anushka",  # Female English (Indian accent)
        Language.UNKNOWN: "anushka",
    }

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            headers={
                "api-subscription-key": self._settings.sarvam_api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(TTSError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        reraise=True,
    )
    async def synthesize(
        self,
        text: str,
        language: Language = Language.ENGLISH,
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesize text to speech.

        Returns:
            Raw audio bytes (WAV format, 22050 Hz, mono).

        speed: 0.5 – 2.0 (1.0 = normal). Slightly slower (0.9) works better
               on Indian phone lines.
        """
        if not text.strip():
            return b""

        lang_code = _language_to_sarvam_tts_code(language)
        speaker = self.SPEAKER_MAP.get(language, "anushka")

        payload = {
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "pitch": 0,
            "pace": speed,
            "loudness": 1.5,
            "speech_sample_rate": 8000,   # 8kHz for telephony
            "enable_preprocessing": True,  # Normalise numbers, dates, etc.
            "model": self._settings.sarvam_tts_model,
        }

        logger.debug("synthesizing_speech", text_length=len(text), language=language)

        try:
            response = await self._client.post(SARVAM_TTS_URL, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TTSError(
                f"Sarvam TTS API error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise TTSError(f"Sarvam TTS connection error: {exc}") from exc

        result = response.json()
        # Sarvam returns base64-encoded audio in audios[0]
        import base64
        audio_b64: str = result.get("audios", [""])[0]
        if not audio_b64:
            raise TTSError("Sarvam TTS returned empty audio")

        return base64.b64decode(audio_b64)


# ─── ElevenLabs TTS (fallback) ────────────────────────────────────────────────


class ElevenLabsTTSClient:
    """
    ElevenLabs Turbo v2.5 — fallback / high-expressiveness alternative.
    Used when Sarvam is unavailable or for specific emotional moments.
    """

    MODEL_ID = "eleven_turbo_v2_5"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            headers={
                "xi-api-key": self._settings.elevenlabs_api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _get_voice_id(self, language: Language) -> str:
        mapping = {
            Language.TELUGU: self._settings.elevenlabs_voice_id_telugu,
            Language.HINDI: self._settings.elevenlabs_voice_id_hindi,
            Language.ENGLISH: self._settings.elevenlabs_voice_id_english,
        }
        return mapping.get(language, self._settings.elevenlabs_voice_id_english)

    @retry(
        retry=retry_if_exception_type(TTSError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        reraise=True,
    )
    async def synthesize(
        self,
        text: str,
        language: Language = Language.ENGLISH,
    ) -> bytes:
        """Returns MP3 audio bytes."""
        if not text.strip():
            return b""

        voice_id = self._get_voice_id(language)
        payload = {
            "text": text,
            "model_id": self.MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True,
            },
            "output_format": "pcm_8000",  # 8kHz PCM for telephony
        }

        try:
            response = await self._client.post(
                f"{ELEVENLABS_TTS_URL}/{voice_id}",
                json=payload,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            raise TTSError(
                f"ElevenLabs TTS error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise TTSError(f"ElevenLabs TTS connection error: {exc}") from exc


# ─── Unified TTS facade ───────────────────────────────────────────────────────


class TTSService:
    """
    Unified TTS entry point with automatic fallback.
    Try Sarvam first; fall back to ElevenLabs on failure.
    """

    def __init__(self) -> None:
        self._sarvam = SarvamTTSClient()
        self._elevenlabs = ElevenLabsTTSClient()

    async def synthesize(
        self,
        text: str,
        language: Language = Language.ENGLISH,
    ) -> bytes:
        try:
            return await self._sarvam.synthesize(text, language)
        except TTSError as primary_err:
            logger.warning(
                "sarvam_tts_failed_falling_back",
                error=str(primary_err),
                language=language,
            )
            return await self._elevenlabs.synthesize(text, language)

    async def close(self) -> None:
        await self._sarvam.close()
        await self._elevenlabs.close()


def _language_to_sarvam_tts_code(language: Language) -> str:
    mapping = {
        Language.TELUGU: "te-IN",
        Language.HINDI: "hi-IN",
        Language.ENGLISH: "en-IN",
        Language.UNKNOWN: "en-IN",
    }
    return mapping.get(language, "en-IN")


# ── Singleton ─────────────────────────────────────────────────────────────────

_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
