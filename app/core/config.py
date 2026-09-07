"""
Central configuration — all settings loaded from environment variables.
Uses pydantic-settings for type validation and automatic .env loading.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["production", "development"] = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_base_url: str = Field(..., description="Public base URL of this server")
    webhook_secret: str = Field(..., description="HMAC secret for webhook verification")

    # ── Lead / Developer ─────────────────────────────────────────────────────
    target_phone_number: str = Field(..., description="E.164 phone number to call")
    developer_phone_number: str = Field(..., description="Your phone number for WhatsApp")
    developer_name: str = Field(..., description="Your full name")

    # ── Exotel ───────────────────────────────────────────────────────────────
    exotel_api_key: str
    exotel_api_token: str
    exotel_account_sid: str
    exotel_subdomain: str = "api.exotel.com"
    exotel_caller_id: str = Field(..., description="+91 Exophone number")

    # ── Vapi ─────────────────────────────────────────────────────────────────
    vapi_api_key: str
    vapi_phone_number_id: str
    vapi_webhook_secret: str

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # ── Sarvam AI ────────────────────────────────────────────────────────────
    sarvam_api_key: str
    sarvam_stt_model: str = "saaras:v2"
    sarvam_tts_model: str = "bulbul:v1"

    # ── ElevenLabs ───────────────────────────────────────────────────────────
    elevenlabs_api_key: str
    elevenlabs_voice_id_hindi: str
    elevenlabs_voice_id_telugu: str
    elevenlabs_voice_id_english: str

    # ── WhatsApp ─────────────────────────────────────────────────────────────
    whatsapp_access_token: str
    whatsapp_phone_number_id: str
    whatsapp_business_account_id: str
    whatsapp_api_version: str = "v20.0"

    # ── Google Calendar ──────────────────────────────────────────────────────
    google_calendar_id: str = "primary"
    google_service_account_json: str = Field(
        default="", description="Raw JSON string of service account credentials"
    )
    google_service_account_file: str = Field(
        default="", description="Path to service account key file (alternative to JSON string)"
    )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Assets ───────────────────────────────────────────────────────────────
    resume_pdf_url: str = Field(..., description="Publicly accessible URL of resume PDF")
    architecture_image_url: str = Field(
        ..., description="Publicly accessible URL of architecture image"
    )

    # ── Rate Limiting ────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Computed properties ──────────────────────────────────────────────────
    @property
    def whatsapp_api_base(self) -> str:
        return f"https://graph.facebook.com/{self.whatsapp_api_version}"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def vapi_webhook_url(self) -> str:
        return f"{self.app_base_url}/api/v1/webhooks/vapi"

    @property
    def google_credentials_dict(self) -> dict | None:
        """Return parsed service account dict, preferring JSON string over file."""
        if self.google_service_account_json:
            return json.loads(self.google_service_account_json)
        return None

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("target_phone_number", "developer_phone_number", "exotel_caller_id")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("+"):
            raise ValueError(f"Phone number must be in E.164 format (start with +): {v}")
        digits = v[1:].replace(" ", "")
        if not digits.isdigit() or not (7 <= len(digits) <= 15):
            raise ValueError(f"Invalid E.164 phone number: {v}")
        return v

    @model_validator(mode="after")
    def validate_google_credentials(self) -> "Settings":
        if not self.google_service_account_json and not self.google_service_account_file:
            raise ValueError(
                "Either GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE must be set"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    Called once at startup; subsequent calls return the same object.
    """
    return Settings()  # type: ignore[call-arg]
