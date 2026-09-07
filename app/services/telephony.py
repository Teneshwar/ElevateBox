"""
Exotel telephony service.

Responsibilities:
- Place outbound calls to the target number via Exotel REST API
- The call is bridged into Vapi via SIP trunking so Vapi handles all
  voice AI logic (STT → LLM → TTS) while Exotel provides the +91 caller ID
- Retrieve call status / recording metadata after the call ends

Exotel outbound flow:
  POST /v1/Accounts/{sid}/Calls/connect
  → Exotel dials target_number
  → On answer, connects to Vapi SIP trunk
  → Vapi runs the voice agent
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import TelephonyError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ExotelClient:
    """
    Async Exotel REST client.
    Uses HTTP Basic Auth (api_key:api_token) as required by Exotel's API.
    A single shared httpx.AsyncClient is reused across calls for connection pooling.
    """

    _client: httpx.AsyncClient | None = None

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── HTTP client lifecycle ─────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                auth=(
                    self._settings.exotel_api_key,
                    self._settings.exotel_api_token,
                ),
                base_url=(
                    f"https://{self._settings.exotel_api_key}:{self._settings.exotel_api_token}"
                    f"@{self._settings.exotel_subdomain}/v1/Accounts"
                    f"/{self._settings.exotel_account_sid}"
                ),
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Core: place outbound call ─────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(TelephonyError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    async def place_outbound_call(
        self,
        to_number: str,
        vapi_call_id: str,
    ) -> dict[str, Any]:
        """
        Dial `to_number` from our Exophone.
        On answer, bridge the call into Vapi via Exotel's SIP URL passthrough.

        Exotel's Calls/connect API:
          From    = our Exophone (caller ID shown to recipient)
          To      = target mobile (+91xxxxxxxxxx)
          CallerId= same as From (required field)
          Url     = Exotel app URL that controls call flow (we use our webhook)
                    OR SIP trunking endpoint from Vapi

        For Vapi SIP trunking the approach is:
          - Create an Exotel applet that plays nothing and connects to
            our FastAPI /exotel/bridge endpoint
          - That endpoint returns TwiML-style ExoML to forward audio to Vapi

        Alternatively (simpler): use Exotel's "Connect to SIP" applet pointing
        to Vapi's SIP URI. Both methods are supported; we use the webhook bridge
        for maximum control and observability.
        """
        settings = self._settings
        client = await self._get_client()

        # Exotel ExoML app that answers and immediately connects to our
        # /api/v1/exotel/bridge endpoint which returns the SIP forward XML.
        exoml_url = f"{settings.app_base_url}/api/v1/exotel/bridge/{vapi_call_id}"

        payload = {
            "From": settings.exotel_caller_id,
            "To": to_number,
            "CallerId": settings.exotel_caller_id,
            "Url": exoml_url,
            "StatusCallback": f"{settings.app_base_url}/api/v1/exotel/status/{vapi_call_id}",
            "StatusCallbackEvents[0]": "terminal",
            "Record": "false",
        }

        logger.info(
            "placing_outbound_call",
            to=to_number,
            caller_id=settings.exotel_caller_id,
            vapi_call_id=vapi_call_id,
        )

        try:
            response = await client.post("/Calls/connect.json", data=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TelephonyError(
                f"Exotel API error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise TelephonyError(
                f"Exotel connection error: {exc}",
            ) from exc

        data: dict[str, Any] = response.json()
        call_data = data.get("Call", data)

        logger.info(
            "outbound_call_placed",
            exotel_sid=call_data.get("Sid"),
            status=call_data.get("Status"),
            to=to_number,
        )

        return call_data

    # ── ExoML response for bridge ─────────────────────────────────────────────

    def build_sip_bridge_exoml(self, vapi_sip_uri: str) -> str:
        """
        Return ExoML XML that Exotel fetches when the callee answers.
        This forwards the live audio to Vapi's SIP endpoint so Vapi can
        run the voice AI agent.

        vapi_sip_uri format: sip:<assistant_id>@sip.vapi.ai
        """
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Sip>{vapi_sip_uri}</Sip>
    </Dial>
</Response>"""

    # ── Call status ───────────────────────────────────────────────────────────

    async def get_call_status(self, exotel_sid: str) -> dict[str, Any]:
        """Fetch live status of an Exotel call by its SID."""
        client = await self._get_client()
        try:
            response = await client.get(f"/Calls/{exotel_sid}.json")
            response.raise_for_status()
            return response.json().get("Call", {})
        except httpx.HTTPStatusError as exc:
            raise TelephonyError(
                f"Failed to fetch call status: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc

    async def end_call(self, exotel_sid: str) -> None:
        """Programmatically end an active call."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"/Calls/{exotel_sid}.json",
                data={"Status": "completed"},
            )
            response.raise_for_status()
            logger.info("call_ended_programmatically", exotel_sid=exotel_sid)
        except httpx.HTTPStatusError as exc:
            raise TelephonyError(
                f"Failed to end call: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc


# ── Module-level singleton ────────────────────────────────────────────────────

_exotel_client: ExotelClient | None = None


def get_exotel_client() -> ExotelClient:
    global _exotel_client
    if _exotel_client is None:
        _exotel_client = ExotelClient()
    return _exotel_client
