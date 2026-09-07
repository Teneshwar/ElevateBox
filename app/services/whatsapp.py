"""
WhatsApp messaging service — Meta Cloud API.

Handles two distinct message types:

1. MID-CALL message (fires while call is still live):
   - Triggered the moment HOT intent is detected
   - Contains: brief intro + price range + website + developer number
   - Non-blocking async task so it doesn't pause the voice conversation

2. POST-CALL message (fires when call ends):
   - Contains: personalised call summary (their exact words)
   - Resume PDF attachment
   - Architecture image attachment
   - Developer's phone number

Meta Cloud API pricing (India, 2026):
  - Marketing template: ~$0.0118/message
  - Service (within 24h window): free
  We use a pre-approved marketing template for the mid-call message
  since the customer has not messaged us first.
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
from app.core.exceptions import WhatsAppError
from app.core.logging import get_logger
from app.models.call import Language

logger = get_logger(__name__)


# ─── Message templates ────────────────────────────────────────────────────────

MID_CALL_MESSAGES: dict[Language, str] = {
    Language.ENGLISH: (
        "Hi! This is Priya from *ElevateBox* 👋\n\n"
        "As promised on our call, here are our details:\n\n"
        "🛒 *E-commerce Website Development*\n"
        "💰 Starting from ₹25,000 — fully customised\n"
        "⚡ Live in 2–4 weeks\n"
        "✅ Mobile-first | Payment gateway | Admin panel\n\n"
        "📞 Call us directly: {developer_phone}\n\n"
        "We'd love to build something great for your business!"
    ),
    Language.HINDI: (
        "Namaste! Main Priya hoon *ElevateBox* se 👋\n\n"
        "Jaise humne call mein baat ki, yahan details hain:\n\n"
        "🛒 *E-commerce Website Development*\n"
        "💰 ₹25,000 se shuru — poori tarah customised\n"
        "⚡ 2–4 hafte mein ready\n"
        "✅ Mobile-first | Payment gateway | Admin panel\n\n"
        "📞 Directly call karein: {developer_phone}\n\n"
        "Aapka business online lana humein bahut achha lagega!"
    ),
    Language.TELUGU: (
        "Namaskaram! Nenu Priya, *ElevateBox* nundi 👋\n\n"
        "Mana call lo cheppinattuga, ikkada details unnaayi:\n\n"
        "🛒 *E-commerce Website Development*\n"
        "💰 ₹25,000 nundi — complete ga customised\n"
        "⚡ 2–4 weeks lo ready\n"
        "✅ Mobile-first | Payment gateway | Admin panel\n\n"
        "📞 Direct ga call cheyyandi: {developer_phone}\n\n"
        "Meeru business ni online lo pettadaaniki maaku chaalaa santhosham!"
    ),
    Language.UNKNOWN: (
        "Hi! This is Priya from *ElevateBox* 👋\n\n"
        "As discussed on our call:\n\n"
        "🛒 *E-commerce Website Development*\n"
        "💰 Starting from ₹25,000\n"
        "⚡ Live in 2–4 weeks\n\n"
        "📞 {developer_phone}\n\n"
        "Looking forward to building your website!"
    ),
}


class WhatsAppClient:
    """
    Meta WhatsApp Cloud API client.
    All methods are async and use connection pooling via a shared httpx client.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    @property
    def _base_url(self) -> str:
        s = self._settings
        return f"{s.whatsapp_api_base}/{s.whatsapp_phone_number_id}"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.whatsapp_access_token}",
            "Content-Type": "application/json",
        }

    async def close(self) -> None:
        await self._client.aclose()

    # ── Core send ─────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(WhatsAppError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _send(self, payload: dict[str, Any]) -> str:
        """
        POST to /messages. Returns message_id on success.
        """
        try:
            response = await self._client.post(
                f"{self._base_url}/messages",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise WhatsAppError(
                f"WhatsApp API error: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise WhatsAppError(f"WhatsApp connection error: {exc}") from exc

        data = response.json()
        messages = data.get("messages", [])
        message_id: str = messages[0].get("id", "") if messages else ""
        logger.info("whatsapp_message_sent", message_id=message_id)
        return message_id

    # ── Upload media ──────────────────────────────────────────────────────────

    async def _upload_media_from_url(self, media_url: str, mime_type: str) -> str:
        """
        Download a file from a public URL and upload it to Meta's media store.
        Returns the media_id for use in messages.

        Meta requires media to be uploaded to their servers to send as attachments.
        """
        # Download the file
        try:
            dl_response = await self._client.get(media_url, follow_redirects=True)
            dl_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WhatsAppError(f"Failed to download media from {media_url}: {exc}") from exc

        file_bytes = dl_response.content
        filename = media_url.split("/")[-1] or "attachment"

        # Upload to Meta
        try:
            upload_response = await self._client.post(
                f"{self._base_url}/media",
                headers={"Authorization": f"Bearer {self._settings.whatsapp_access_token}"},
                files={
                    "file": (filename, file_bytes, mime_type),
                    "type": (None, mime_type),
                    "messaging_product": (None, "whatsapp"),
                },
            )
            upload_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise WhatsAppError(
                f"Media upload failed: HTTP {exc.response.status_code}",
                detail=exc.response.text,
            ) from exc

        media_id: str = upload_response.json().get("id", "")
        logger.info("media_uploaded", media_id=media_id, mime_type=mime_type)
        return media_id

    # ── Mid-call message ──────────────────────────────────────────────────────

    async def send_mid_call_message(
        self,
        to_number: str,
        language: Language,
        context_summary: str,
    ) -> str:
        """
        Send the mid-call WhatsApp message.
        This is the HOT-intent trigger message — fires while call is live.

        Returns message_id.
        """
        settings = self._settings
        template = MID_CALL_MESSAGES.get(language, MID_CALL_MESSAGES[Language.UNKNOWN])
        body = template.format(developer_phone=settings.developer_phone_number)

        # Append context summary if meaningful
        if context_summary and len(context_summary.strip()) > 10:
            body += f"\n\n_Your requirement: {context_summary.strip()}_"

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": body,
            },
        }

        logger.info(
            "sending_mid_call_whatsapp",
            to=to_number,
            language=language,
            body_length=len(body),
        )

        return await self._send(payload)

    # ── Post-call message ─────────────────────────────────────────────────────

    async def send_post_call_message(
        self,
        to_number: str,
        personalised_summary: str,
        language: Language,
    ) -> str:
        """
        Send the post-call WhatsApp text message (personalised follow-up).
        Returns message_id.
        """
        settings = self._settings

        # Add developer contact at the end
        footer = (
            f"\n\n---\n"
            f"📞 *{settings.developer_name}*\n"
            f"Mobile: {settings.developer_phone_number}\n"
            f"ElevateBox — Banjara Hills, Hyderabad"
        )

        body = personalised_summary + footer

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": body,
            },
        }

        logger.info("sending_post_call_text_whatsapp", to=to_number)
        return await self._send(payload)

    async def send_resume(self, to_number: str) -> str:
        """
        Upload and send the resume PDF.
        Returns message_id.
        """
        settings = self._settings
        media_id = await self._upload_media_from_url(
            settings.resume_pdf_url, "application/pdf"
        )

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "document",
            "document": {
                "id": media_id,
                "caption": f"Resume — {settings.developer_name}",
                "filename": "Resume.pdf",
            },
        }

        logger.info("sending_resume_whatsapp", to=to_number)
        return await self._send(payload)

    async def send_architecture_image(self, to_number: str) -> str:
        """
        Upload and send the architecture diagram image.
        Returns message_id.
        """
        settings = self._settings
        media_id = await self._upload_media_from_url(
            settings.architecture_image_url, "image/png"
        )

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "image",
            "image": {
                "id": media_id,
                "caption": (
                    "🏗️ *System Architecture*\n"
                    "How the AI voice agent was built — "
                    "from outbound call to mid-call WhatsApp."
                ),
            },
        }

        logger.info("sending_architecture_image_whatsapp", to=to_number)
        return await self._send(payload)

    # ── Full post-call sequence ───────────────────────────────────────────────

    async def send_full_post_call_sequence(
        self,
        to_number: str,
        personalised_summary: str,
        language: Language,
    ) -> dict[str, str]:
        """
        Send all three post-call messages in sequence:
          1. Personalised text summary
          2. Resume PDF
          3. Architecture image

        Messages are sent sequentially with a short delay between each
        so they appear in logical order in WhatsApp.

        Returns dict of {message_type: message_id}.
        """
        results: dict[str, str] = {}

        # 1. Personalised summary text
        try:
            results["summary"] = await self.send_post_call_message(
                to_number, personalised_summary, language
            )
            await asyncio.sleep(0.5)
        except WhatsAppError as exc:
            logger.error("post_call_summary_send_failed", error=str(exc))

        # 2. Resume PDF
        try:
            results["resume"] = await self.send_resume(to_number)
            await asyncio.sleep(0.5)
        except WhatsAppError as exc:
            logger.error("resume_send_failed", error=str(exc))

        # 3. Architecture image
        try:
            results["architecture"] = await self.send_architecture_image(to_number)
        except WhatsAppError as exc:
            logger.error("architecture_image_send_failed", error=str(exc))

        logger.info(
            "post_call_sequence_complete",
            to=to_number,
            messages_sent=len(results),
        )
        return results


# ── Singleton ─────────────────────────────────────────────────────────────────

_whatsapp_client: WhatsAppClient | None = None


def get_whatsapp_client() -> WhatsAppClient:
    global _whatsapp_client
    if _whatsapp_client is None:
        _whatsapp_client = WhatsAppClient()
    return _whatsapp_client
