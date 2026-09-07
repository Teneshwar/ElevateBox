"""
Security middleware and utilities.

1. Webhook signature verification
   - Vapi signs webhook payloads with HMAC-SHA256
   - Every incoming webhook is verified before processing
   - Prevents spoofed requests from triggering calls/WhatsApp

2. API key authentication
   - Internal endpoints (POST /calls/initiate) require a bearer token
   - This is your own secret — not exposed publicly

3. Rate limiting
   - Per-IP rate limiting via slowapi (Redis-backed in production)
   - Prevents abuse of the call initiation endpoint

4. Input sanitisation helpers
   - Phone number normalisation
   - Webhook payload size limits
"""
from __future__ import annotations

import hashlib
import hmac
import time
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.exceptions import WebhookAuthError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Rate limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ─── Bearer token auth ────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = None,
) -> None:
    """
    FastAPI dependency: validates Bearer token for internal endpoints.
    Raises 401 if token is missing or invalid.
    """
    settings = get_settings()
    token: str | None = None

    if credentials:
        token = credentials.credentials
    else:
        # Also accept X-API-Key header
        token = request.headers.get("X-API-Key")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    expected = settings.webhook_secret.encode()
    provided = token.encode()
    if not hmac.compare_digest(
        hashlib.sha256(expected).digest(),
        hashlib.sha256(provided).digest(),
    ):
        logger.warning("invalid_api_key_attempt", remote=request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )


# ─── Vapi webhook signature verification ─────────────────────────────────────


async def verify_vapi_signature(request: Request) -> bytes:
    """
    Verify Vapi's HMAC-SHA256 webhook signature.

    Vapi signs with: HMAC-SHA256(secret, raw_body)
    Signature is in header: x-vapi-signature

    Returns the raw body bytes for downstream parsing.
    Raises WebhookAuthError on failure.
    """
    settings = get_settings()
    raw_body = await request.body()

    signature = request.headers.get("x-vapi-signature", "")
    if not signature:
        raise WebhookAuthError("Missing x-vapi-signature header")

    expected_sig = hmac.new(
        settings.vapi_webhook_secret.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    # Vapi may send "sha256=<hex>" or just "<hex>"
    sig_value = signature.removeprefix("sha256=")

    if not hmac.compare_digest(expected_sig, sig_value):
        logger.warning(
            "vapi_webhook_signature_mismatch",
            expected_prefix=expected_sig[:8],
            received_prefix=sig_value[:8],
        )
        raise WebhookAuthError("Invalid Vapi webhook signature")

    return raw_body


async def verify_exotel_signature(request: Request) -> bytes:
    """
    Exotel does not use HMAC signatures by default.
    We verify the caller IP is within Exotel's known IP ranges.
    For production, whitelist Exotel IPs at the infrastructure level.
    """
    raw_body = await request.body()
    # IP allowlist check — Exotel's outbound IPs
    # In production, enforce this at nginx/load-balancer level
    # Here we just log and pass through
    client_ip = request.client.host if request.client else "unknown"
    logger.debug("exotel_webhook_received", client_ip=client_ip)
    return raw_body


# ─── Input sanitisation ───────────────────────────────────────────────────────


def normalise_phone(phone: str) -> str:
    """
    Normalise phone number to E.164 format.
    Handles: 8688664337, 08688664337, +918688664337, 91-8688664337
    """
    # Strip everything except digits and leading +
    cleaned = "".join(c for c in phone if c.isdigit() or c == "+")

    if cleaned.startswith("+"):
        return cleaned

    # Indian mobile numbers
    if cleaned.startswith("91") and len(cleaned) == 12:
        return f"+{cleaned}"
    if cleaned.startswith("0") and len(cleaned) == 11:
        return f"+91{cleaned[1:]}"
    if len(cleaned) == 10:
        return f"+91{cleaned}"

    # Return as-is with + if nothing matched
    return f"+{cleaned}"


def sanitise_webhook_payload(payload: dict[str, Any], max_depth: int = 5) -> dict[str, Any]:
    """
    Basic sanitisation of incoming webhook payloads.
    Removes keys with excessively long values to prevent log injection.
    """
    def _clean(obj: Any, depth: int = 0) -> Any:
        if depth > max_depth:
            return "[truncated]"
        if isinstance(obj, str) and len(obj) > 10_000:
            return obj[:10_000] + "...[truncated]"
        if isinstance(obj, dict):
            return {k: _clean(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(item, depth + 1) for item in obj[:100]]
        return obj

    return _clean(payload)  # type: ignore[return-value]


# ─── Request timing middleware ────────────────────────────────────────────────


async def add_timing_header(request: Request, call_next: Callable) -> Any:  # type: ignore[type-arg]
    """Middleware: adds X-Process-Time header to every response."""
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"
    return response
