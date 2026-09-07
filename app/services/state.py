"""
Redis-backed call state manager.

Every active and completed call has a CallState object stored in Redis.
All state mutations go through this manager so the system is:
  - Stateless at the application layer (any instance can handle any webhook)
  - Resilient to process restarts mid-call
  - Observable (full history queryable)

Key schema:
  call:{call_id}          → JSON-serialised CallState   (TTL: 7 days)
  calls:active            → Redis Set of active call_ids
  calls:by_phone:{phone}  → Redis List of call_ids for a phone number

Serialisation: Pydantic model → JSON string → Redis string value
"""
from __future__ import annotations

import json
from datetime import datetime

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.exceptions import StateError
from app.core.logging import get_logger
from app.models.call import CallState, CallStatus

logger = get_logger(__name__)

_CALL_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
_ACTIVE_SET_KEY = "calls:active"


class CallStateManager:
    """
    Async Redis-backed manager for CallState objects.
    One instance is created at startup and injected via FastAPI dependency.
    """

    def __init__(self, redis_client: Redis) -> None:  # type: ignore[type-arg]
        self._redis = redis_client

    # ── Key helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _call_key(call_id: str) -> str:
        return f"call:{call_id}"

    @staticmethod
    def _phone_key(phone: str) -> str:
        # Normalise phone: strip spaces, ensure +91 prefix
        phone = phone.replace(" ", "").replace("-", "")
        return f"calls:by_phone:{phone}"

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(self, call_state: CallState) -> None:
        """Persist a new call state. Raises StateError if call_id already exists."""
        key = self._call_key(call_state.call_id)

        exists = await self._redis.exists(key)
        if exists:
            raise StateError(f"Call state already exists for call_id: {call_state.call_id}")

        pipe = self._redis.pipeline()
        pipe.setex(key, _CALL_TTL_SECONDS, call_state.model_dump_json())
        pipe.sadd(_ACTIVE_SET_KEY, call_state.call_id)
        pipe.lpush(self._phone_key(call_state.phone_number), call_state.call_id)
        await pipe.execute()

        logger.info("call_state_created", call_id=call_state.call_id)

    async def get(self, call_id: str) -> CallState | None:
        """Retrieve call state by call_id. Returns None if not found."""
        key = self._call_key(call_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return CallState.model_validate_json(raw)
        except Exception as exc:
            raise StateError(
                f"Failed to deserialise call state for {call_id}: {exc}"
            ) from exc

    async def get_or_raise(self, call_id: str) -> CallState:
        """Retrieve call state; raise StateError if not found."""
        state = await self.get(call_id)
        if state is None:
            raise StateError(f"No call state found for call_id: {call_id}")
        return state

    async def save(self, call_state: CallState) -> None:
        """Update an existing call state. Refreshes TTL."""
        call_state.updated_at = datetime.utcnow()
        key = self._call_key(call_state.call_id)
        await self._redis.setex(key, _CALL_TTL_SECONDS, call_state.model_dump_json())

    async def update_status(self, call_id: str, status: CallStatus) -> None:
        """Quick status update without loading/saving the full object."""
        state = await self.get_or_raise(call_id)
        state.status = status
        await self.save(state)

    async def mark_completed(self, call_id: str) -> None:
        """Mark call as completed and remove from active set."""
        state = await self.get_or_raise(call_id)
        state.status = CallStatus.COMPLETED
        await self.save(state)
        await self._redis.srem(_ACTIVE_SET_KEY, call_id)
        logger.info("call_marked_completed", call_id=call_id)

    async def mark_failed(self, call_id: str, reason: str = "") -> None:
        """Mark call as failed and remove from active set."""
        state = await self.get_or_raise(call_id)
        state.status = CallStatus.FAILED
        await self.save(state)
        await self._redis.srem(_ACTIVE_SET_KEY, call_id)
        logger.warning("call_marked_failed", call_id=call_id, reason=reason)

    # ── Queries ───────────────────────────────────────────────────────────────

    async def get_active_call_ids(self) -> list[str]:
        """Return all currently active call IDs."""
        members = await self._redis.smembers(_ACTIVE_SET_KEY)
        return [m.decode() if isinstance(m, bytes) else m for m in members]

    async def get_calls_for_phone(self, phone: str, limit: int = 10) -> list[CallState]:
        """Return recent call states for a phone number."""
        call_ids_raw = await self._redis.lrange(self._phone_key(phone), 0, limit - 1)
        states: list[CallState] = []
        for cid in call_ids_raw:
            cid_str = cid.decode() if isinstance(cid, bytes) else cid
            state = await self.get(cid_str)
            if state:
                states.append(state)
        return states

    async def health_check(self) -> bool:
        """Ping Redis; returns True if healthy."""
        try:
            return await self._redis.ping()
        except Exception:
            return False


# ── Redis client factory ──────────────────────────────────────────────────────

_redis_client: Redis | None = None  # type: ignore[type-arg]
_state_manager: CallStateManager | None = None


async def get_redis_client() -> Redis:  # type: ignore[type-arg]
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info("redis_client_created", url=settings.redis_url.split("@")[-1])
    return _redis_client


async def get_state_manager() -> CallStateManager:
    global _state_manager
    if _state_manager is None:
        redis_client = await get_redis_client()
        _state_manager = CallStateManager(redis_client)
    return _state_manager


async def close_redis() -> None:
    global _redis_client, _state_manager
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        _state_manager = None
        logger.info("redis_connection_closed")
