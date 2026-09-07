"""
Context variables — thread/async-safe per-request state.
call_id flows through every async task in a request automatically.
"""
from __future__ import annotations

from contextvars import ContextVar

_call_id_var: ContextVar[str | None] = ContextVar("call_id", default=None)


def set_call_id(call_id: str) -> None:
    _call_id_var.set(call_id)


def get_call_id() -> str | None:
    return _call_id_var.get()
