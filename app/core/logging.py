"""
Structured logging setup using structlog.
Every log line is JSON in production, human-readable in development.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_call_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Pull call_id from context vars if present."""
    from app.core.context import get_call_id

    call_id = get_call_id()
    if call_id:
        event_dict["call_id"] = call_id
    return event_dict


def configure_logging(log_level: str = "INFO", is_production: bool = True) -> None:
    """
    Configure structlog for the application.
    - Production: JSON output (machine-parseable, works with Railway/Datadog)
    - Development: colored console output
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_call_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libs log through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(log_level),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
