"""
Callback scheduling service.

Converts natural language time expressions into real datetime objects
and creates Google Calendar events.

Flow:
  "call me back tomorrow morning"
    → dateparser.parse("tomorrow morning", IST timezone)
    → 2026-09-03 09:00:00+05:30
    → Google Calendar event created
    → Returns event link for confirmation

Handles vague expressions:
  "tomorrow morning"   → next day 09:00 IST
  "kal subah"          → next day 09:00 IST
  "Friday afternoon"   → next Friday 14:00 IST
  "after 3"            → today or tomorrow 15:00 IST
  "next week"          → next Monday 10:00 IST
  "evening"            → today 18:00 IST (or tomorrow if already past)

Google Calendar auth uses a Service Account so no OAuth browser
flow is required — works in production without user intervention.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import dateparser
import pytz
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import CalendarError
from app.core.logging import get_logger

logger = get_logger(__name__)

IST = pytz.timezone("Asia/Kolkata")
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Business hours defaults for vague expressions
MORNING_HOUR = 9
AFTERNOON_HOUR = 14
EVENING_HOUR = 18
DEFAULT_HOUR = 10

# Hindi/Telugu time word mappings → English equivalents for dateparser
_HINGLISH_TIME_MAP: dict[str, str] = {
    "subah": "morning",
    "dopahar": "afternoon",
    "shaam": "evening",
    "raat": "night",
    "kal": "tomorrow",
    "parso": "day after tomorrow",
    "aaj": "today",
    "agli": "next",
    "next week": "next week",
    # Telugu
    "repu": "tomorrow",
    "modnati": "morning",
    "maadhyahnam": "afternoon",
    "saayantram": "evening",
    "ee vaaram": "this week",
    "tarvata vaaram": "next week",
}


def _normalise_time_expression(raw: str) -> str:
    """
    Transliterate Hindi/Telugu time words to English so dateparser can handle them.
    Also normalises common Hinglish patterns.
    """
    normalised = raw.lower().strip()
    for vernacular, english in _HINGLISH_TIME_MAP.items():
        normalised = re.sub(r"\b" + vernacular + r"\b", english, normalised)
    return normalised


def _apply_business_hours(dt: datetime) -> datetime:
    """
    If the parsed time has no specific hour (midnight / 00:00),
    default to 09:00 business hours.
    Also push to next business day if time has already passed.
    """
    now_ist = datetime.now(IST)

    if dt.hour == 0 and dt.minute == 0:
        dt = dt.replace(hour=DEFAULT_HOUR, minute=0, second=0)

    # Map vague period words that dateparser resolves to midnight
    # (dateparser sometimes returns midnight for "morning")
    # Re-read the original expression for clues — handled in parse_callback_time

    # If the resolved time is in the past, push to next day same time
    if dt <= now_ist:
        dt = dt + timedelta(days=1)
        logger.debug("callback_time_in_past_pushed_to_next_day", new_dt=dt.isoformat())

    return dt


def parse_callback_time(raw_expression: str) -> datetime:
    """
    Parse a natural language time expression to a datetime in IST.

    Raises CalendarError if the expression cannot be parsed.
    """
    normalised = _normalise_time_expression(raw_expression)
    now_ist = datetime.now(IST)

    # Inject period-of-day hours manually for reliable results
    hour_override: int | None = None
    if any(w in normalised for w in ("morning", "subah", "modnati")):
        hour_override = MORNING_HOUR
    elif any(w in normalised for w in ("afternoon", "dopahar", "maadhyahnam")):
        hour_override = AFTERNOON_HOUR
    elif any(w in normalised for w in ("evening", "shaam", "saayantram")):
        hour_override = EVENING_HOUR

    parsed = dateparser.parse(
        normalised,
        settings={
            "TIMEZONE": "Asia/Kolkata",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "PREFER_DAY_OF_MONTH": "first",
            "TO_TIMEZONE": "Asia/Kolkata",
        },
    )

    if parsed is None:
        # Last resort: if nothing works, default to tomorrow 09:00
        logger.warning(
            "dateparser_failed_using_default",
            raw_expression=raw_expression,
            normalised=normalised,
        )
        parsed = (now_ist + timedelta(days=1)).replace(
            hour=DEFAULT_HOUR, minute=0, second=0, microsecond=0
        )
    else:
        # Ensure timezone aware
        if parsed.tzinfo is None:
            parsed = IST.localize(parsed)

    # Apply hour override if we detected a time-of-day word
    if hour_override is not None:
        parsed = parsed.replace(hour=hour_override, minute=0, second=0, microsecond=0)

    # Apply business hours and past-time adjustment
    parsed = _apply_business_hours(parsed)

    logger.info(
        "callback_time_parsed",
        raw=raw_expression,
        normalised=normalised,
        resolved=parsed.isoformat(),
    )
    return parsed


# ─── Google Calendar ──────────────────────────────────────────────────────────


def _get_calendar_service() -> Any:
    """
    Build and return an authorised Google Calendar API service.
    Uses a service account — no user OAuth required.
    """
    settings = get_settings()

    if settings.google_credentials_dict:
        creds_info = settings.google_credentials_dict
    elif settings.google_service_account_file:
        with open(settings.google_service_account_file) as f:
            creds_info = json.load(f)
    else:
        raise CalendarError("No Google service account credentials configured")

    credentials = service_account.Credentials.from_service_account_info(
        creds_info, scopes=SCOPES
    )

    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
    return service


@retry(
    retry=retry_if_exception_type(CalendarError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
def create_callback_event(
    callback_dt: datetime,
    customer_name: str | None,
    phone_number: str,
    discovery_summary: str,
) -> dict[str, str]:
    """
    Create a Google Calendar event for the callback.
    Returns {"event_id": str, "event_link": str}.

    Note: This is synchronous (Google client library is sync).
    The webhook handler calls this in a thread pool executor.
    """
    settings = get_settings()
    service = _get_calendar_service()

    name_str = customer_name or "Potential Customer"
    duration_minutes = 30

    # Event body
    event: dict[str, Any] = {
        "summary": f"📞 ElevateBox Callback — {name_str}",
        "description": (
            f"Customer: {name_str}\n"
            f"Phone: {phone_number}\n\n"
            f"Call Summary:\n{discovery_summary}\n\n"
            f"Auto-scheduled by ElevateBox Voice Agent."
        ),
        "start": {
            "dateTime": callback_dt.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": (callback_dt + timedelta(minutes=duration_minutes)).isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 15},
                {"method": "email", "minutes": 30},
            ],
        },
        "colorId": "2",  # Green — sales callback
    }

    try:
        created = (
            service.events()
            .insert(calendarId=settings.google_calendar_id, body=event)
            .execute()
        )
    except HttpError as exc:
        raise CalendarError(
            f"Google Calendar API error: {exc.status_code}",
            detail=str(exc),
        ) from exc

    event_id: str = created.get("id", "")
    event_link: str = created.get("htmlLink", "")

    logger.info(
        "calendar_event_created",
        event_id=event_id,
        callback_dt=callback_dt.isoformat(),
        customer=name_str,
    )

    return {"event_id": event_id, "event_link": event_link}


async def schedule_callback(
    time_expression: str,
    customer_name: str | None,
    phone_number: str,
    discovery_summary: str,
) -> dict[str, str]:
    """
    Async entry point:
    1. Parse the time expression
    2. Create the calendar event in a thread pool (sync SDK)
    3. Return event details
    """
    import asyncio

    callback_dt = parse_callback_time(time_expression)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: create_callback_event(
            callback_dt, customer_name, phone_number, discovery_summary
        ),
    )
    result["callback_datetime_ist"] = callback_dt.isoformat()
    return result
