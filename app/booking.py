"""Test-drive slot validation: Monday-Saturday, 8am-8pm only."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from app.retrieval import get_listing_by_id

_ALLOWED_DAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
}

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _parse_day(requested_day: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (canonical_day_name, error) - canonical_day_name is None on failure."""
    raw = requested_day.strip()

    lowered = raw.lower()
    for name, _idx in _ALLOWED_DAYS.items():
        if lowered == name or lowered == name[:3]:
            return name.capitalize(), None

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            weekday_name = dt.strftime("%A")
            if weekday_name.lower() == "sunday":
                return None, "That date falls on a Sunday, and we're closed Sundays. Please pick Monday-Saturday."
            return weekday_name, None
        except ValueError:
            continue

    if lowered == "sunday" or lowered == "sun":
        return None, "We're closed on Sundays. Please pick a day between Monday and Saturday."

    return None, f"Could not understand the requested day '{requested_day}'. Please use a weekday name (e.g. 'Wednesday') or a date (YYYY-MM-DD)."


def _parse_time(requested_time: str) -> tuple[Optional[str], Optional[str]]:
    match = _TIME_RE.match(requested_time.strip())
    if not match:
        return None, f"Could not understand the requested time '{requested_time}'. Please use 24h HH:MM format (e.g. '14:30')."
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None, f"'{requested_time}' is not a valid time."
    if hour < 8 or hour >= 20:
        return None, f"'{requested_time}' is outside our viewing hours (8:00-20:00)."
    return f"{hour:02d}:{minute:02d}", None


def book_test_drive(listing_id: int, requested_day: str, requested_time: str, user_id: str) -> dict:
    listing = get_listing_by_id(listing_id)
    if listing is None:
        return {
            "success": False,
            "message": f"No listing found with ID {listing_id}.",
            "listing_id": listing_id,
            "day": None,
            "time": None,
        }

    day, day_err = _parse_day(requested_day)
    if day_err:
        return {"success": False, "message": day_err, "listing_id": listing_id, "day": None, "time": None}

    time_str, time_err = _parse_time(requested_time)
    if time_err:
        return {"success": False, "message": time_err, "listing_id": listing_id, "day": day, "time": None}

    title = listing.get("title", f"Listing {listing_id}")
    return {
        "success": True,
        "message": f"Test drive booked for '{title}' on {day} at {time_str}.",
        "listing_id": listing_id,
        "day": day,
        "time": time_str,
    }
