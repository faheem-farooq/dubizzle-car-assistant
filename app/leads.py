"""Appends qualified leads to a local CSV file, per the assignment's explicit
requirement to simulate lead recording separate from the long-term SQLite store."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LEADS_PATH = Path(__file__).resolve().parent.parent / "leads.csv"

FIELDNAMES = [
    "timestamp",
    "user_id",
    "price_min",
    "price_max",
    "notes",
    "car_of_interest_id",
    "booked_slot",
]


def append_lead(
    user_id: str,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    notes: Optional[str] = None,
    car_of_interest_id: Optional[int] = None,
    booked_slot: Optional[str] = None,
) -> None:
    is_new = not LEADS_PATH.exists()
    with open(LEADS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "price_min": price_min,
                "price_max": price_max,
                "notes": notes,
                "car_of_interest_id": car_of_interest_id,
                "booked_slot": booked_slot,
            }
        )
