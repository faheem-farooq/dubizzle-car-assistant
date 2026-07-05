"""Two memory tiers:

- Long-term: SQLite (users, preferences) - persists across process restarts,
  keyed by user_id, so a returning user is recognized in a brand new session.
- Short-term: in-process dict keyed by session_id - holds the raw chat
  message history replayed into the LLM each turn within one conversation.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                price_min REAL,
                price_max REAL,
                body_type TEXT,
                make_pref TEXT,
                notes TEXT,
                car_of_interest_id INTEGER,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            """
        )
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def touch_user(user_id: str, name: Optional[str] = None) -> None:
    """Create the user if new, or update last_seen (and name, if newly provided)."""
    now = _now()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT user_id, name FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (user_id, name, created_at, last_seen) VALUES (?, ?, ?, ?)",
                (user_id, name, now, now),
            )
        else:
            new_name = name if name else row["name"]
            conn.execute(
                "UPDATE users SET last_seen = ?, name = ? WHERE user_id = ?",
                (now, new_name, user_id),
            )
        conn.commit()


def get_user_profile(user_id: str) -> Optional[dict]:
    with _lock, _connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if user is None:
            return None
        pref = conn.execute(
            "SELECT * FROM preferences WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()

    profile = dict(user)
    if pref:
        profile.update(
            {
                "price_min": pref["price_min"],
                "price_max": pref["price_max"],
                "body_type": pref["body_type"],
                "make_pref": pref["make_pref"],
                "notes": pref["notes"],
                "car_of_interest_id": pref["car_of_interest_id"],
                "updated_at": pref["updated_at"],
            }
        )
    return profile


def upsert_preferences(
    user_id: str,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    body_type: Optional[str] = None,
    make_pref: Optional[str] = None,
    notes: Optional[str] = None,
    car_of_interest_id: Optional[int] = None,
) -> None:
    """Merge the given fields into the user's latest known preferences and
    append a new row with the merged result. A caller that only knows one
    field (e.g. book_test_drive only knows car_of_interest_id) must not wipe
    out fields a previous call already captured (e.g. price_min/body_type)."""
    touch_user(user_id)
    now = _now()
    with _lock, _connect() as conn:
        existing = conn.execute(
            "SELECT * FROM preferences WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()

        def merged(new_value, field: str):
            if new_value is not None:
                return new_value
            return existing[field] if existing else None

        conn.execute(
            """
            INSERT INTO preferences
                (user_id, price_min, price_max, body_type, make_pref, notes, car_of_interest_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                merged(price_min, "price_min"),
                merged(price_max, "price_max"),
                merged(body_type, "body_type"),
                merged(make_pref, "make_pref"),
                merged(notes, "notes"),
                merged(car_of_interest_id, "car_of_interest_id"),
                now,
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Short-term (in-memory, per-session) conversation history
# ---------------------------------------------------------------------------

_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()


def get_history(session_id: str) -> list[dict]:
    with _sessions_lock:
        return list(_sessions.get(session_id, []))


def append_message(session_id: str, role: str, content, tool_call_id: Optional[str] = None, name: Optional[str] = None) -> None:
    entry: dict = {"role": role}
    if content is not None:
        entry["content"] = content
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id
    if name:
        entry["name"] = name
    with _sessions_lock:
        _sessions.setdefault(session_id, []).append(entry)


def set_history(session_id: str, messages: list[dict]) -> None:
    with _sessions_lock:
        _sessions[session_id] = messages
