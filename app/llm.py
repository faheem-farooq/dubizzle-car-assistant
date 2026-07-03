"""LiteLLM orchestration: system prompt (guardrails), tool definitions, tool
dispatch, and the multi-turn tool-calling loop that ends in a streamed reply."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator, Optional

import litellm
from dotenv import load_dotenv

from app import booking, leads, memory, retrieval

load_dotenv()

MODEL = os.environ.get("LITELLM_MODEL", "gemini/gemini-2.5-flash-lite")
MAX_TOOL_ITERATIONS = 5
MAX_RETRIES = 3

# Transient/server-side errors worth a retry - never retry on client-side
# errors (bad request, auth, etc.), since those will just fail again.
RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.MidStreamFallbackError,
)


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """A per-day quota error will never succeed on retry within the same day -
    retrying just burns more of the (very small) daily budget for nothing."""
    return "PerDay" in str(exc)


async def _completion_with_retry(**kwargs):
    delay = 2
    for attempt in range(MAX_RETRIES):
        try:
            return litellm.completion(**kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            if _is_daily_quota_exhausted(e) or attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2

SYSTEM_PROMPT = """You are the dubizzle cars AI Assistant, a friendly, knowledgeable agent that helps users \
explore a used-car inventory, book test drives, and get their needs qualified as a lead.

SCOPE - you may ONLY discuss:
- The car inventory you can search via the search_inventory tool.
- Booking test drives via the book_test_drive tool.
- Gathering the user's budget/needs and saving them as a lead via the save_lead tool.
- Light greetings/small talk that is car-shopping related (e.g. "hi", "thanks").

GUARDRAILS:
- If the user asks about anything unrelated to cars/this dealership (coding help, trivia, history, \
general knowledge, other topics), politely decline in one sentence and steer the conversation back to \
car shopping. Do not answer the off-topic question, even partially.
- NEVER mention or compare any other car marketplace or dealership by name (e.g. no naming competitors). \
Speak only about this inventory.
- NEVER invent or guess a spec (price, mileage, year, feature, color, etc.) that is not literally present \
in a row returned by search_inventory. If the user asks for something not in the data (e.g. price, since \
this dataset has no structured price column), say plainly that it isn't listed and offer to note their \
budget as a lead instead, or point out if a price appears to be mentioned in the free-text description/title.
- Always ground car descriptions in the exact fields returned by the tools. If a search returns no results, \
say so and suggest broadening the search rather than making something up.

GROUNDING:
- Always call search_inventory before describing any specific cars - never answer from memory or assumption.
- When a user refers to "that one", "the first Honda", "it", etc., resolve the reference using the \
conversation history already in context (do not ask the user to repeat themselves if it's inferable).
- The inventory's structured `model` field is often a family name (e.g. "gls-class"), not the colloquial \
trim name a user says (e.g. "GLS450", which actually lives in `trim`/`title`). If a structured search by \
make/model/trim returns zero results, before telling the user nothing matches, retry once using the \
free-text `keyword` parameter with the term they used - only report "not found" if that also comes up empty.

BOOKING:
- Test drives are only available Monday-Saturday, 8:00-20:00. If a user requests a Sunday or an out-of-hours \
time, explain the constraint and ask them to pick a valid slot.
- Use book_test_drive to attempt bookings and relay its result (success or the rejection reason) verbatim in \
your own words.

LEAD QUALIFICATION:
- Naturally, over the course of the conversation, try to learn the user's price range and other needs \
(body type, make preference, etc.) and call save_lead to record them once you have at least something \
meaningful to save. You can call save_lead more than once as you learn more.

RETURNING USERS:
- If a long-term profile is provided in context (previous preferences, a car of interest, etc.), greet the \
user warmly by referencing it naturally (e.g. "Welcome back! Last time you were looking for a white SUV \
under 20k - still on the hunt?"). Use get_user_profile if you need to look it up explicitly.

Keep replies concise and conversational, like a helpful salesperson texting - not a wall of text."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_inventory",
            "description": (
                "Search the used-car inventory. All filters are optional and combine with AND. "
                "'keyword' free-text searches the title and description fields (useful for body type "
                "like 'SUV', or features like 'sunroof', or things a structured filter can't catch)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "make": {"type": "string", "description": "Car make, e.g. 'toyota'"},
                    "model": {"type": "string", "description": "Car model, e.g. 'camry'"},
                    "trim": {"type": "string", "description": "Trim level, e.g. 'c300 luxury'"},
                    "year_min": {"type": "integer", "description": "Minimum model year (inclusive)"},
                    "year_max": {"type": "integer", "description": "Maximum model year (inclusive)"},
                    "keyword": {"type": "string", "description": "Free-text keyword to search title/description, e.g. 'SUV', 'sunroof', 'GCC'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_test_drive",
            "description": "Book a test drive for a specific listing. Only Monday-Saturday, 8:00-20:00 slots are valid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "integer", "description": "The Listing_ID of the car"},
                    "requested_day": {"type": "string", "description": "Day of week (e.g. 'Wednesday') or an ISO date (YYYY-MM-DD)"},
                    "requested_time": {"type": "string", "description": "24h time as HH:MM, e.g. '14:00'"},
                    "user_id": {"type": "string", "description": "The user's ID"},
                },
                "required": ["listing_id", "requested_day", "requested_time", "user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_lead",
            "description": (
                "Record/update the user's budget and needs as a qualified lead. Appends to the leads log "
                "and updates their long-term profile. Call whenever you learn new qualifying info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "price_min": {"type": "number", "description": "Lower end of stated budget, if any"},
                    "price_max": {"type": "number", "description": "Upper end of stated budget, if any"},
                    "notes": {"type": "string", "description": "Free-text summary of needs/preferences"},
                    "car_of_interest_id": {"type": "integer", "description": "Listing_ID of a car they showed interest in, if any"},
                    "body_type": {"type": "string", "description": "e.g. 'SUV', 'sedan', 'coupe'"},
                    "make_pref": {"type": "string", "description": "Preferred make, if stated"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Fetch a returning user's long-term profile/preferences by user_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
        },
    },
]


def _execute_tool(name: str, args: dict) -> dict:
    if name == "search_inventory":
        results = retrieval.search_inventory(
            make=args.get("make"),
            model=args.get("model"),
            trim=args.get("trim"),
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
            keyword=args.get("keyword"),
        )
        return {"count": len(results), "results": results}

    if name == "book_test_drive":
        result = booking.book_test_drive(
            listing_id=args["listing_id"],
            requested_day=args["requested_day"],
            requested_time=args["requested_time"],
            user_id=args["user_id"],
        )
        if result["success"]:
            memory.upsert_preferences(user_id=args["user_id"], car_of_interest_id=args["listing_id"])
            leads.append_lead(
                user_id=args["user_id"],
                car_of_interest_id=args["listing_id"],
                booked_slot=f"{result['day']} {result['time']}",
            )
        return result

    if name == "save_lead":
        leads.append_lead(
            user_id=args["user_id"],
            price_min=args.get("price_min"),
            price_max=args.get("price_max"),
            notes=args.get("notes"),
            car_of_interest_id=args.get("car_of_interest_id"),
        )
        memory.upsert_preferences(
            user_id=args["user_id"],
            price_min=args.get("price_min"),
            price_max=args.get("price_max"),
            body_type=args.get("body_type"),
            make_pref=args.get("make_pref"),
            notes=args.get("notes"),
            car_of_interest_id=args.get("car_of_interest_id"),
        )
        return {"success": True, "message": "Lead saved."}

    if name == "get_user_profile":
        profile = memory.get_user_profile(args["user_id"])
        return profile or {"found": False}

    return {"error": f"Unknown tool '{name}'"}


_PROFILE_SIGNAL_FIELDS = ("name", "notes", "price_min", "price_max", "body_type", "make_pref", "car_of_interest_id")


def _has_meaningful_history(profile: Optional[dict]) -> bool:
    """A `users` row alone (just created_at/last_seen) doesn't count as history -
    only actual name/preference data recalled from a prior turn does."""
    if not profile:
        return False
    return any(profile.get(field) is not None for field in _PROFILE_SIGNAL_FIELDS)


def _build_messages(session_id: str, user_id: Optional[str], message: str) -> list[dict]:
    system_content = SYSTEM_PROMPT
    if user_id:
        profile = memory.get_user_profile(user_id)
        if _has_meaningful_history(profile):
            system_content += f"\n\nRETURNING USER CONTEXT (user_id={user_id}): {json.dumps(profile, default=str)}"
        else:
            system_content += f"\n\nThis is a new user (user_id={user_id}) with no prior history."

    history = memory.get_history(session_id)
    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message}]
    return messages


async def stream_chat_turn(session_id: str, user_id: Optional[str], message: str) -> AsyncIterator[str]:
    """Runs the tool-calling loop, then streams the final natural-language reply.
    Yields text chunks. Also persists the turn into short-term session history."""
    memory.append_message(session_id, "user", message)
    messages = _build_messages(session_id, user_id, message)

    if user_id:
        memory.touch_user(user_id)

    fallback_msg = (
        "Sorry, I'm having trouble reaching the AI service right now (it's temporarily overloaded or "
        "rate-limited). Please try again in a moment."
    )

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = await _completion_with_retry(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        except RETRYABLE_EXCEPTIONS:
            yield fallback_msg
            memory.append_message(session_id, "assistant", fallback_msg)
            return
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            break

        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result, default=str),
                }
            )
    else:
        messages.append({"role": "user", "content": "Please give your final answer now based on the tool results above."})

    full_reply = ""
    delay = 2
    for attempt in range(MAX_RETRIES):
        yielded_any_this_attempt = False
        try:
            stream = litellm.completion(model=MODEL, messages=messages, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yielded_any_this_attempt = True
                    full_reply += text
                    yield text
            break
        except RETRYABLE_EXCEPTIONS as e:
            if yielded_any_this_attempt or _is_daily_quota_exhausted(e) or attempt == MAX_RETRIES - 1:
                if not full_reply:
                    yield fallback_msg
                    full_reply = fallback_msg
                break
            await asyncio.sleep(delay)
            delay *= 2

    memory.append_message(session_id, "assistant", full_reply)
