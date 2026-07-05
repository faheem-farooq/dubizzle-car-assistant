"""FastAPI backend: owns all state, tool execution, and LLM orchestration.
The Streamlit client is a thin caller of these endpoints only."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app import booking as booking_module
from app import leads as leads_module
from app import llm, memory, retrieval
from app.schemas import (
    BookingRequest,
    BookingResult,
    ChatRequest,
    InventorySearchParams,
    InventorySearchResponse,
    UserProfile,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.init_db()
    retrieval.load_inventory()
    yield


app = FastAPI(title="dubizzle Car Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    async def event_gen():
        async for chunk in llm.stream_chat_turn(req.session_id, req.user_id, req.message):
            yield {"event": "message", "data": chunk}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())


@app.get("/inventory/search", response_model=InventorySearchResponse)
def inventory_search(
    params: InventorySearchParams = Depends(),
    limit: int = Query(10, ge=1, le=100),
) -> InventorySearchResponse:
    results = retrieval.search_inventory(
        make=params.make,
        model=params.model,
        trim=params.trim,
        year_min=params.year_min,
        year_max=params.year_max,
        keyword=params.keyword,
        limit=limit,
    )
    return InventorySearchResponse(count=len(results), results=results)


@app.post("/booking", response_model=BookingResult)
def book(req: BookingRequest) -> BookingResult:
    result = booking_module.book_test_drive(
        listing_id=req.listing_id,
        requested_day=req.requested_day,
        requested_time=req.requested_time,
        user_id=req.user_id,
    )
    if result["success"]:
        memory.upsert_preferences(user_id=req.user_id, car_of_interest_id=req.listing_id)
        leads_module.append_lead(
            user_id=req.user_id,
            car_of_interest_id=req.listing_id,
            booked_slot=f"{result['day']} {result['time']}",
        )
    return BookingResult(**result)


@app.get("/users/{user_id}", response_model=Optional[UserProfile])
def get_user(user_id: str):
    profile = memory.get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfile(user_id=user_id, **{k: v for k, v in profile.items() if k != "user_id"})
