"""Pydantic request/response models shared across the API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated ID for the current chat session")
    user_id: Optional[str] = Field(None, description="Known long-term user ID, if the user has identified themselves")
    message: str


class InventorySearchParams(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    trim: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    keyword: Optional[str] = None


class CarListing(BaseModel):
    Listing_ID: int
    year: int
    make: str
    model: str
    trim: str
    title: str
    description: str
    photo_url: str


class BookingRequest(BaseModel):
    listing_id: int
    requested_day: str = Field(..., description="Day of week, e.g. 'Monday' or an ISO date")
    requested_time: str = Field(..., description="Time in HH:MM 24h format, e.g. '14:00'")
    user_id: str


class BookingResult(BaseModel):
    success: bool
    message: str
    listing_id: int
    day: Optional[str] = None
    time: Optional[str] = None


class LeadRequest(BaseModel):
    user_id: str
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    notes: Optional[str] = None
    car_of_interest_id: Optional[int] = None
    body_type: Optional[str] = None
    make_pref: Optional[str] = None


class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = None
    created_at: Optional[str] = None
    last_seen: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    body_type: Optional[str] = None
    make_pref: Optional[str] = None
    notes: Optional[str] = None
    car_of_interest_id: Optional[int] = None
    updated_at: Optional[str] = None
