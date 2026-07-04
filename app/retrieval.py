"""Pandas-backed inventory search. Exact filtering only - no fuzzy/vector matching,
so the LLM can never be handed a car that doesn't actually exist in the dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cars_cleaned.csv"

_df: Optional[pd.DataFrame] = None


def load_inventory() -> pd.DataFrame:
    global _df
    if _df is None:
        df = pd.read_csv(DATA_PATH)
        for col in ("make", "model", "trim", "title", "description"):
            df[col] = df[col].astype(str)
        _df = df
    return _df


def search_inventory(
    make: Optional[str] = None,
    model: Optional[str] = None,
    trim: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Filter the inventory dataframe. All text filters are case-insensitive
    substring matches; keyword additionally searches title + description so
    body-type / feature words (e.g. "SUV", "sunroof") can be picked up even
    though there's no dedicated body-type column in the dataset."""
    df = load_inventory()
    mask = pd.Series(True, index=df.index)

    if make:
        mask &= df["make"].str.contains(make, case=False, na=False)
    if model:
        mask &= df["model"].str.contains(model, case=False, na=False)
    if trim:
        mask &= df["trim"].str.contains(trim, case=False, na=False)
    if year_min is not None:
        mask &= df["year"] >= year_min
    if year_max is not None:
        mask &= df["year"] <= year_max
    if keyword:
        kw_mask = (
            df["title"].str.contains(keyword, case=False, na=False)
            | df["description"].str.contains(keyword, case=False, na=False)
            | df["make"].str.contains(keyword, case=False, na=False)
            | df["model"].str.contains(keyword, case=False, na=False)
        )
        mask &= kw_mask

    results = df[mask]

    if results.empty:
        # A structured make/model/trim search can legitimately miss because a
        # colloquial term (e.g. "SUV", "GLS450") doesn't literally appear in
        # that column even though it's really in the data (title/description,
        # or a different column than the LLM guessed). Retry treating every
        # non-empty structured term as free text across title/description/
        # trim/model before concluding there's genuinely no match - this
        # doesn't depend on the LLM noticing its own miss and re-calling.
        fallback_terms = [t for t in (model, trim, keyword) if t]
        if fallback_terms:
            fb_mask = pd.Series(True, index=df.index)
            if make:
                fb_mask &= df["make"].str.contains(make, case=False, na=False)
            if year_min is not None:
                fb_mask &= df["year"] >= year_min
            if year_max is not None:
                fb_mask &= df["year"] <= year_max
            term_mask = pd.Series(False, index=df.index)
            for term in fallback_terms:
                term_mask |= (
                    df["title"].str.contains(term, case=False, na=False)
                    | df["description"].str.contains(term, case=False, na=False)
                    | df["trim"].str.contains(term, case=False, na=False)
                    | df["model"].str.contains(term, case=False, na=False)
                )
            fb_mask &= term_mask
            results = df[fb_mask]

    results = results.head(limit)
    return [_to_native(r) for r in results.to_dict(orient="records")]


def get_listing_by_id(listing_id: int) -> Optional[dict]:
    df = load_inventory()
    row = df[df["Listing_ID"] == listing_id]
    if row.empty:
        return None
    return _to_native(row.iloc[0].to_dict())


def _to_native(row: dict) -> dict:
    """Convert numpy scalar types (int64 etc.) to plain Python types so the
    row can be safely json.dumps'd when passed back to the LLM as a tool result."""
    return {
        "Listing_ID": int(row["Listing_ID"]),
        "year": int(row["year"]),
        "make": str(row["make"]),
        "model": str(row["model"]),
        "trim": str(row["trim"]),
        "title": str(row["title"]),
        "description": str(row["description"]),
        "photo_url": str(row["photo_url"]),
    }
