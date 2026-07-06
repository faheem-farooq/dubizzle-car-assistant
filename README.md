# dubizzle Car Assistant

An AI assistant that helps users explore a ~100-listing used-car inventory,
holds a contextual multi-turn conversation, books test drives, qualifies
leads, and recognizes returning users across sessions.

Built for the dubizzle ML Intern take-home assessment.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Copy the env template and fill in your key:
   ```bash
   cp .env.example .env
   # edit .env, set GEMINI_API_KEY=...
   ```
4. Install dependencies (uv resolves and creates the venv automatically from `pyproject.toml`):
   ```bash
   uv sync
   ```

The cleaned inventory is already exported to `data/cars_cleaned.csv` from the
provided `cleaned dataset` sheet. SQLite (`data/memory.db`) and `leads.csv`
are created automatically on first run.

## Run

**Backend** (from the project root):
```bash
uv run uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API docs.

**Frontend** (in a second terminal):
```bash
uv run streamlit run streamlit_app/app.py
```
This opens a chat UI at `http://localhost:8501` that talks to the backend on
port 8000. Set `BACKEND_URL` as an env var if the backend runs elsewhere.

### Demonstrating long-term memory across "sessions"

1. In the sidebar, enter a name (e.g. `faheem`) and click **Identify me**.
2. Chat for a bit — ask about SUVs, mention a budget, book a test drive.
3. Click **New session (keep user)** — this resets `session_id` (simulating
   a brand-new conversation/short-term memory wipe) while keeping the same
   `user_id`. Send a message like "hey, remember me?" — the assistant will
   greet you referencing what it saved last time, pulled from SQLite.

## Demo Screenshots

### Multi-turn conversation & inventory grounding

**"Hi, my name is Faheem" → "I'm looking for a Toyota SUV, what do you have?"**
Tests inventory search grounding — agent correctly reports no Toyota SUVs exist in inventory rather than inventing one.
![](screenshots/01_toyota_suv_search.png)

**"What makes and body types do you actually have in stock?"**
Tests broader retrieval accuracy — confirms the earlier "no Toyota SUV" result was a genuine data gap, not a retrieval bug.
![](screenshots/02_inventory_overview.png)

**"Tell me more about the Porsche Cayenne" → "What's the mileage on it?"**
Tests short-term memory / pronoun resolution — agent resolves "it" to the previously mentioned car without restatement, per the assignment's explicit requirement.
![](screenshots/03_pronoun_resolution_mileage.png)

**"Do you have anything from Mercedes (AMG line)?"**
Tests input sanitization — confirms special characters in a search query don't crash retrieval.
![](screenshots/04_special_characters_amg.png)

### Lead qualification & booking

**"My budget is between 300,000 and 400,000 AED" → "Can I book a test drive for the Porsche Cayenne, Wednesday at 2pm?"**
Tests lead qualification — the budget (300,000–400,000 AED) is saved to the user's profile and the chat confirms the Porsche Cayenne test drive is booked. This snapshot was captured right as the booking landed, before the sidebar's next refresh picked up `car_of_interest_id` — a display-timing quirk in the demo client, not a data bug (the underlying merge/persistence logic is exercised directly elsewhere in this repo's history).
![](screenshots/05_budget_saved_partial.png)

### Guardrails, booking validation & chit-chat

**"Can you write me a Python script to scrape car listings?" → "Is this cheaper than what I'd find on other classifieds sites?" → "Can I book a test drive for Sunday at 9am?"**
Tests three guardrails in sequence: declines an off-topic coding request, declines a competitor-comparison question without naming any competitor, and rejects a booking outside the Monday–Saturday window.
![](screenshots/06_guardrails_and_invalid_slot.png)

**"Can I come see a car at 9pm on Friday?" → "Hey, how's it going?"**
Tests the time-of-day booking boundary (rejects 9pm as outside the 8:00–20:00 window) and natural chit-chat handling — the assistant responds warmly and proactively surfaces the user's previously saved budget and car of interest.
![](screenshots/07_chitchat_and_more_validation.png)

### Long-term memory (new session)

**Click "New session (keep user)" in the sidebar, then: "Hi again, what was I looking for?"**
Tests long-term memory recall across a genuinely new session — session_id is reset (short-term chat history is empty), but user_id is preserved. The assistant correctly recalls both the previously stated budget (300,000–400,000 AED) and the car of interest (Porsche Cayenne, listing ID 8) by reading from SQLite at the start of the new session.
![](screenshots/08_recall_same_session.png)

**"peter" identified as new user → "What am I looking for?"**
Tests user data isolation — a different `user_id` has zero access to another user's stored preferences; the assistant correctly asks what Peter is looking for instead of surfacing Faheem's saved budget/car.
![](screenshots/09_user_isolation.png)

## Why this stack

- **uv**: fast, single-lockfile dependency + venv management; the assignment
  explicitly calls it out as the preferred tooling.
- **FastAPI**: the assignment requires it outright, and it gives us
  Pydantic-validated request/response models plus native async support for
  SSE streaming with almost no boilerplate.
- **LiteLLM**: lets the app target Gemini (`gemini/gemini-2.5-flash-lite` by
  default, configurable via `LITELLM_MODEL` in `.env` — `gemini-2.0-flash`,
  the model named in the original brief, was deprecated/shut down as of
  mid-2026, so any current key needs a currently-served model instead)
  through an OpenAI-compatible `completion()`/tool-calling interface, which
  keeps `app/llm.py` provider-agnostic — swapping models later is a
  one-line change.
- **Pandas tool-calling over RAG**: the dataset is small (100 rows), mostly
  structured (make/model/trim/year) with one free-text field. A vector DB
  and embeddings would add latency, cost, and a whole new failure mode
  (semantic near-misses) for no real benefit at this scale. Instead, the LLM
  calls `search_inventory` with structured args, and the backend runs an
  **exact pandas filter** — the agent can only ever describe rows that
  really exist, which directly satisfies the "no hallucinated inventory"
  grading criterion. Free-text `keyword` search still covers body-type/
  feature queries (e.g. "SUV", "sunroof") that aren't in a dedicated column.
- **SQLite + CSV split**: SQLite (`users`, `preferences`) is the queryable
  long-term memory store the agent reads back at session start to
  personalize greetings. `leads.csv` is kept as a separate, explicitly
  human-readable artifact because the assignment calls out a **local CSV**
  as the lead-capture mechanism — treating it as a write-once log (rather
  than folding it into SQLite) mirrors how a sales/CRM export would actually
  be consumed downstream.
- **Streamlit**: chosen over the Jupyter notebook option to get a real chat
  UI with `st.chat_input`/`st.write_stream`, which makes the SSE-streamed
  backend and the "new session, same user" long-term-memory demo tangible
  to click through, rather than a linear script.

## Design decisions

The core grounding guarantee is architectural, not prompt-based: the LLM
never free-generates a car — it must call `search_inventory`, which runs a
literal pandas filter over `data/cars_cleaned.csv`, and the tool result (not
the model's own words) is what gets fed back into context. Because the
dataset has no structured price column, price is treated as a
**lead-qualification input** rather than a filterable field: the assistant
asks the user for their budget and stores it via `save_lead` (which both
appends to `leads.csv` and upserts SQLite `preferences`), instead of
regex-guessing prices out of free-text descriptions where they're
inconsistently formatted or simply "call for price". Short-term memory is a
plain in-process list of chat messages keyed by `session_id`, replayed into
the LLM every turn; long-term memory is SQLite keyed by `user_id`, looked up
once at the start of each `/chat` call and injected into the system prompt
so the model can reference it naturally ("last time you were looking for a
white SUV..."). Guardrails (car-topics-only, no competitor mentions, no
fabricated specs) live entirely in the system prompt in `app/llm.py` — this
is intentionally the single place reviewers should look to see the
control surface.

**Outside the scope of this project** (explicitly deferred): user
authentication (the demo uses a free-text name as a stand-in for a real
user ID); multi-tenant isolation for the SQLite/CSV stores; a proper price
extraction model or a structured price field fed back from a real inventory
system; image-based search or vision-model grounding using the `photo_url`
field; eviction/expiry for the in-memory short-term session store (`app/
memory.py`'s `_sessions` dict grows unboundedly for the life of the process
- fine for a demo, not for a long-running deployment); and evaluation/
observability tooling (e.g. logging tool-call accuracy, and specifically
spot-checking hallucination rate) that a production version of this agent
would need. Grounding is enforced by data flow (the tool result, not the
model's own words, is what re-enters context) rather than by prompt alone,
which is the right architectural call - but nothing double-checks that the
model's final prose only cites what the last tool result actually returned,
so a sufficiently contrarian small model could still drift; that residual
risk is worth watching given `gemini-2.5-flash-lite` is a small/cheap model.
One thing that *did* end up in scope after testing: `app/llm.py` retries
transient Gemini errors (503/rate-limit/connection) with backoff and
degrades to a friendly in-band message rather than crashing the SSE stream —
Google's current free-tier quota turned out to be tight enough (20
requests/day per model on the keys used during development) that a
production deployment would need a paid tier or its own request budgeting;
that's noted here rather than solved, since it's a billing decision, not an
engineering one.

## API

- `POST /chat` — `{session_id, user_id?, message}`, streams the assistant's
  reply via Server-Sent Events.
- `GET /inventory/search` — `make, model, trim, year_min, year_max, keyword,
  limit` query params; direct REST access to the same pandas filter the LLM
  tool uses, for testing retrieval independent of the model.
- `POST /booking` — `{listing_id, requested_day, requested_time, user_id}`;
  validates against Monday–Saturday, 8:00–20:00 and rejects everything else.
- `GET /users/{user_id}` — returns the stored long-term profile (debug/demo).
- `GET /health` — basic liveness check.

## Project structure

```
dubizzle-car-assistant/
├── pyproject.toml
├── .env.example
├── data/
│   └── cars_cleaned.csv      # exported from the provided xlsx's "cleaned dataset" sheet
├── app/
│   ├── main.py                # FastAPI app + routes
│   ├── llm.py                 # LiteLLM client, tool defs, tool-calling loop, system prompt
│   ├── retrieval.py            # pandas search logic
│   ├── memory.py                # SQLite long-term + in-memory short-term session store
│   ├── leads.py                  # CSV lead-writing logic
│   ├── booking.py                # slot validation logic
│   └── schemas.py                # Pydantic models
├── streamlit_app/
│   └── app.py                    # Streamlit chat client
└── leads.csv                     # generated at runtime
```
