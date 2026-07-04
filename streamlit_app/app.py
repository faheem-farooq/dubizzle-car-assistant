"""Thin Streamlit chat client. Holds zero business logic - every action is a
call to the FastAPI backend, which owns retrieval, memory, booking, and leads."""
import os
import uuid

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="dubizzle Car Assistant", page_icon="🚗")

st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], .stChatMessage, button, input, textarea {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            "Helvetica Neue", Arial, sans-serif !important;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 dubizzle Car Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Returning user?")
    st.caption("Enter the same name/ID you used before to have the assistant recall your preferences.")
    id_input = st.text_input("Your name or user ID", value=st.session_state.user_id or "")
    if st.button("Identify me"):
        if id_input.strip():
            st.session_state.user_id = id_input.strip().lower().replace(" ", "_")
            st.success(f"Identified as '{st.session_state.user_id}'")
        else:
            st.warning("Enter a name or ID first.")

    if st.session_state.user_id:
        st.info(f"Current user: **{st.session_state.user_id}**")
        try:
            resp = httpx.get(f"{BACKEND_URL}/users/{st.session_state.user_id}", timeout=10)
            if resp.status_code == 200:
                st.json(resp.json())
        except httpx.HTTPError:
            pass

    st.divider()
    if st.button("New session (keep user)"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def stream_reply(message: str):
    payload = {
        "session_id": st.session_state.session_id,
        "user_id": st.session_state.user_id,
        "message": message,
    }
    with httpx.stream("POST", f"{BACKEND_URL}/chat", json=payload, timeout=60) as resp:
        resp.raise_for_status()
        event_type = "message"
        data_lines: list[str] = []
        for line in resp.iter_lines():
            if line is None:
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
            elif line == "":
                # Blank line = end of one SSE event.
                data = "\n".join(data_lines)
                data_lines = []
                if event_type == "done":
                    return
                if data:
                    yield data
                event_type = "message"


user_message = st.chat_input("Ask about a car, book a test drive, or tell us what you're looking for...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        full_reply = st.write_stream(stream_reply(user_message))

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
