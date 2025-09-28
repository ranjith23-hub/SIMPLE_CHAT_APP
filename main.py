import os
import time
import json
import re
from typing import List, Dict, Any
import streamlit as st
import requests

# --- Default persona ---
DEFAULT_PERSONA = (
    "You are Breezy — a funny, warm, slightly sarcastic friend who loves puns and short jokes. "
    "Be supportive, playful, and concise. Prefer light humor; never be mean or insulting. "
    "Reference memories when helpful, and ask follow-ups only if they genuinely help."
)

# --- Gemini API settings ---
GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-2.0-flash-lite"

# --- Memory reducer function ---
def memory_reducer(state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    typ = action.get("type")
    payload = action.get("payload")
    if typ == "ADD_USER":
        entry = {"role": "user", "text": payload["text"], "ts": time.time()}
        state["history"].append(entry)
        extracted = extract_memories_from_text(payload["text"])
        for mem in extracted:
            mem_obj = {
                "id": f"mem_{int(time.time()*1000)}",
                "text": mem,
                "tags": guess_tags(mem),
                "ts": time.time(),
            }
            state["memories"].append(mem_obj)
        return state
    if typ == "ADD_ASSISTANT":
        entry = {"role": "assistant", "text": payload["text"], "ts": time.time()}
        state["history"].append(entry)
        return state
    if typ == "RESET":
        state = {"history": [], "memories": [], "persona": DEFAULT_PERSONA}
        return state
    if typ == "PRUNE_MEMORIES":
        keep_n = payload.get("keep_n", 20)
        state["memories"] = sorted(
            state["memories"], key=lambda m: m["ts"], reverse=True
        )[:keep_n]
        return state
    return state


def extract_memories_from_text(text: str) -> List[str]:
    candidates = []
    text = text.strip()
    m = re.search(r"\bmy name is ([A-Za-z \-']{2,40})", text, re.I)
    if m:
        candidates.append(f"Name: {m.group(1).strip()}")
    m2 = re.search(r"\bi(?:'| a)?m ([A-Za-z \-']{2,40})\b", text, re.I)
    if m2:
        maybe_name = m2.group(1).strip()
        if len(maybe_name.split()) <= 3:
            candidates.append(f"Name (maybe): {maybe_name}")
    m3 = re.findall(
        r"\bI (?:love|like|hate|enjoy|prefer) ([A-Za-z0-9 \-']{2,40})", text, re.I
    )
    for it in m3:
        candidates.append(f"Preference: {it.strip()}")
    m4 = re.search(r"\bI live in ([A-Za-z0-9 ,\-']{2,80})", text, re.I)
    if m4:
        candidates.append(f"Lives in: {m4.group(1).strip()}")
    m5 = re.search(r"\bI work (?:as|at) ([A-Za-z0-9 ,\-']{2,80})", text, re.I)
    if m5:
        candidates.append(f"Job: {m5.group(1).strip()}")
    if len(text.split()) <= 6 and any(c.isalpha() for c in text):
        candidates.append(text)
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique

def guess_tags(text: str) -> List[str]:
    tags = []
    if re.search(r"name", text, re.I):
        tags.append("identity")
    if re.search(r"love|like|hate|prefer|enjoy", text, re.I):
        tags.append("preference")
    if re.search(r"live|city|lives", text, re.I):
        tags.append("location")
    if re.search(r"job|work", text, re.I):
        tags.append("work")
    return tags

# --- Gemini API call ---
def call_gemini_api(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_output_tokens: int = 512,
) -> str:
    if not api_key:
        raise ValueError("API key is required to call Gemini.")
    url = GEMINI_API_URL_TEMPLATE.format(model=model)
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    resp = requests.post(url, headers=headers, params=params, json=body, timeout=30)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e} - {resp.text}")
    data = resp.json()
    if "candidates" in data and data["candidates"]:
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
    return json.dumps(data)

# --- State init ---
def init_state():
    if "lg_state" not in st.session_state:
        st.session_state["lg_state"] = {
            "history": [],
            "memories": [],
            "persona": DEFAULT_PERSONA,
        }
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.environ.get("GOOGLE_API_KEY", "")

# --- One chat turn ---
def run_chat_turn(user_text: str, api_key: str, model: str, temperature: float):
    state = st.session_state["lg_state"]
    state = memory_reducer(state, {"type": "ADD_USER", "payload": {"text": user_text}})
    persona = state.get("persona", DEFAULT_PERSONA)

    mem_summary = ""
    if state["memories"]:
        recent_mems = sorted(state["memories"], key=lambda m: m["ts"], reverse=True)[:6]
        mem_lines = [f"- {m['text']}" for m in recent_mems]
        mem_summary = "\n".join(
            ["Here are some remembered facts about the user:"] + mem_lines
        )

    history_lines = []
    for h in state["history"][-16:]:
        prefix = "User:" if h["role"] == "user" else ("Breezy:")
        history_lines.append(f"{prefix} {h['text']}")

    prompt_parts = [
        persona,
        mem_summary,
        "Recent chat:",
        *history_lines,
        "Assistant (Breezy) reply:",
    ]
    full_prompt = "\n".join([p for p in prompt_parts if p])

    try:
        assistant_text = call_gemini_api(
            full_prompt, api_key=api_key, model=model, temperature=temperature
        )
    except Exception as e:
        assistant_text = f"(Error calling Gemini: {e})"

    state = memory_reducer(
        state, {"type": "ADD_ASSISTANT", "payload": {"text": assistant_text}}
    )
    st.session_state["lg_state"] = state

# --- Main Streamlit UI ---
def main():
    st.set_page_config(page_title="Breezy Chatbot", layout="wide")
    init_state()

    col1, col2 = st.columns([3, 1])
    with col1:
        for msg in st.session_state["lg_state"]["history"]:
            if msg["role"] == "user":
                st.markdown(f"*You:* {msg['text']}")
            elif msg["role"] == "assistant":
                st.markdown(f"*Breezy:* {msg['text']}")

        user_input = st.text_input("Say something:", key="chat_input")
        if st.button("Send") and user_input:
            api_key = st.session_state.get("api_key", "")
            if not api_key:
                st.error("Please set your Gemini API key in the sidebar.")
            else:
                run_chat_turn(user_input, api_key, DEFAULT_MODEL, 0.7)
                st.rerun()

    with col2:
        st.sidebar.title("Settings & Memory")
        api_key_col = st.sidebar.text_input(
            "Gemini API key",
            type="password",
            value=st.session_state.get("api_key", ""),
        )
        st.session_state["api_key"] = api_key_col

        persona_in = st.sidebar.text_area(
            "Persona",
            value=st.session_state["lg_state"].get("persona", DEFAULT_PERSONA),
            height=100,
        )
        if st.sidebar.button("Update persona"):
            st.session_state["lg_state"]["persona"] = persona_in
        if st.sidebar.button("Prune memories"):
            st.session_state["lg_state"] = memory_reducer(
                st.session_state["lg_state"],
                {"type": "PRUNE_MEMORIES", "payload": {"keep_n": 10}},
            )
        if st.sidebar.button("Reset"):
            st.session_state["lg_state"] = memory_reducer(
                st.session_state["lg_state"], {"type": "RESET", "payload": {}}
            )

        st.sidebar.subheader("Current memories")
        for m in sorted(
            st.session_state["lg_state"]["memories"], key=lambda x: x["ts"], reverse=True
        ):
            st.sidebar.write(f"• {m['text']}")

if __name__ == "__main__":
    main()
