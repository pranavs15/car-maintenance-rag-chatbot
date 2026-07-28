"""
GarageGPT — RAG-based Automotive Maintenance Assistant

This is your original app.py (real ask_question() backend from src/chatbot)
restyled with the GarageGPT UI shell. No placeholder logic — every chat
turn calls your actual RAG pipeline.
"""

import streamlit as st
import sys
import random
import base64
import os
from datetime import datetime

sys.path.append("src")

from chatbot import ask_question

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GarageGPT",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS — minimal, modern, mid-tone garage aesthetic
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    :root {
        --accent: #FF6A3D;
        --accent-soft: rgba(255, 106, 61, 0.14);
        --bg-main: #E9E6E1;
        --bg-card: #F4F1EC;
        --border: #D8D3CB;
        --text-dim: #6B6660;
        --text-main: #2B2823;
    }

    /* Hide default streamlit chrome */
    #MainMenu, header, footer {visibility: hidden;}

    /* App background */
    .stApp {
        background: var(--bg-main);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ── Header / brand ── */
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 6px;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-soft);
        flex-shrink: 0;
    }
    .brand-icon img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .brand-icon.fallback {
        font-size: 22px;
    }
    .brand-title {
        font-size: 21px;
        font-weight: 700;
        color: var(--text-main);
        letter-spacing: -0.3px;
    }
    .brand-sub {
        color: var(--text-dim);
        font-size: 12.5px;
        margin-top: -4px;
        margin-bottom: 18px;
    }

    /* ── Upload box ── */
    .upload-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-main);
        border: 1.5px dashed var(--border);
        border-radius: 12px;
    }

    /* ── Doc chips ── */
    .doc-chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-main);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 7px 10px;
        margin-bottom: 6px;
        font-size: 13px;
        color: var(--text-main);
    }
    .doc-chip span.tag {
        color: var(--accent);
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── History items ── */
    .history-item {
        padding: 9px 11px;
        border-radius: 9px;
        margin-bottom: 4px;
        font-size: 13.5px;
        color: var(--text-main);
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.15s ease;
    }
    .history-item:hover {
        background: var(--bg-main);
        border-color: var(--border);
    }
    .history-item.active {
        background: var(--accent-soft);
        border-color: rgba(255,106,61,0.35);
        color: #fff;
    }
    .history-time {
        color: var(--text-dim);
        font-size: 11px;
        margin-top: 2px;
    }

    .section-label {
        font-size: 11.5px;
        font-weight: 600;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin: 18px 0 8px 0;
    }

    /* ── Chat area ── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-top: 9vh;
        text-align: center;
    }
    .empty-emoji {
        font-size: 44px;
        margin-bottom: 10px;
    }
    .empty-title {
        font-size: 26px;
        font-weight: 700;
        color: var(--text-main);
        letter-spacing: -0.5px;
    }
    .empty-sub {
        color: var(--text-dim);
        font-size: 14.5px;
        margin-top: 6px;
        max-width: 420px;
    }

    /* Suggestion pills */
    .stButton button {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-main) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 13.5px !important;
        text-align: left !important;
        transition: all 0.15s ease !important;
        width: 100%;
    }
    .stButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: transparent;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #6FCF97;
        background: rgba(111, 207, 151, 0.1);
        border: 1px solid rgba(111, 207, 151, 0.25);
        padding: 4px 10px;
        border-radius: 20px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #6FCF97;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CAR-THEMED GREETING MESSAGES
# ──────────────────────────────────────────────────────────────────
GREETINGS = [
    ("🔧", "What's clunking?"),
    ("🚗", "Hey, night shift mechanic."),
    ("🛞", "What's rattling under there?"),
    ("🏁", "Ready when you are, chief."),
    ("⚙️", "What's grinding today?"),
    ("🔩", "Something loose we should talk about?"),
    ("🚦", "Let's diagnose the situation."),
    ("🧰", "Toolbox is open. What's the issue?"),
    ("🛠️", "What's stalling on you?"),
]

SUGGESTIONS = [
    "🛑  Why does my brake pedal feel spongy?",
    "🔋  Car won't start — clicking sound only",
    "🌡️  Engine temp gauge keeps spiking",
    "🧴  Recommended oil change interval for diesel?",
]

# ──────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []  # populated from real questions asked
if "greeting" not in st.session_state:
    st.session_state.greeting = random.choice(GREETINGS)


def get_logo_b64(path="logo.jpg"):
    """Reads logo.jpg from the same folder as this script and returns a base64 data URI."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    return None


def process_uploaded_file(uploaded_file):
    """
    Cosmetic only for now — lists the file in the sidebar.
    TODO: wire this into your actual ingestion pipeline (chunk, embed
    with nomic-embed-text, upsert into ChromaDB) if you want uploads
    here to actually extend the knowledge base that ask_question() reads from.
    """
    return {"name": uploaded_file.name, "size_kb": round(uploaded_file.size / 1024, 1)}


def get_bot_response(user_query: str) -> dict:
    """
    Calls the real RAG pipeline via src/chatbot.ask_question().
    Expects a dict back with "answer" and "context" keys, same as
    your original app.py.
    """
    try:
        result = ask_question(user_query)
    except Exception as e:
        return {
            "answer": f"⚠️ Something went wrong while searching the manuals:\n\n`{e}`",
            "context": "",
        }
    return {
        "answer": result.get("answer", "No answer returned."),
        "context": result.get("context", ""),
    }


def handle_new_message(user_text: str):
    """Appends a user message, runs the real RAG call, appends the answer, reruns."""
    is_new_conversation = len(st.session_state.messages) == 0

    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.spinner("Checking the manuals…"):
        result = get_bot_response(user_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "context": result["context"],
    })

    if is_new_conversation:
        st.session_state.chat_sessions.insert(0, {
            "title": user_text[:42] + ("…" if len(user_text) > 42 else ""),
            "time": datetime.now().strftime("%H:%M"),
        })

    st.rerun()


# ──────────────────────────────────────────────────────────────────
# SIDEBAR — brand, upload, chat history
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_uri = get_logo_b64("logo.jpg")
    if logo_uri:
        icon_html = f'<div class="brand-icon"><img src="{logo_uri}" alt="GarageGPT logo"></div>'
    else:
        icon_html = '<div class="brand-icon fallback">🔧</div>'

    st.markdown(f"""
        <div class="brand-wrap">
            {icon_html}
            <div class="brand-title">GarageGPT</div>
        </div>
        <div class="brand-sub">Your RAG-powered shop manual assistant</div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="status-badge"><span class="status-dot"></span>Knowledge base ready</span>', unsafe_allow_html=True)

    if st.button("➕  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.greeting = random.choice(GREETINGS)
        st.rerun()

    st.markdown('<div class="section-label">Reference Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload manuals, spec sheets, DTC guides",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            existing_names = [d["name"] for d in st.session_state.documents]
            if f.name not in existing_names:
                st.session_state.documents.append(process_uploaded_file(f))

    if st.session_state.documents:
        for doc in st.session_state.documents:
            st.markdown(f"""
                <div class="doc-chip">
                    <span>📄 {doc['name']}</span>
                    <span class="tag">{doc['size_kb']} KB</span>
                </div>
            """, unsafe_allow_html=True)
        st.caption("Listed here only — not yet indexed. Wire process_uploaded_file() to your ingestion pipeline to add these to the knowledge base.")
    else:
        st.caption("No documents indexed yet")

    st.markdown('<div class="section-label">Chat History</div>', unsafe_allow_html=True)
    if st.session_state.chat_sessions:
        for i, chat in enumerate(st.session_state.chat_sessions):
            st.markdown(f"""
                <div class="history-item">
                    💬 {chat['title']}
                    <div class="history-time">{chat['time']}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("Your past questions will show up here")

# ──────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    emoji, headline = st.session_state.greeting
    st.markdown(f"""
        <div class="empty-state">
            <div class="empty-emoji">{emoji}</div>
            <div class="empty-title">{headline}</div>
            <div class="empty-sub">Ask about diagnostics, part specs, or repair steps — GarageGPT pulls answers straight from your indexed manuals.</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    cols = st.columns(2)
    clicked_suggestion = None
    for i, s in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                clicked_suggestion = s.split("  ", 1)[1] if "  " in s else s
    if clicked_suggestion:
        handle_new_message(clicked_suggestion)
else:
    for msg in st.session_state.messages:
        avatar = "🔧" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("context"):
                with st.expander("📄 Retrieved context"):
                    st.text(msg["context"])

# ──────────────────────────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask GarageGPT about a repair, part, or diagnostic code…")
if prompt:
    handle_new_message(prompt)
