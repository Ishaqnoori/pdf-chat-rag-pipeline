import hashlib
import html
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

st.set_page_config(page_title="Ask your PDF", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- config (item 1)
# Keys come from .env locally or the host's Secrets when deployed; never from code.
# .env loads first so placeholder values in secrets.toml can't override real keys.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))  # always the .env next to app.py
CONFIG_KEYS = ["GOOGLE_API_KEY", "GROQ_API_KEY", "FIREBASE_API_KEY", "GROQ_MODEL", "GEMINI_EMBEDDING_MODEL"]


def _is_placeholder(value) -> bool:
    v = str(value or "").strip().strip('"').strip("'")
    return not v or v.lower().startswith("your_") or v.lower().endswith("_here")


for key in CONFIG_KEYS:
    if _is_placeholder(os.getenv(key)):
        os.environ.pop(key, None)
        try:
            if not _is_placeholder(st.secrets[key]):
                os.environ[key] = str(st.secrets[key]).strip()
        except Exception:
            pass

from chain import get_conversational_chain  # noqa: E402
from rag_pipeline import build_index, get_retriever, merge_indexes  # noqa: E402
from utils import auth  # noqa: E402
from utils.citations import sources_label  # noqa: E402
from utils.llm import AVAILABLE_MODELS, MODEL_DAILY_LIMITS, default_model, is_rate_limit, retry_after_seconds  # noqa: E402
from utils.loaders import validate_pdf  # noqa: E402
from utils.render import render_answer, split_followups, stream_preview  # noqa: E402
from utils.security import MAX_QUESTION, clean_text, limiter, log, register_model_limits, wait_text  # noqa: E402

with open(os.path.join(APP_DIR, "assets", "style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

missing = [k for k in CONFIG_KEYS[:3] if not os.getenv(k)]
if missing:
    # Names of missing settings only; values are never shown (item 1, 17)
    st.error(
        f"The app isn't configured yet. Missing: {', '.join(missing)}. "
        "Put them in the `.env` file in the same folder as app.py (see `.env.example`), then restart run.bat."
    )
    st.stop()


# ---------------------------------------------------------------- login gate (item 6)
def render_login():
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'],"
        "[data-testid='stExpandSidebarButton'] { display: none !important; }</style>",
        unsafe_allow_html=True,
    )
    view = st.session_state.get("auth_view", "login")
    title, sub = {
        "signup": ("Create your account", "Sign up to start chatting with your documents."),
        "reset": ("Reset your password", "We'll email you a link to choose a new one."),
    }.get(view, ("Welcome back", "Log in to chat with your documents."))
    st.markdown(
        f'<div class="login"><div class="brand-lg">Ask your PDF</div><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )

    def switch(label, to, key):
        with st.container(key=key):
            if st.button(label, use_container_width=True):
                st.session_state.auth_view = to
                st.rerun()

    with st.container(key="login_box"):
        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.info(notice)

        if view != "reset" and auth.google_configured():
            with st.container(key="google_login"):
                if st.button("Continue with Google", use_container_width=True):
                    st.login("google")
            st.markdown('<div class="or">or</div>', unsafe_allow_html=True)

        if view == "signup":
            with st.form("signup_form", border=False):
                name = st.text_input("Name", placeholder="Full name", max_chars=80, label_visibility="collapsed")
                email = st.text_input("Email", placeholder="Email address", max_chars=254, label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password (8+ characters)",
                                         max_chars=128, label_visibility="collapsed")
                confirm = st.text_input("Confirm", type="password", placeholder="Confirm password",
                                        max_chars=128, label_visibility="collapsed")
                submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                with st.spinner("Creating your account…"):
                    ok, msg = auth.sign_up(name, email, password, confirm)
                if ok:
                    st.session_state.auth_view = "login"
                    st.session_state.auth_notice = msg
                    st.rerun()
                st.error(msg)
            switch("Already have an account? Log in", "login", "auth_switch")

        elif view == "reset":
            with st.form("reset_form", border=False):
                email = st.text_input("Email", placeholder="Email address", max_chars=254, label_visibility="collapsed")
                submitted = st.form_submit_button("Send reset link", use_container_width=True)
            if submitted:
                ok, msg = auth.send_reset(email)
                (st.success if ok else st.error)(msg)
            switch("Back to log in", "login", "auth_switch")

        else:
            with st.form("login_form", border=False):
                email = st.text_input("Email", placeholder="Email", max_chars=254, label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password",
                                         max_chars=128, label_visibility="collapsed")
                submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                with st.spinner("Logging in…"):
                    ok, msg = auth.login(email, password)
                if ok:
                    st.rerun()
                st.error(msg)
            switch("Forgot password?", "reset", "auth_forgot")
            switch("Don't have an account? Sign up", "signup", "auth_switch")

    st.markdown('<p class="legal">Your PDFs stay in this session only.</p>', unsafe_allow_html=True)


user = auth.current_user()
if user is None:
    render_login()
    st.stop()

# ---------------------------------------------------------------- state (per session, item 7)
# The "current chat" lives in these keys; finished chats are kept in st.session_state.chats
# (session only: nothing is written to disk or a database).
CHAT_FIELDS = ("messages", "retriever", "doc_name", "docs", "chat_history_store")
MAX_CHATS = 20
MAX_DOCS = 5             # PDFs per chat
MAX_TOTAL_CHUNKS = 4000  # keeps a multi-PDF chat as fast as one large PDF

for k, v in {"messages": [], "retriever": None, "doc_name": None, "docs": [], "to_answer": None,
             "answer_model": None, "chats": {}}.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("chat_history_store", InMemoryChatMessageHistory())
st.session_state.setdefault("chat_id", uuid.uuid4().hex)
USER_KEY = user["uid"] or user["email"]
register_model_limits({m: MODEL_DAILY_LIMITS.get(m, (50, 500)) for m in AVAILABLE_MODELS})


def model_name(model: str) -> str:
    return AVAILABLE_MODELS.get(model, (model,))[0]


def model_paused_for(model: str) -> int:
    """Seconds until a model the provider rate-limited can be used again (0 = available)."""
    return limiter().blocked_for(f"model:{model}")


def model_headroom(model: str) -> float:
    """Share of capacity left (0..1): the tighter of the app-wide budget and this user's allowance; -1 if paused."""
    if model_paused_for(model):
        return -1.0
    shares = []
    for kind, key in ((f"app_model_day:{model}", "app"), (f"model_day:{model}", USER_KEY)):
        u = limiter().usage(kind, key)
        shares.append((u["limit"] - u["used"]) / u["limit"])
    return min(shares)


def best_model() -> str:
    """The model with the most capacity left; ties go to the preferred default (GROQ_MODEL)."""
    preferred = default_model()
    order = sorted(AVAILABLE_MODELS, key=lambda m: m != preferred)  # preferred first, then menu order
    return max(order, key=model_headroom)  # max() keeps the first of equal scores


def user_model_lock():
    """
    (model, seconds) if this user has used up their own allowance on any model; else None.
    While locked the user can't use or switch to any model, so switching can't get around the limits.
    """
    for model in AVAILABLE_MODELS:
        u = limiter().usage(f"model_day:{model}", USER_KEY)
        if u["used"] >= u["limit"]:
            return model, u["next_free_in"]
    return None


def lock_message(lock) -> str:
    model, wait = lock
    return (f"You've reached your daily limit on {model_name(model)}, so all models are paused for you "
            f"until it resets in {wait_text(wait)}.")


def reserve_question(model: str):
    """
    Takes one question from the user's overall allowance and from this model's own allowance.
    Never switches model: if this one is out of quota, the question is refused. Returns (ok, message).
    """
    lock = user_model_lock()
    if lock:
        return False, lock_message(lock)
    paused = model_paused_for(model)
    if paused:
        return False, f"{model_name(model)} has reached its usage limit. It resets in {wait_text(paused)}."
    ok, message = limiter().take(f"model:{model}", USER_KEY, label=model_name(model))
    if not ok:
        return False, message
    ok, message = limiter().take("question", USER_KEY)
    if not ok:
        limiter().undo(f"model:{model}", USER_KEY)  # nothing was asked, so give the model allowance back
        return False, message
    return True, None


def release_question(model: str):
    """Refunds a reserved question when no answer could be produced."""
    limiter().undo(f"model:{model}", USER_KEY)
    limiter().undo("question", USER_KEY)


def get_session_history(session_id: str):
    return st.session_state.chat_history_store


def chat_title(messages) -> str:
    first = next((m for m in messages if m["role"] == "user"), {})
    files = first.get("attachments") or ([first["attachment"]] if first.get("attachment") else [])
    title = first.get("content") or ", ".join(files) or "New chat"
    return title if len(title) <= 40 else title[:39].rstrip() + "…"


def save_current_chat():
    """Copies the current chat into the Recents list (only once it has messages)."""
    ss = st.session_state
    if not ss.messages:
        return
    saved = ss.chats.get(ss.chat_id, {})
    ss.chats[ss.chat_id] = {
        **{f: ss[f] for f in CHAT_FIELDS},
        "title": saved["title"] if saved.get("custom_title") else chat_title(ss.messages),
        "custom_title": saved.get("custom_title", False),
        "updated": saved.get("updated", time.time()) if saved.get("count") == len(ss.messages) else time.time(),
        "count": len(ss.messages),
    }
    while len(ss.chats) > MAX_CHATS:  # drop the oldest
        ss.chats.pop(min(ss.chats, key=lambda c: ss.chats[c]["updated"]))


def reset_current_chat():
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.doc_name = None
    st.session_state.docs = []
    st.session_state.to_answer = None
    st.session_state.chat_history_store = InMemoryChatMessageHistory()  # new object: old chats keep theirs
    st.session_state.chat_id = uuid.uuid4().hex


def new_chat():
    save_current_chat()
    reset_current_chat()
    if not st.session_state.get("model_chosen"):
        st.session_state.model_pick = best_model()  # spread usage across models


def delete_chat(chat_id: str):
    st.session_state.chats.pop(chat_id, None)
    if chat_id == st.session_state.chat_id:
        reset_current_chat()  # not saved again: it's gone


def rename_chat(chat_id: str, title: str):
    title = clean_text(title, 40)
    if title and chat_id in st.session_state.chats:
        st.session_state.chats[chat_id].update(title=title, custom_title=True)


def open_chat(chat_id: str):
    save_current_chat()
    chat = st.session_state.chats[chat_id]
    defaults = {"messages": [], "retriever": None, "doc_name": None, "docs": []}
    for f in CHAT_FIELDS:
        st.session_state[f] = chat.get(f, defaults.get(f, InMemoryChatMessageHistory()))
    st.session_state.to_answer = None
    st.session_state.chat_id = chat_id


# A 300-page PDF's index is roughly 25 MB (3,072-dim vectors), so caches stay small for ~1 GB hosts.
@st.cache_resource(show_spinner=False, max_entries=8, ttl=1800)
def process_pdf(_file_bytes: bytes, filename: str, file_hash: str, _allow_ocr=None):
    """Builds the index once per unique file. The temp copy is deleted immediately."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name
    try:
        return build_index(tmp_path, source_name=filename, allow_ocr=_allow_ocr)
    finally:
        os.remove(tmp_path)


@st.cache_resource(show_spinner=False, max_entries=6, ttl=1800)
def chat_index(file_hashes: tuple, _indexes: tuple):
    """One search index over every PDF in the chat. Cached, so switching chats doesn't rebuild it."""
    return merge_indexes(list(_indexes))


def attach_pdf(file_name: str, file_hash: str, index) -> str | None:
    """Adds a PDF to the current chat. Returns a message for the user if it can't be added."""
    docs = st.session_state.docs
    if any(d["hash"] == file_hash for d in docs):
        return f"**{file_name}** is already in this chat."
    if len(docs) >= MAX_DOCS:
        return f"A chat can hold up to {MAX_DOCS} PDFs. Start a **New chat** for more."
    if sum(len(d["index"].chunks) for d in docs) + len(index.chunks) > MAX_TOTAL_CHUNKS:
        return "These PDFs are too large to search together. Start a **New chat** for this one."
    docs.append({"name": file_name, "hash": file_hash, "index": index})
    combined = chat_index(tuple(d["hash"] for d in docs), tuple(d["index"] for d in docs))
    st.session_state.retriever = get_retriever(combined)
    st.session_state.doc_name = ", ".join(d["name"] for d in docs)
    return None


def say(text: str, **extra):
    st.session_state.messages.append({"role": "assistant", "content": text, **extra})


STREAM_REFRESH = 0.05  # seconds between screen updates while streaming (~20 fps keeps the browser light)


def stream_answer(question: str, model: str):
    """
    Streams the answer from `model` only (never another one). If the provider says the model is over
    its limit, the model is paused app-wide until the provider's reset time and the user is refunded.
    Returns (text, retrieved_docs, seconds).
    """
    box = st.empty()
    box.markdown('<p class="thinking">Thinking…</p>', unsafe_allow_html=True)
    text, docs, last_draw = "", [], 0.0
    started = time.perf_counter()
    chain = get_conversational_chain(st.session_state.retriever, get_session_history, model)
    try:
        for chunk in chain.stream({"question": question}, config={"configurable": {"session_id": "default"}}):
            docs = chunk.get("docs", docs)
            piece = chunk.get("answer")
            if piece:
                text += piece
                now = time.perf_counter()
                if now - last_draw >= STREAM_REFRESH:
                    box.markdown(stream_preview(text) + " ▍")
                    last_draw = now
    except Exception as e:
        release_question(model)  # no answer, so this question doesn't count
        if is_rate_limit(e):
            wait = retry_after_seconds(e)
            limiter().block(f"model:{model}", wait)
            log.warning("model=%s rate-limited by provider for %ss", model, wait)
            text = f"{model_name(model)} has reached its usage limit. It resets in {wait_text(wait)}."
        else:
            log.exception("Answer generation failed")
            text = "I couldn't get an answer just now. Please try again."
        docs = []
    box.empty()
    log.info("answer model=%s seconds=%.2f chars=%d chunks=%d",  # observability, server log only
             model, time.perf_counter() - started, len(text), len(docs))
    return text, docs, time.perf_counter() - started


def render_sources(label):
    if isinstance(label, list):  # messages saved before multi-PDF support
        label = f"{'Page' if len(label) == 1 else 'Pages'} {', '.join(label)}" if label else ""
    if label:
        st.markdown(f'<p class="sources">{html.escape(label)}</p>', unsafe_allow_html=True)


def chat_as_markdown() -> str:
    lines = [f"# Chat about {st.session_state.doc_name or 'a PDF'}", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        files = m.get("attachments") or ([m["attachment"]] if m.get("attachment") else [])
        body = m.get("content") or (f"[attached {', '.join(files)}]" if files else "")
        lines += [f"**{who}:**", "", body, ""]
    return "\n".join(lines)


def regenerate_last():
    """Drops the last answer (and its memory) and asks the same question again."""
    msgs = st.session_state.messages
    question = next((m["content"] for m in reversed(msgs) if m["role"] == "user" and m.get("content")), None)
    if not question or msgs[-1].get("kind") != "answer":
        return
    ok, message = reserve_question(st.session_state.model_pick)
    if not ok:
        return say(message)  # keep the old answer if we can't make a new one
    st.session_state.answer_model = st.session_state.model_pick
    msgs.pop()
    history = st.session_state.chat_history_store
    history.messages = history.messages[:-2]  # the last question/answer pair
    st.session_state.to_answer = question


def add_pdf(pdf) -> bool:
    """Validates, indexes and attaches one uploaded PDF. Reports problems in the chat; True if it was added."""
    file_name = clean_text(pdf.name, 120)
    ok, message = limiter().take("upload", USER_KEY)
    if not ok:
        say(message)
        return False
    data = pdf.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    ocr_taken = []

    def take_ocr():
        ok, msg = limiter().take("ocr", USER_KEY)
        if ok:
            ocr_taken.append(True)
        return ok, msg

    try:
        validate_pdf(data)
        with st.spinner(f"Reading {file_name}… (scanned PDFs take up to a minute)"):
            # A scanned PDF only spends OCR quota if it actually needs OCR (checked inside)
            index = process_pdf(data, file_name, file_hash, take_ocr)
    except Exception as e:
        if ocr_taken:
            limiter().undo("ocr", USER_KEY)  # the read failed, so it shouldn't count
        if isinstance(e, ValueError):  # our own, user-safe messages
            say(f"I couldn't use **{file_name}**. {e}")
        else:
            log.exception("PDF processing failed")  # details stay in the server log (item 17)
            say(f"Something went wrong while reading **{file_name}**. Please try again.")
        return False
    problem = attach_pdf(file_name, file_hash, index)
    if problem:
        say(problem)
        return False
    if index.chunks and index.chunks[0].metadata.get("ocr"):
        say(f"**{file_name}** is a scanned PDF, so I read it with text recognition. "
            "Small recognition errors are possible, so double-check important numbers.", kind="notice")
    return True


def handle_submission(text: str, files):
    """Validates the user turn, indexes attached PDFs and queues the question. Answers come after rerun."""
    text = clean_text(text, MAX_QUESTION + 1)
    files = list(files or [])[:MAX_DOCS]
    if not text and not files:
        return
    names = [clean_text(f.name, 120) for f in files]
    st.session_state.messages.append({"role": "user", "content": text, "attachments": names})

    if len(text) > MAX_QUESTION:
        return say(f"That message is too long. Keep questions under {MAX_QUESTION:,} characters.")

    if files:
        added = [f for f in files if add_pdf(f)]
        if not added and st.session_state.retriever is None:
            return
        if not text:
            if added:
                n = len(st.session_state.docs)
                names_added = ", ".join(f"**{clean_text(f.name, 120)}**" for f in added)
                say(f"{names_added} {'is' if len(added) == 1 else 'are'} ready. What would you like to know?"
                    if n == len(added) else f"{names_added} added. You can now ask about all {n} PDFs in this chat.")
            return

    if st.session_state.retriever is None:
        return say("Attach a PDF with the **+** button first, then ask your question.")

    ok, message = reserve_question(st.session_state.model_pick)
    if not ok:
        return say(message)
    st.session_state.answer_model = st.session_state.model_pick
    st.session_state.to_answer = text


# ---------------------------------------------------------------- settings
st.session_state.setdefault("view", "chat")          # "chat" or "settings"
st.session_state.setdefault("settings_tab", "usage")  # "usage" or "account"

USAGE_CARDS = [  # (title, limit kind) on Settings → Usage → Overview
    ("Daily limit", "question_day"),
    ("Short-term limit (10 minutes)", "question"),
]


def visitor_timezone():
    """The browser's time zone (Streamlit >= 1.44), so charts show local hours even on a UTC server."""
    try:
        return ZoneInfo(st.context.timezone)
    except Exception:
        return timezone.utc


def duration_text(seconds: int) -> str:
    h, m = divmod(max(0, seconds) // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m" if m else f"{max(1, seconds)}s"


def usage_card(title: str, kind: str) -> str:
    u = limiter().usage(kind, USER_KEY)
    left_pct = max(0, round(100 * (u["limit"] - u["used"]) / u["limit"]))
    reset = f"Resets in {duration_text(u['full_reset_in'])}" if u["used"] else "Full allowance available"
    level = "full" if left_pct == 0 else "low" if left_pct <= 20 else ""
    return (
        f'<div class="set-card"><div class="set-card-title">{title}</div>'
        f'<div class="set-row"><span>{reset}</span><span>{left_pct}% remaining</span></div>'
        f'<div class="set-bar {level}"><i style="width:{left_pct}%"></i></div></div>'
    )


def model_card(model: str) -> str:
    paused = model_paused_for(model)
    if not paused:
        return usage_card(model_name(model), f"model_day:{model}")
    return (  # paused by the provider for everyone: show it as empty until the reset
        f'<div class="set-card"><div class="set-card-title">{html.escape(model_name(model))}</div>'
        f'<div class="set-row"><span>Usage limit reached. Resets in {duration_text(paused)}</span><span>0% remaining</span></div>'
        f'<div class="set-bar full"><i style="width:0%"></i></div></div>'
    )


def session_answers() -> list[dict]:
    """Every answer in this session's chats (current chat included, counted once)."""
    ss = st.session_state
    chats = {cid: c["messages"] for cid, c in ss.chats.items()}
    chats[ss.chat_id] = ss.messages
    return [m for msgs in chats.values() for m in msgs if m.get("kind") == "answer"]


def render_analytics():
    answers = session_answers()
    timed = [m["seconds"] for m in answers if m.get("seconds")]
    models = [AVAILABLE_MODELS.get(m["model"], (m["model"],))[0] for m in answers if m.get("model")]
    used_today = limiter().usage("question_day", USER_KEY)["used"]
    tiles = [
        ("Questions today", str(used_today)),
        ("Answers this session", str(len(answers))),
        ("Average answer time", f"{sum(timed) / len(timed):.1f}s" if timed else "–"),
        ("Most used model", max(set(models), key=models.count) if models else "–"),
    ]
    st.markdown(
        '<div class="set-page"><h2>Your activity</h2><p>Based on today’s questions and this session’s chats.</p>'
        '<div class="set-tiles">'
        + "".join(f'<div class="set-tile"><span>{k}</span><b>{html.escape(v)}</b></div>' for k, v in tiles)
        + "</div></div>",
        unsafe_allow_html=True,
    )
    times = limiter().events("question_day", USER_KEY)
    if times:
        now = time.time()
        hours_ago = [int((now - t) // 3600) for t in times]
        df = pd.DataFrame({"hours_ago": list(range(23, -1, -1))})
        df["Questions"] = [hours_ago.count(h) for h in df["hours_ago"]]
        tz = visitor_timezone()
        df["Hour"] = [datetime.fromtimestamp(now - h * 3600, tz).strftime("%H:00") for h in df["hours_ago"]]
        chart = (
            alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#3B82F6")
            .encode(x=alt.X("Hour:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
                    y=alt.Y("Questions:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                    tooltip=["Hour", "Questions"])
            .properties(height=220, title=alt.Title("Questions in the last 24 hours", anchor="start"))
            .configure(background="transparent").configure_view(strokeWidth=0)
            .configure_title(fontSize=14, fontWeight=600)
        )
        with st.container(key="set_chart"):
            st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown('<div class="set-page"><p>No questions yet today. Your activity will show up here.</p></div>',
                    unsafe_allow_html=True)


def render_settings():
    tab = st.session_state.settings_tab
    if tab == "account":
        method = {"google": "Google", "firebase": "Email and password"}.get(user.get("method"), "Email and password")
        rows = [("Name", user["name"]), ("Email", user["email"]), ("Signed in with", method)]
        st.markdown(
            '<div class="set-page"><h1>Account</h1><div class="set-card">'
            + "".join(f'<div class="set-kv"><span>{k}</span><span>{html.escape(v)}</span></div>' for k, v in rows)
            + "</div></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="set_logout"):
            if st.button("Log out", icon=":material/logout:", key="set_logout_btn"):
                auth.logout()
                st.rerun()
        return
    st.markdown('<div class="set-page"><h1>Usage</h1></div>', unsafe_allow_html=True)
    with st.container(key="usage_tabs"):
        view = st.segmented_control("View", ["Overview", "Analytics"], default="Overview",
                                    key="usage_view", label_visibility="collapsed") or "Overview"
    if view == "Analytics":
        return render_analytics()
    st.markdown(
        '<div class="set-page"><h2>Question limits</h2>'
        "<p>Shared across all your chats and PDFs. These limits keep the app’s AI quota from running out "
        "for everyone; each one refills gradually after you use it.</p>"
        + "".join(usage_card(t, k) for t, k in USAGE_CARDS)
        + "<h2 class=\"set-h2-gap\">Model limits</h2>"
        "<p>Each model has its own daily allowance. If you use up any one of them, all models are paused "
        "for you until it resets, so the limits can't be bypassed by switching models.</p>"
        + (f'<div class="set-alert">{html.escape(lock_message(user_model_lock()))}</div>' if user_model_lock() else "")
        + "".join(model_card(m) for m in MODEL_IDS)
        + "</div>",
        unsafe_allow_html=True,
    )


def settings_sidebar():
    with st.container(key="settings_nav"):
        if st.button("Back to app", icon=":material/arrow_back:", key="set_back", use_container_width=True):
            st.session_state.view = "chat"
            st.rerun()
        st.markdown('<p class="side-label">Settings</p>', unsafe_allow_html=True)
        for tab, label, icon in (("usage", "Usage", ":material/speed:"), ("account", "Account", ":material/person:")):
            if st.button(label, icon=icon, key=f"set_{tab}", use_container_width=True):
                st.session_state.settings_tab = tab
                st.rerun()


def avatar_css(name: str) -> str:
    """Initials avatar as an SVG background inside the profile button, so it always lines up with the name."""
    letters = "".join(ch for ch in auth.initials(name) if ch.isalnum())[:2] or "?"
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
        "<circle cx='16' cy='16' r='16' fill='%232F7CF6'/>"
        "<text x='16' y='16' dy='.35em' text-anchor='middle' fill='white' "
        "font-family='Segoe UI,Arial,sans-serif' font-size='12' font-weight='600'>"
        f"{html.escape(letters)}</text></svg>"
    )
    return f'.st-key-profile [data-testid="stPopover"] button {{ background-image: url("data:image/svg+xml,{svg}"); }}'


def email_css(email: str) -> str:
    """Shows the email in small text under the name in the profile row (ChatGPT shows the plan there)."""
    safe = "".join(ch for ch in email if ch.isalnum() or ch in "@._+-")[:60]
    return f'.st-key-profile [data-testid="stPopover"] button p::after {{ content: "{safe}"; }}'


# ---------------------------------------------------------------- composer
MODEL_IDS = list(AVAILABLE_MODELS)
if default_model() not in MODEL_IDS:  # a custom GROQ_MODEL from .env is still offered
    AVAILABLE_MODELS[default_model()] = (default_model().split("/")[-1], "Set in your .env")
    MODEL_IDS.insert(0, default_model())
    register_model_limits({default_model(): (50, 500)})
# Start on whichever model has the most capacity left, unless the user picked one this session
st.session_state.setdefault("model_chosen", False)
st.session_state.setdefault("model_pick", best_model())


def composer(key: str, placeholder: str):
    with st.container(key=key):
        return st.chat_input(placeholder, key=f"{key}_input", accept_file="multiple", file_type=["pdf"])


def model_menu():
    """Model picker in the top-left of the chat area, like ChatGPT's."""
    with st.container(key="model_top"):
        lock = user_model_lock()
        with st.popover(AVAILABLE_MODELS[st.session_state.model_pick][0]):
            if lock:
                st.markdown(f'<p class="model-lock">{html.escape(lock_message(lock))}</p>', unsafe_allow_html=True)
            st.radio(
                "Model", MODEL_IDS, key="model_pick", label_visibility="collapsed", disabled=bool(lock),
                on_change=lambda: st.session_state.update(model_chosen=True),
                format_func=lambda m: AVAILABLE_MODELS[m][0],
                captions=[model_caption(m) for m in MODEL_IDS],
            )


def model_caption(model: str) -> str:
    """Model description, or why it can't be used right now."""
    paused = model_paused_for(model)
    if paused:
        return f"Limit reached, resets in {duration_text(paused)}"
    u = limiter().usage(f"model_day:{model}", USER_KEY)
    if u["used"] >= u["limit"]:
        return f"Daily limit reached, resets in {duration_text(u['full_reset_in'])}"
    return AVAILABLE_MODELS[model][1]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    if st.session_state.view == "settings":
        settings_sidebar()
    else:
        st.markdown('<div class="brand">Ask your PDF</div>', unsafe_allow_html=True)
        save_current_chat()
        ss = st.session_state
        # Highlight the open chat in Recents (New chat only highlights on hover)
        if ss.messages:
            st.markdown(f"<style>.st-key-recents .st-key-chat_{ss.chat_id} button,"
                        f".st-key-recents .st-key-chat_{ss.chat_id} button:focus {{ background: var(--hover) !important;"
                        f" padding-right: 2.2rem !important; }}</style>",
                        unsafe_allow_html=True)

        if st.button("New chat", icon=":material/edit_square:", key="new_chat", use_container_width=True):
            new_chat()
            st.rerun()

        recent = sorted(ss.chats, key=lambda c: ss.chats[c]["updated"], reverse=True)
        if recent:
            st.markdown('<p class="side-label">Recents</p>', unsafe_allow_html=True)
            with st.container(key="recents"):
                for cid in recent:
                    if st.button(ss.chats[cid]["title"], key=f"chat_{cid}", use_container_width=True):
                        if cid != ss.chat_id:
                            open_chat(cid)
                        st.rerun()
                    if cid == ss.chat_id:  # one menu at a time keeps the sidebar light
                        with st.container(key="chat_menu"):
                            with st.popover("⋯"):
                                new_title = st.text_input("Rename", value=ss.chats[cid]["title"], max_chars=40,
                                                          key=f"rename_{cid}")
                                if st.button("Save name", key="rename_save", use_container_width=True):
                                    rename_chat(cid, new_title)
                                    st.rerun()
                                if st.button("Delete chat", key="chat_delete", icon=":material/delete:",
                                             use_container_width=True):
                                    delete_chat(cid)
                                    st.rerun()

    with st.container(key="profile"):
        st.markdown(f"<style>{avatar_css(user['name'])}{email_css(user['email'])}</style>", unsafe_allow_html=True)
        with st.popover(user["name"], use_container_width=True):
            st.markdown(f'<p class="pf-email">{html.escape(user["email"])}</p>', unsafe_allow_html=True)
            if st.button("Settings", icon=":material/settings:", key="open_settings", use_container_width=True):
                st.session_state.view = "settings"
                st.rerun()
            if st.button("Log out", icon=":material/logout:", key="logout", use_container_width=True):
                auth.logout()
                st.rerun()


# ---------------------------------------------------------------- main
if st.session_state.view == "settings":
    render_settings()
    st.stop()

submission = None
model_menu()

if not st.session_state.messages:
    st.markdown(
        '<div class="hero"><h1>What do you want to know?</h1>'
        "<p>Attach a PDF with + and ask anything about it.</p></div>",
        unsafe_allow_html=True,
    )
    submission = composer("composer_hero", "Ask about your PDF")
else:
    if st.session_state.doc_name:
        st.markdown(f'<div class="doc-title">Chatting about <b>{html.escape(st.session_state.doc_name)}</b></div>',
                    unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                # User text is escaped and shown as plain text, never rendered as markup (item 15)
                files = msg.get("attachments") or ([msg["attachment"]] if msg.get("attachment") else [])
                if files:
                    st.markdown("".join(f'<div class="attach"><i>PDF</i>{html.escape(f)}</div>' for f in files),
                                unsafe_allow_html=True)
                if msg.get("content"):
                    st.markdown(f'<div class="umsg">{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
            else:
                render_answer(msg["content"])
                render_sources(msg.get("sources", msg.get("pages")))

    question = st.session_state.to_answer
    if question:
        st.session_state.to_answer = None
        used = st.session_state.answer_model or st.session_state.model_pick
        with st.chat_message("assistant"):
            text, docs, seconds = stream_answer(question, used)
            answer, followups = split_followups(text)
            sources = sources_label(docs)
            history = st.session_state.chat_history_store.messages
            if history and history[-1].type == "ai":
                history[-1].content = answer  # keep follow-up suggestions out of the model's memory (fewer tokens)
            render_answer(answer)
            render_sources(sources)
        say(answer, sources=sources, followups=followups, kind="answer" if docs else "notice",
            model=used, seconds=round(seconds, 1))

    # Actions under the latest answer: follow-up suggestions, regenerate, export
    last = st.session_state.messages[-1] if st.session_state.messages else {}
    if last.get("kind") == "answer" and not st.session_state.to_answer:
        if last.get("followups"):
            with st.container(key="followups"):
                for i, q in enumerate(last["followups"]):
                    if st.button(q, key=f"fu_{i}", icon=":material/subdirectory_arrow_right:"):
                        handle_submission(q, None)
                        st.rerun()
        with st.container(key="actions", horizontal=True):
            if st.button("Regenerate", key="regen", icon=":material/refresh:"):
                regenerate_last()
                st.rerun()
            st.download_button("Export chat", chat_as_markdown(), file_name="chat.md", mime="text/markdown",
                               key="export", icon=":material/download:")

    with getattr(st, "_bottom", None) or st.container():  # pinned to the bottom like a chat app
        submission = composer("composer_bottom", "Ask anything")

if submission:
    handle_submission(getattr(submission, "text", ""), getattr(submission, "files", None))
    st.rerun()
