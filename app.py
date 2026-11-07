import hashlib
import html
import os
import tempfile
import time
import uuid

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
from utils.llm import AVAILABLE_MODELS, default_model, is_rate_limit  # noqa: E402
from utils.loaders import validate_pdf  # noqa: E402
from utils.render import render_answer, split_followups, stream_preview  # noqa: E402
from utils.security import MAX_QUESTION, clean_text, limiter, log  # noqa: E402

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

for k, v in {"messages": [], "retriever": None, "doc_name": None, "docs": [], "to_answer": None, "chats": {}}.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("chat_history_store", InMemoryChatMessageHistory())
st.session_state.setdefault("chat_id", uuid.uuid4().hex)
USER_KEY = user["uid"] or user["email"]


def get_session_history(session_id: str):
    return st.session_state.chat_history_store


def chat_title(messages) -> str:
    first = next((m for m in messages if m["role"] == "user"), {})
    title = first.get("content") or first.get("attachment") or "New chat"
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


@st.cache_resource(show_spinner=False, max_entries=20, ttl=3600)
def process_pdf(_file_bytes: bytes, filename: str, file_hash: str, _allow_ocr=None):
    """Builds the index once per unique file. The temp copy is deleted immediately."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name
    try:
        return build_index(tmp_path, source_name=filename, allow_ocr=_allow_ocr)
    finally:
        os.remove(tmp_path)


@st.cache_resource(show_spinner=False, max_entries=20, ttl=3600)
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


def stream_answer(question: str):
    """
    Streams the answer into the page as it arrives. If the chosen model is rate-limited before any
    text arrives, retries once with another model. Returns (text, retrieved_docs, model_used).
    """
    box = st.empty()
    box.markdown('<p class="thinking">Thinking…</p>', unsafe_allow_html=True)
    chosen = st.session_state.model_pick
    candidates = [chosen] + [m for m in MODEL_IDS if m != chosen][:1]
    text, docs, used = "", [], chosen
    started = time.perf_counter()
    for model in candidates:
        used, text, docs, last_draw = model, "", [], 0.0
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
            break
        except Exception as e:
            if is_rate_limit(e) and not text and model != candidates[-1]:
                log.warning("model=%s rate-limited, falling back", model)
                box.markdown('<p class="thinking">That model is busy, trying another…</p>', unsafe_allow_html=True)
                continue
            log.exception("Answer generation failed")
            text, docs = (
                "The AI models have reached their usage limit for now. Please try again in a few minutes."
                if is_rate_limit(e) else
                "I couldn't get an answer just now. Try again, or pick another model from the top-left menu."
            ), []
            break
    box.empty()
    log.info("answer model=%s seconds=%.2f chars=%d chunks=%d",  # observability, server log only
             used, time.perf_counter() - started, len(text), len(docs))
    return text, docs, used, time.perf_counter() - started


def render_sources(label):
    if isinstance(label, list):  # messages saved before multi-PDF support
        label = f"{'Page' if len(label) == 1 else 'Pages'} {', '.join(label)}" if label else ""
    if label:
        st.markdown(f'<p class="sources">{html.escape(label)}</p>', unsafe_allow_html=True)


def chat_as_markdown() -> str:
    lines = [f"# Chat about {st.session_state.doc_name or 'a PDF'}", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        body = m.get("content") or (f"[attached {m['attachment']}]" if m.get("attachment") else "")
        lines += [f"**{who}:**", "", body, ""]
    return "\n".join(lines)


def regenerate_last():
    """Drops the last answer (and its memory) and asks the same question again."""
    msgs = st.session_state.messages
    question = next((m["content"] for m in reversed(msgs) if m["role"] == "user" and m.get("content")), None)
    if not question or msgs[-1].get("kind") != "answer":
        return
    ok, message = limiter().take("question", USER_KEY)
    if not ok:
        return say(message)  # keep the old answer if we can't make a new one
    msgs.pop()
    history = st.session_state.chat_history_store
    history.messages = history.messages[:-2]  # the last question/answer pair
    st.session_state.to_answer = question


def handle_submission(text: str, files):
    """Validates the user turn, indexes an attached PDF and queues the question. Answers come after rerun."""
    text = clean_text(text, MAX_QUESTION + 1)
    pdf = files[0] if files else None
    if not text and not pdf:
        return
    file_name = clean_text(pdf.name, 120) if pdf else None
    st.session_state.messages.append({"role": "user", "content": text, "attachment": file_name})

    if len(text) > MAX_QUESTION:
        return say(f"That message is too long. Keep questions under {MAX_QUESTION:,} characters.")

    if pdf:
        ok, message = limiter().take("upload", USER_KEY)
        if not ok:
            return say(message)
        data = pdf.getvalue()
        file_hash = hashlib.sha256(data).hexdigest()
        try:
            validate_pdf(data)
            with st.spinner(f"Reading {file_name}… (scanned PDFs take up to a minute)"):
                # A scanned PDF only spends OCR quota if it actually needs OCR (checked inside)
                index = process_pdf(data, file_name, file_hash, lambda: limiter().take("ocr", USER_KEY))
        except ValueError as e:  # our own, user-safe messages
            return say(f"I couldn't use **{file_name}**. {e}")
        except Exception:
            log.exception("PDF processing failed")  # details stay in the server log (item 17)
            return say(f"Something went wrong while reading **{file_name}**. Please try again.")
        problem = attach_pdf(file_name, file_hash, index)
        if problem:
            return say(problem)
        if index.chunks and index.chunks[0].metadata.get("ocr"):
            say(f"**{file_name}** is a scanned PDF, so I read it with text recognition. "
                "Small recognition errors are possible, so double-check important numbers.", kind="notice")
        if not text:
            n = len(st.session_state.docs)
            return say(f"**{file_name}** is ready. What would you like to know about it?" if n == 1 else
                       f"**{file_name}** added. You can now ask about all {n} PDFs in this chat.")

    if st.session_state.retriever is None:
        return say("Attach a PDF with the **+** button first, then ask your question.")

    ok, message = limiter().take("question", USER_KEY)
    if not ok:
        return say(message)
    st.session_state.to_answer = text


# ---------------------------------------------------------------- settings
st.session_state.setdefault("view", "chat")          # "chat" or "settings"
st.session_state.setdefault("settings_tab", "usage")  # "usage" or "account"

USAGE_CARDS = [  # (title, limit kind) on Settings → Usage → Overview
    ("Daily limit", "question_day"),
    ("Short-term limit (10 minutes)", "question"),
]


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
        df["Hour"] = [time.strftime("%H:00", time.localtime(now - h * 3600)) for h in df["hours_ago"]]
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
st.session_state.setdefault("model_pick", default_model())


def composer(key: str, placeholder: str):
    with st.container(key=key):
        return st.chat_input(placeholder, key=f"{key}_input", accept_file=True, file_type=["pdf"])


def model_menu():
    """Model picker in the top-left of the chat area, like ChatGPT's."""
    with st.container(key="model_top"):
        with st.popover(AVAILABLE_MODELS[st.session_state.model_pick][0]):
            st.radio(
                "Model", MODEL_IDS, key="model_pick", label_visibility="collapsed",
                format_func=lambda m: AVAILABLE_MODELS[m][0],
                captions=[AVAILABLE_MODELS[m][1] for m in MODEL_IDS],
            )


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
                if msg.get("attachment"):
                    st.markdown(f'<div class="attach"><i>PDF</i>{html.escape(msg["attachment"])}</div>',
                                unsafe_allow_html=True)
                if msg.get("content"):
                    st.markdown(f'<div class="umsg">{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
            else:
                render_answer(msg["content"])
                render_sources(msg.get("sources", msg.get("pages")))

    question = st.session_state.to_answer
    if question:
        st.session_state.to_answer = None
        with st.chat_message("assistant"):
            text, docs, used, seconds = stream_answer(question)
            answer, followups = split_followups(text)
            sources = sources_label(docs)
            if used != st.session_state.model_pick and docs:  # say so when the fallback answered
                sources = f"Answered by {AVAILABLE_MODELS[used][0]} (the selected model was busy)" + (
                    f". {sources}" if sources else "")
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
