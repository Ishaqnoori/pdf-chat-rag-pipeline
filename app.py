import hashlib
import html
import os
import re
import tempfile
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

st.set_page_config(
    page_title="Ask your PDF",
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.png"),
    layout="wide",
)

# ---------------------------------------------------------------- config (item 1)
# Keys come from .env locally or the host's Secrets when deployed; never from code.
# .env loads first so placeholder values in secrets.toml can't override real keys.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))  # always the .env next to app.py
CONFIG_KEYS = ["GOOGLE_API_KEY", "GROQ_API_KEY", "FIREBASE_API_KEY", "GROQ_MODEL", "GEMINI_EMBEDDING_MODEL",
               "ADMIN_EMAILS", "CONTACT_EMAIL", "SENTRY_DSN", "SESSION_SECRET"]


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

# Imported after the config loop above, because these modules read the keys at import time
from chain import get_conversational_chain  # noqa: E402
from rag_pipeline import build_index, get_retriever, merge_indexes  # noqa: E402
from utils import auth, legal, monitoring, quota, session, store  # noqa: E402
from utils.brand import logo_svg  # noqa: E402
from utils.citations import sources_label  # noqa: E402
from utils.llm import AVAILABLE_MODELS, MODEL_IDS, is_rate_limit, retry_after_seconds  # noqa: E402
from utils.loaders import validate_pdf  # noqa: E402
from utils.render import render_answer, split_followups, stream_preview  # noqa: E402
from utils.security import MAX_QUESTION, clean_text, limiter, log, wait_text  # noqa: E402
from utils.stats import stats  # noqa: E402
from views import login as login_view  # noqa: E402
from views import settings as settings_view  # noqa: E402

monitoring.init()  # error alerts by email (Sentry), only if SENTRY_DSN is set


@st.cache_resource(show_spinner=False)
def stylesheet() -> str:
    """assets/style.css, minified once per process (it's re-sent to the browser on every click)."""
    with open(os.path.join(APP_DIR, "assets", "style.css"), encoding="utf-8") as f:
        css = re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)  # comments
    css = re.sub(r"\s+", " ", css)
    return re.sub(r"\s*([{};:,>])\s*", r"\1", css).strip()

st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)

missing = [k for k in CONFIG_KEYS[:3] if not os.getenv(k)]
if missing:
    # Names of missing settings only; values are never shown (item 1, 17)
    st.error(
        f"The app isn't configured yet. Missing: {', '.join(missing)}. "
        "Put them in the `.env` file in the same folder as app.py (see `.env.example`), then restart run.bat."
    )
    st.stop()


# ---------------------------------------------------------------- login gate (item 6)


user = auth.current_user()
if user is None:
    login_view.render()
    st.stop()
if "pending_cookie" in st.session_state:
    session.write_cookie(st.session_state.pop("pending_cookie"))  # "keep me signed in" was ticked

# ---------------------------------------------------------------- chat state (per session, item 7)
# The "current chat" lives in these keys; finished chats are kept in st.session_state.chats
# (session only: nothing is written to disk or a database).
CHAT_FIELDS = ("messages", "retriever", "doc_name", "docs", "chat_history_store")
MAX_CHATS = 20
MAX_DOCS = 5             # PDFs per chat
MAX_TOTAL_CHUNKS = 4000  # keeps a multi-PDF chat as fast as one large PDF

for k, v in {"messages": [], "retriever": None, "doc_name": None, "docs": [], "to_answer": None,
             "answer_model": None, "answer_receipt": None, "chats": {}}.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("chat_history_store", InMemoryChatMessageHistory())
st.session_state.setdefault("chat_id", uuid.uuid4().hex)
st.session_state.setdefault("view", "chat")          # "chat" or "settings"
st.session_state.setdefault("settings_tab", "usage")  # usage, account, privacy, terms, admin
USER_KEY = user["uid"] or user["email"]


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
        "doc_names": [d["name"] for d in ss.docs] or saved.get("doc_names", []),
        "title": saved["title"] if saved.get("custom_title") else chat_title(ss.messages),
        "custom_title": saved.get("custom_title", False),
        "updated": saved.get("updated", time.time()) if saved.get("count") == len(ss.messages) else time.time(),
        "count": len(ss.messages),
    }
    while len(ss.chats) > MAX_CHATS:  # drop the oldest
        ss.chats.pop(min(ss.chats, key=lambda c: ss.chats[c]["updated"]))


# ---------------------------------------------------------------- Recents (saved to Firestore, text only)
STORE_UID = str(USER_KEY).replace("/", "_")  # Firestore document ids can't contain "/"


# ---------------------------------------------------------------- terms of service gate
def require_terms():
    """Everyone accepts the current Terms once (again when they change) before using the app."""
    ss = st.session_state
    if ss.get("terms_ok"):
        return
    if store.accepted_terms(STORE_UID) == legal.TERMS_VERSION:  # one read per login
        ss.terms_ok = True
        return
    with st.container(key="legal_page"):
        st.markdown(legal.terms())
        with st.expander("Privacy Policy"):
            st.markdown(legal.privacy())
        agreed = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="gate_agree")
        if st.button("Continue", key="gate_continue", type="primary", disabled=not agreed):
            store.accept_terms(STORE_UID, legal.TERMS_VERSION)
            ss.terms_ok = True
            st.rerun()
    st.stop()


require_terms()


def history_from(messages: list) -> InMemoryChatMessageHistory:
    """Rebuilds the model's short-term memory (last 3 exchanges) from saved messages."""
    history = InMemoryChatMessageHistory()
    pairs, question = [], None
    for m in messages:
        if m["role"] == "user" and m.get("content"):
            question = m["content"]
        elif m.get("kind") == "answer" and question:
            pairs.append((question, m["content"]))
            question = None
    for q, a in pairs[-3:]:
        history.add_user_message(q)
        history.add_ai_message(a)
    return history


def load_saved_chats():
    """Once per login: puts the user's 20 most recent saved chats into Recents."""
    ss = st.session_state
    if ss.get("chats_loaded"):
        return
    ss.chats_loaded = True
    ss.setdefault("saved_signatures", {})
    for c in store.load_chats(STORE_UID):
        msgs = c.get("messages", [])
        names = c.get("doc_names", [])
        ss.chats[c["id"]] = {
            "messages": msgs, "retriever": None, "docs": [], "doc_names": names,
            "doc_name": ", ".join(names) or None, "chat_history_store": history_from(msgs),
            "title": c.get("title", "Chat"), "custom_title": c.get("custom_title", False),
            "updated": c.get("updated", 0), "count": len(msgs),
        }
        ss.saved_signatures[c["id"]] = chat_signature(ss.chats[c["id"]])


def chat_signature(chat: dict) -> tuple:
    """Changes whenever something worth saving changes: messages, title or a feedback vote."""
    return (chat["count"], chat["title"], tuple(m.get("rating") for m in chat["messages"]))


def persist_current_chat():
    """Updates Recents and, if the open chat changed, saves it to Firestore (one background write)."""
    ss = st.session_state
    save_current_chat()
    if not store.enabled() or not ss.messages:
        return
    chat = ss.chats.get(ss.chat_id)
    if not chat:
        return
    sig = chat_signature(chat)
    if ss.setdefault("saved_signatures", {}).get(ss.chat_id) == sig:
        return
    ss.saved_signatures[ss.chat_id] = sig
    store.save_in_background(store.save_chat, STORE_UID, ss.chat_id, chat["title"], list(chat["messages"]),
                             chat.get("doc_names", []), chat.get("custom_title", False), chat["updated"])


def reset_current_chat():
    st.session_state.messages = []
    st.session_state.retriever = None
    st.session_state.doc_name = None
    st.session_state.docs = []
    st.session_state.to_answer = None
    st.session_state.chat_history_store = InMemoryChatMessageHistory()  # new object: old chats keep theirs
    st.session_state.chat_id = uuid.uuid4().hex
    st.session_state.doc_names = []


def new_chat():
    save_current_chat()
    reset_current_chat()
    if not st.session_state.get("model_chosen"):
        st.session_state.model_pick = quota.best_model(USER_KEY)  # spread usage across models


def delete_chat(chat_id: str):
    st.session_state.chats.pop(chat_id, None)
    st.session_state.get("saved_signatures", {}).pop(chat_id, None)
    store.save_in_background(store.delete_chat, STORE_UID, chat_id)
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
    st.session_state.doc_names = chat.get("doc_names", [])


# A 300-page PDF's index is roughly 25 MB (3,072-dim vectors), so caches stay small for ~1 GB hosts.
# ---------------------------------------------------------------- documents (PDF → search index)
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


# ---------------------------------------------------------------- answers
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
        quota.release(model, USER_KEY, st.session_state.answer_receipt)  # no answer, so it doesn't count
        if is_rate_limit(e):
            wait = retry_after_seconds(e)
            limiter().block(f"model:{model}", wait)
            log.warning("model=%s rate-limited by provider for %ss", model, wait)
            text = f"{quota.model_name(model)} has reached its usage limit. It resets in {wait_text(wait)}."
        else:
            log.exception("Answer generation failed")
            text = "I couldn't get an answer just now. Please try again."
        docs = []
    box.empty()
    seconds = time.perf_counter() - started
    stats().record_answer(model, seconds, ok=bool(docs))  # for the owner's Admin page
    log.info("answer model=%s seconds=%.2f chars=%d chunks=%d",  # observability, server log only
             model, seconds, len(text), len(docs))
    return text, docs, seconds


def save_feedback(index: int, key: str):
    """Stores 👍/👎 for the answer at messages[index] (once; changing the vote updates it)."""
    rating = st.session_state.get(key)
    msgs = st.session_state.messages
    if rating is None or index >= len(msgs):
        return
    msg = msgs[index]
    msg["rating"] = rating
    question = next((m.get("content") for m in reversed(msgs[:index]) if m["role"] == "user" and m.get("content")), "")
    entry = {"rating": rating, "question": question[:500], "answer": msg["content"][:2000],
             "model": msg.get("model"), "sources": msg.get("sources", "")}
    stats().record_feedback(entry)
    store.save_in_background(store.add_feedback, {**entry, "uid": STORE_UID,
                                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")})


def feedback_widget(index: int):
    key = f"fb_{st.session_state.chat_id}_{index}"
    with st.container(key=f"fbwrap_{index}"):
        st.feedback("thumbs", key=key, on_change=save_feedback, args=(index, key))


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
    ok, result = quota.reserve(st.session_state.model_pick, USER_KEY)
    if not ok:
        return say(result)  # keep the old answer if we can't make a new one
    st.session_state.answer_model, st.session_state.answer_receipt = st.session_state.model_pick, result
    msgs.pop()
    history = st.session_state.chat_history_store
    history.messages = history.messages[:-2]  # the last question/answer pair
    st.session_state.to_answer = question


# ---------------------------------------------------------------- user input (questions and attachments)
def add_pdf(pdf) -> bool:
    """Validates, indexes and attaches one uploaded PDF. Reports problems in the chat; True if it was added."""
    file_name = clean_text(pdf.name, 120)
    ok, message = limiter().take("upload", USER_KEY)
    if not ok:
        say(message)
        return False
    data = pdf.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    ocr_receipts = []

    def take_ocr():
        ok, result = limiter().take("ocr", USER_KEY)
        if not ok:
            return False, result
        ocr_receipts.append(result)
        return True, None

    try:
        validate_pdf(data)
        with st.spinner(f"Reading {file_name}… (scanned PDFs take up to a minute)"):
            # A scanned PDF only spends OCR quota if it actually needs OCR (checked inside)
            index = process_pdf(data, file_name, file_hash, take_ocr)
    except Exception as e:
        for receipt in ocr_receipts:
            limiter().undo("ocr", USER_KEY, receipt)  # the read failed, so it shouldn't count
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

    ok, result = quota.reserve(st.session_state.model_pick, USER_KEY)
    if not ok:
        return say(result)
    st.session_state.answer_model, st.session_state.answer_receipt = st.session_state.model_pick, result
    st.session_state.to_answer = text


# ---------------------------------------------------------------- profile row


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


# ---------------------------------------------------------------- composer and model menu
# Start on whichever model has the most capacity left, unless the user picked one this session
st.session_state.setdefault("model_chosen", False)
st.session_state.setdefault("model_pick", quota.best_model(USER_KEY))


def composer(key: str, placeholder: str):
    with st.container(key=key):
        return st.chat_input(placeholder, key=f"{key}_input", accept_file="multiple", file_type=["pdf"])


def model_menu():
    """Model picker in the top-left of the chat area, like ChatGPT's."""
    with st.container(key="model_top"):
        lock = quota.user_lock(USER_KEY)
        with st.popover(AVAILABLE_MODELS[st.session_state.model_pick][0]):
            if lock:
                st.markdown(f'<p class="model-lock">{html.escape(quota.lock_message(lock))}</p>',
                            unsafe_allow_html=True)
            st.radio(
                "Model", MODEL_IDS, key="model_pick", label_visibility="collapsed", disabled=bool(lock),
                on_change=lambda: st.session_state.update(model_chosen=True),
                format_func=lambda m: AVAILABLE_MODELS[m][0],
                captions=[quota.caption(m, USER_KEY) for m in MODEL_IDS],
            )


# What the settings pages need to know about the signed-in user
SETTINGS = settings_view.Ctx(user=user, user_key=USER_KEY, store_uid=STORE_UID, save_chat=save_current_chat)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    if st.session_state.view == "settings":
        settings_view.sidebar(SETTINGS)
    else:
        st.markdown(f'<div class="brand">{logo_svg(22)}<span>Ask your PDF</span></div>', unsafe_allow_html=True)
        load_saved_chats()
        persist_current_chat()
        ss = st.session_state
        # Highlight the open chat in Recents (New chat only highlights on hover)
        if ss.messages:
            st.markdown(f"<style>.st-key-recents .st-key-chat_{ss.chat_id} button,"
                        f".st-key-recents .st-key-chat_{ss.chat_id} button:focus"
                        f" {{ background: var(--hover) !important;"
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
    settings_view.render(SETTINGS)
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
    if st.session_state.retriever is None and st.session_state.get("doc_names"):
        st.markdown(
            '<div class="reattach">PDFs aren’t stored, only the chat text. To ask follow-up questions, attach '
            f'<b>{html.escape(", ".join(st.session_state.doc_names))}</b> again with <b>+</b>.</div>',
            unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
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
                if msg.get("kind") == "answer":
                    feedback_widget(i)

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
            if docs:
                feedback_widget(len(st.session_state.messages))  # index the answer is about to get
        say(answer, sources=sources, followups=followups, kind="answer" if docs else "notice",
            model=used, seconds=round(seconds, 1))
        persist_current_chat()  # save right away, even if the user closes the tab now

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
