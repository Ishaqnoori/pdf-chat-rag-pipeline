"""
Persistent storage in Cloud Firestore (free Spark plan), accessed only from the server with a
service account, so users can never read each other's data. Stores:
  - usage counters (so limits survive restarts)      doc  system/limits
  - each user's chats: text only, never PDFs          coll users/{uid}/chats/{chat_id}
  - answer feedback                                   coll feedback
If no credentials are configured, every function quietly does nothing and the app runs in memory only.

Credentials: a local file (FIREBASE_CREDENTIALS, default firebase-service-account.json) or, when
deployed, a [firebase_service_account] table in the host's Secrets.
"""
import json
import os
import threading
import time

import streamlit as st

from utils.security import log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SAVED_MESSAGES = 60     # per chat (Firestore documents are capped at 1 MB)
MAX_SAVED_CHARS = 8000      # per message
MAX_CHATS_LOADED = 20


def _credentials_info():
    try:
        if "firebase_service_account" in st.secrets:
            return dict(st.secrets["firebase_service_account"])
    except Exception:
        pass
    path = os.getenv("FIREBASE_CREDENTIALS", "firebase-service-account.json")
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource(show_spinner=False)
def _client():
    info = _credentials_info()
    if not info:
        log.info("Firestore not configured: running in memory only")
        return None
    try:
        from google.cloud import firestore
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(project=info.get("project_id"), credentials=creds)
    except Exception:
        log.exception("Firestore connection failed: running in memory only")
        return None


def enabled() -> bool:
    return _client() is not None


def _safe(default=None):
    """Storage problems are logged, never shown to users or allowed to break the app."""
    def wrap(fn):
        def inner(*a, **k):
            if not enabled():
                return default
            try:
                return fn(*a, **k)
            except Exception:
                log.exception("Firestore %s failed", fn.__name__)
                return default
        return inner
    return wrap


# ---------------------------------------------------------------- usage counters
@_safe(default=None)
def load_limits() -> dict | None:
    doc = _client().collection("system").document("limits").get()
    return doc.to_dict() if doc.exists else None


@_safe()
def save_limits(state: dict):
    _client().collection("system").document("limits").set(state)


# ---------------------------------------------------------------- chats (text only)
def _clean_message(m: dict) -> dict:
    keep = ("role", "content", "attachments", "sources", "kind", "model", "seconds", "rating")
    out = {k: m[k] for k in keep if k in m and m[k] is not None}
    if m.get("attachment") and "attachments" not in out:  # older message format
        out["attachments"] = [m["attachment"]]
    out["content"] = str(out.get("content", ""))[:MAX_SAVED_CHARS]
    return out


@_safe(default=[])
def load_chats(uid: str) -> list[dict]:
    from google.cloud import firestore

    docs = (_client().collection("users").document(uid).collection("chats")
            .order_by("updated", direction=firestore.Query.DESCENDING).limit(MAX_CHATS_LOADED).stream())
    return [{"id": d.id, **d.to_dict()} for d in docs]


@_safe()
def save_chat(uid: str, chat_id: str, title: str, messages: list, doc_names: list, custom_title: bool, updated: float):
    _client().collection("users").document(uid).collection("chats").document(chat_id).set({
        "title": title[:80],
        "custom_title": bool(custom_title),
        "updated": updated,
        "doc_names": list(doc_names)[:10],
        "messages": [_clean_message(m) for m in messages[-MAX_SAVED_MESSAGES:]],
    })


@_safe()
def delete_chat(uid: str, chat_id: str):
    _client().collection("users").document(uid).collection("chats").document(chat_id).delete()


@_safe(default=[])
def export_user_chats(uid: str) -> list[dict]:
    """Every saved chat of a user (for 'Export my data')."""
    chats = _client().collection("users").document(uid).collection("chats").limit(500).stream()
    return [{"id": d.id, **d.to_dict()} for d in chats]


@_safe(default=False)
def delete_user_data(uid: str) -> bool:
    """Deletes a user's chats, sessions and feedback (for 'Delete my account')."""
    user = _client().collection("users").document(uid)
    for sub in ("chats", "sessions"):
        for d in user.collection(sub).limit(1000).stream():
            d.reference.delete()
    user.delete()
    from google.cloud.firestore_v1.base_query import FieldFilter

    for d in _client().collection("feedback").where(filter=FieldFilter("uid", "==", uid)).limit(1000).stream():
        d.reference.delete()
    return True


# ---------------------------------------------------------------- login sessions ("keep me signed in")
@_safe(default=None)
def session_secret() -> str | None:
    """A random signing secret kept in Firestore, created on first use (so no extra setup is needed)."""
    import secrets

    ref = _client().collection("system").document("config")
    doc = ref.get()
    if doc.exists and len((doc.to_dict() or {}).get("session_secret", "")) >= 32:
        return doc.to_dict()["session_secret"]
    value = secrets.token_urlsafe(48)
    ref.set({"session_secret": value}, merge=True)
    return value


@_safe(default=False)
def rotate_session_secret() -> bool:
    """Replaces the signing secret, which invalidates every "keep me signed in" cookie at once."""
    import secrets

    _client().collection("system").document("config").set({"session_secret": secrets.token_urlsafe(48)}, merge=True)
    return True


@_safe()
def create_session(uid: str, sid: str, expires: float):
    _client().collection("users").document(uid).collection("sessions").document(sid).set({"expires": expires})


@_safe(default=False)
def session_active(uid: str, sid: str) -> bool:
    doc = _client().collection("users").document(uid).collection("sessions").document(sid).get()
    return doc.exists and (doc.to_dict() or {}).get("expires", 0) > time.time()


@_safe()
def end_session(uid: str, sid: str):
    _client().collection("users").document(uid).collection("sessions").document(sid).delete()


# ---------------------------------------------------------------- terms of service acceptance
@_safe(default=None)
def accepted_terms(uid: str) -> str | None:
    """The Terms version this user last accepted (None if never)."""
    doc = _client().collection("users").document(uid).get()
    return (doc.to_dict() or {}).get("terms_version") if doc.exists else None


@_safe()
def accept_terms(uid: str, version: str):
    _client().collection("users").document(uid).set(
        {"terms_version": version, "terms_accepted_at": time.time()}, merge=True)


# ---------------------------------------------------------------- feedback
@_safe()
def add_feedback(entry: dict):
    _client().collection("feedback").add(entry)


@_safe(default=None)
def recent_feedback(limit: int = 20) -> list[dict] | None:
    from google.cloud import firestore

    docs = _client().collection("feedback").order_by("time", direction=firestore.Query.DESCENDING).limit(limit).stream()
    return [d.to_dict() for d in docs]


# ---------------------------------------------------------------- background saves
def save_in_background(fn, *args):
    """Runs a save without making the user wait; storage errors are logged inside fn."""
    if not enabled():
        return
    thread = threading.Thread(target=fn, args=args, daemon=True)
    try:  # lets the thread use Streamlit's caches without "missing ScriptRunContext" warnings
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
        add_script_run_ctx(thread, get_script_run_ctx())
    except Exception:
        pass
    thread.start()
