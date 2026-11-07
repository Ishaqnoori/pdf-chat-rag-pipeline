"""
"Keep me signed in" for email/password users: a signed, expiring token in a browser cookie.

Token = base64(payload) + "." + HMAC-SHA256(payload, secret). The secret lives in Firestore (created
on first use) or SESSION_SECRET. Each token's session id is also recorded in Firestore, so logging
out revokes it on the server even if a copy of the cookie survives. Google sign-ins don't need this:
Streamlit's own login cookie already keeps them signed in.

Streamlit can't set cookies from the server, so a tiny same-origin script writes it; that means the
cookie can't be HttpOnly (see SECURITY.md). Output sanitising keeps scripts out of the page.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

import streamlit as st
import streamlit.components.v1 as components

from utils import store
from utils.security import log

COOKIE = "askpdf_session"
LIFETIME = int(os.getenv("SESSION_DAYS", "3") or 3) * 24 * 60 * 60  # short window limits a stolen cookie's use


def _browser() -> str:
    """A fingerprint of the browser (User-Agent hash): a cookie copied to another browser stops working."""
    try:
        agent = st.context.headers.get("User-Agent", "")
    except Exception:
        agent = ""
    return hashlib.sha256(agent.encode()).hexdigest()[:16]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


_cached_secret: bytes | None = None


def _secret() -> bytes | None:
    """Signing key: SESSION_SECRET, or one generated once and kept in Firestore (read once per process)."""
    global _cached_secret
    if _cached_secret:
        return _cached_secret
    env = os.getenv("SESSION_SECRET", "").strip()
    value = env if len(env) >= 32 else store.session_secret()
    if value:
        _cached_secret = value.encode()
    return _cached_secret


def available() -> bool:
    """Off when SESSION_DAYS=0 or there's no database to record sessions in."""
    return LIFETIME > 0 and store.enabled() and _secret() is not None


def issue(user: dict) -> tuple[str, str] | None:
    """Creates a session for `user`; returns (signed_token, session_id), or None if unavailable."""
    key = _secret()
    if not key:
        return None
    sid = secrets.token_urlsafe(16)
    exp = int(time.time()) + LIFETIME
    payload = _b64(json.dumps({"uid": user["uid"], "email": user["email"], "name": user["name"],
                               "sid": sid, "exp": exp, "ua": _browser()}, separators=(",", ":")).encode())
    store.create_session(user["uid"], sid, exp)
    return payload + "." + _b64(hmac.new(key, payload.encode(), hashlib.sha256).digest()), sid


def verify(token: str) -> dict | None:
    """Returns {"uid","email","name","sid"} for a valid, unexpired, unrevoked token; else None."""
    key = _secret()
    if not key or not token or token.count(".") != 1:
        return None
    payload, signature = token.split(".")
    expected = _b64(hmac.new(key, payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        data = json.loads(_unb64(payload))
    except Exception:
        return None
    if data.get("exp", 0) < time.time() or data.get("ua") != _browser():
        return None
    if not store.session_active(data["uid"], data["sid"]):
        return None
    return data


def sign_everyone_out() -> bool:
    """Rotates the signing secret: every saved login stops working at once (e.g. after a leak)."""
    global _cached_secret
    if not store.rotate_session_secret():
        return False
    _cached_secret = None
    return True


def from_browser() -> dict | None:
    if LIFETIME <= 0:
        return None
    try:
        token = st.context.cookies.get(COOKIE)
    except Exception:
        return None
    user = verify(token) if token else None
    if token and not user:
        log.info("Rejected an invalid or revoked session cookie")
    return user


def write_cookie(token: str | None):
    """Sets (token) or clears (None) the cookie in the browser via a hidden same-origin frame."""
    value = f"{COOKIE}={token}; Max-Age={LIFETIME}" if token else f"{COOKIE}=; Max-Age=0"
    components.html(
        "<script>try{var s=window.location.protocol==='https:'?'; Secure':'';"
        f"window.parent.document.cookie='{value}; Path=/; SameSite=Strict'+s;}}catch(e){{}}</script>",
        height=0,
    )
