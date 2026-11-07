"""
Shared security helpers: server-wide rate limiting, input validation,
output sanitising and a safe logger. Nothing here is user-facing text.
"""
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque

import streamlit as st

log = logging.getLogger("askpdf")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------- limits (item 11, 12, 16)
MAX_NAME = 80
MAX_EMAIL = 254
MIN_PASSWORD, MAX_PASSWORD = 8, 128
MAX_QUESTION = 4000

LIMITS = {
    # name: (max events, window seconds)
    "login_email": (5, 15 * 60),    # failed logins per email
    "login_ip": (20, 15 * 60),      # failed logins per IP
    "signup_ip": (3, 60 * 60),      # new accounts per IP
    "email_send": (3, 60 * 60),     # reset/verification emails per address
    "question": (30, 10 * 60),      # questions per user
    "upload": (10, 60 * 60),        # PDF uploads per user
}

# ---------------------------------------------------------------- usage limits (protect the shared API quota)
DAY, MINUTE = 24 * 60 * 60, 60


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


# Defaults sit safely under Groq's and Gemini's free tiers; raise them in .env after upgrading a plan.
LIMITS.update({
    "question_day": (_env_int("LIMIT_QUESTIONS_PER_USER_DAY", 100), DAY),
    "upload_day": (_env_int("LIMIT_UPLOADS_PER_USER_DAY", 20), DAY),
    "app_question_minute": (_env_int("LIMIT_APP_QUESTIONS_PER_MINUTE", 20), MINUTE),
    "app_question_day": (_env_int("LIMIT_APP_QUESTIONS_PER_DAY", 800), DAY),
    "app_upload_day": (_env_int("LIMIT_APP_UPLOADS_PER_DAY", 150), DAY),
    "ocr_day": (_env_int("LIMIT_SCANS_PER_USER_DAY", 5), DAY),        # scanned PDFs cost a vision call each
    "app_ocr_day": (_env_int("LIMIT_APP_SCANS_PER_DAY", 40), DAY),
})

# What each kind of request counts against, and what to tell the user when it's full
QUOTAS = {
    "question": [("question", "user", "You're asking very quickly. Try again in {wait}."),
                 ("question_day", "user", "You've reached today's limit of {limit} questions. It resets in {wait}."),
                 ("app_question_minute", "app", "The app is busy right now. Try again in {wait}."),
                 ("app_question_day", "app", "The app has reached its daily usage limit. Please try again in {wait}.")],
    "upload": [("upload", "user", "Upload limit reached. Try again in {wait}."),
               ("upload_day", "user", "You've reached today's limit of {limit} uploads. It resets in {wait}."),
               ("app_upload_day", "app", "The app has reached its daily upload limit. Please try again in {wait}.")],
    "ocr": [("ocr_day", "user", "This is a scanned PDF, and you've used today's {limit} text-recognition reads. "
                                "Try again in {wait}, or upload a PDF with selectable text."),
            ("app_ocr_day", "app", "Text recognition for scanned PDFs has reached today's limit. Try again in {wait}.")],
}


class RateLimiter:
    """Sliding-window counter shared by every session on this server."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, kind: str, key: str, record: bool = True):
        """Returns (allowed, retry_after_seconds). Records the event when allowed and record=True."""
        limit, window = LIMITS[kind]
        now = time.time()
        bucket_key = f"{kind}:{key}"
        with self._lock:
            q = self._hits[bucket_key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False, int(window - (now - q[0])) + 1
            if record:
                q.append(now)
            return True, 0

    def take(self, action: str, user_key: str):
        """
        Checks every limit for `action` ("question" or "upload") and records the event only if all of
        them have room, so a refused request never uses up allowance. Returns (allowed, message).
        """
        now = time.time()
        with self._lock:
            buckets = []
            for kind, scope, message in QUOTAS[action]:
                limit, window = LIMITS[kind]
                q = self._hits[f"{kind}:{user_key if scope == 'user' else 'app'}"]
                while q and now - q[0] > window:
                    q.popleft()
                if len(q) >= limit:
                    wait = int(window - (now - q[0])) + 1
                    return False, message.format(limit=limit, wait=wait_text(wait))
                buckets.append(q)
            for q in buckets:
                q.append(now)
        return True, None

    def remaining(self, kind: str, key: str) -> tuple[int, int]:
        """(left, limit) for one limit, e.g. questions left today for a user."""
        limit, window = LIMITS[kind]
        now = time.time()
        with self._lock:
            q = self._hits.get(f"{kind}:{key}", deque())
            used = sum(1 for t in q if now - t <= window)
        return max(0, limit - used), limit

    def usage(self, kind: str, key: str) -> dict:
        """{"used", "limit", "full_reset_in"} for one limit; full_reset_in is seconds until everything is free again."""
        limit, window = LIMITS[kind]
        now = time.time()
        with self._lock:
            recent = [t for t in self._hits.get(f"{kind}:{key}", deque()) if now - t <= window]
        return {"used": len(recent), "limit": limit,
                "full_reset_in": int(window - (now - recent[-1])) + 1 if recent else 0}

    def events(self, kind: str, key: str) -> list[float]:
        """Timestamps of this key's events still inside the limit's window (for the usage chart)."""
        limit, window = LIMITS[kind]
        now = time.time()
        with self._lock:
            return [t for t in self._hits.get(f"{kind}:{key}", deque()) if now - t <= window]

    def record(self, kind: str, key: str):
        with self._lock:
            self._hits[f"{kind}:{key}"].append(time.time())

    def reset(self, kind: str, key: str):
        with self._lock:
            self._hits.pop(f"{kind}:{key}", None)


def _code_version(cls) -> str:
    """Changes whenever RateLimiter's methods change, so a running app picks up the new code."""
    import hashlib, inspect
    return hashlib.sha1(inspect.getsource(cls).encode()).hexdigest()[:12]


@st.cache_resource
def _shared_limiter(version: str) -> RateLimiter:
    return RateLimiter()


def limiter() -> RateLimiter:
    """One limiter shared by all sessions; rebuilt automatically if its code is updated."""
    return _shared_limiter(_LIMITER_VERSION)


_LIMITER_VERSION = _code_version(RateLimiter)


def client_ip() -> str:
    """Best-effort client IP (Streamlit >= 1.44); falls back to a per-session id."""
    try:
        ip = st.context.ip_address
        if ip:
            return str(ip)
    except Exception:
        pass
    return "session:" + str(id(st.session_state))


def wait_text(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} seconds"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


# ---------------------------------------------------------------- validation (item 14)
EMAIL_RE = re.compile(r"^[^@\s<>\"']{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,}$")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value, max_len: int) -> str:
    return CONTROL_CHARS.sub("", str(value or "")).strip()[:max_len]


def valid_email(email: str) -> bool:
    return len(email) <= MAX_EMAIL and bool(EMAIL_RE.match(email))


# ---------------------------------------------------------------- output sanitising (item 15)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def sanitize_markdown(text: str) -> str:
    """
    Model output can be steered by text inside an uploaded PDF (prompt injection).
    Remove remote images (silent data exfiltration via URL) and raw HTML, and show
    links as plain text so nobody clicks a disguised URL.
    """
    text = _MD_IMAGE.sub("", text or "")
    text = _HTML_TAG.sub("", text)
    return _MD_LINK.sub(r"\1 (\2)", text)
