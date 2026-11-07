"""
Shared security helpers: server-wide rate limiting, input validation,
output sanitising and a safe logger. Nothing here is user-facing text.
"""
import logging
import os
import re
import threading
import uuid
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
            ("app_ocr_day", "app",
             "Text recognition for scanned PDFs has reached today's limit. Try again in {wait}.")],
}


def register_model_limits(models: dict[str, tuple[int, int]]):
    """
    Per-model allowances: {model_id: (questions_per_user_per_day, questions_per_day_for_the_whole_app)}.
    The app-wide budget keeps one model from using more than its share of the provider quota.
    """
    for model, (per_user, per_app) in models.items():
        LIMITS[f"model_day:{model}"] = (per_user, DAY)
        LIMITS[f"app_model_day:{model}"] = (per_app, DAY)
        QUOTAS[f"model:{model}"] = [
            (f"model_day:{model}", "user",
             "{model} has reached your daily limit of {limit} questions. It resets in {wait}."),
            (f"app_model_day:{model}", "app", "{model} has used today's app-wide allowance. It resets in {wait}."),
        ]


class RateLimiter:
    """Sliding-window counter shared by every session on this server."""

    SAVE_EVERY = 20  # seconds; counters are saved to Firestore at most this often (free-quota friendly)

    def __init__(self):
        self._hits = defaultdict(deque)
        self._blocked_until: dict[str, float] = {}  # app-wide pauses, e.g. a model the provider rate-limited
        self._lock = threading.Lock()
        self._dirty, self._last_save = False, 0.0

    # ---- persistence (so limits survive restarts and redeploys)
    def export(self) -> dict:
        now = time.time()
        with self._lock:
            hits = [{"k": k, "t": [t for t in q if now - t <= DAY]} for k, q in self._hits.items()]
            blocked = [{"k": k, "until": u} for k, u in self._blocked_until.items() if u > now]
        return {"hits": [h for h in hits if h["t"]], "blocked": blocked, "saved": now}

    def load(self, state: dict):
        now = time.time()
        with self._lock:
            for h in state.get("hits", []):
                self._hits[h["k"]] = deque(sorted(t for t in h.get("t", []) if now - t <= DAY))
            for b in state.get("blocked", []):
                if b.get("until", 0) > now:
                    self._blocked_until[b["k"]] = b["until"]

    def _changed(self):
        """Marks counters as changed and saves them in the background if it's been a while."""
        self._dirty = True
        if time.time() - self._last_save < self.SAVE_EVERY:
            return
        from utils import store  # lazy: store imports this module
        if not store.enabled():
            return
        self._dirty, self._last_save = False, time.time()
        store.save_in_background(store.save_limits, self.export())

    def block(self, name: str, seconds: float):
        with self._lock:
            self._blocked_until[name] = max(self._blocked_until.get(name, 0), time.time() + seconds)
        self._changed()

    def blocked_for(self, name: str) -> int:
        """Seconds left on an app-wide pause (0 if not paused)."""
        with self._lock:
            return max(0, int(self._blocked_until.get(name, 0) - time.time()))

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
        if record:
            self._changed()
        return True, 0

    def take(self, action: str, user_key: str, label: str = ""):
        """
        Checks every limit for `action` ("question" or "upload") and records the event only if all of
        them have room, so a refused request never uses up allowance.
        Returns (False, message) when refused, or (True, receipt): pass the receipt to undo() for a refund.
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
                    return False, message.format(limit=limit, wait=wait_text(wait), model=label or "This model")
                buckets.append(q)
            for q in buckets:
                q.append(now)
        self._changed()
        return True, now

    def undo(self, action: str, user_key: str, receipt: float):
        """
        Refunds exactly the event take() recorded (its receipt), even if other requests were recorded since,
        e.g. for an answer or an OCR read that failed.
        """
        with self._lock:
            for kind, scope, _ in QUOTAS[action]:
                q = self._hits.get(f"{kind}:{user_key if scope == 'user' else 'app'}")
                if q and receipt in q:
                    q.remove(receipt)
        self._changed()

    def usage(self, kind: str, key: str) -> dict:
        """{"used", "limit", "full_reset_in"} for one limit; full_reset_in is seconds until everything is free again."""
        limit, window = LIMITS[kind]
        now = time.time()
        with self._lock:
            recent = [t for t in self._hits.get(f"{kind}:{key}", deque()) if now - t <= window]
        return {"used": len(recent), "limit": limit,
                "full_reset_in": int(window - (now - recent[-1])) + 1 if recent else 0,
                # when the oldest event expires, i.e. when a full limit gets room again
                "next_free_in": int(window - (now - recent[0])) + 1 if recent else 0}

    def events(self, kind: str, key: str) -> list[float]:
        """Timestamps of this key's events still inside the limit's window (for the usage chart)."""
        limit, window = LIMITS[kind]
        now = time.time()
        with self._lock:
            return [t for t in self._hits.get(f"{kind}:{key}", deque()) if now - t <= window]

    def record(self, kind: str, key: str):
        with self._lock:
            self._hits[f"{kind}:{key}"].append(time.time())
        self._changed()

    def reset(self, kind: str, key: str):
        with self._lock:
            self._hits.pop(f"{kind}:{key}", None)
        self._changed()


def _code_version(cls) -> str:
    """Changes whenever RateLimiter's methods change, so a running app picks up the new code."""
    import hashlib, inspect
    return hashlib.sha1(inspect.getsource(cls).encode()).hexdigest()[:12]


@st.cache_resource
def _shared_limiter(version: str) -> RateLimiter:
    limiter_ = RateLimiter()
    from utils import store  # lazy: store imports this module
    state = store.load_limits()  # counters from before the last restart (None if no database)
    if state:
        limiter_.load(state)
    return limiter_


def limiter() -> RateLimiter:
    """One limiter shared by all sessions; rebuilt automatically if its code is updated."""
    return _shared_limiter(_LIMITER_VERSION)


_LIMITER_VERSION = _code_version(RateLimiter)


TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "1") or 0)  # proxies you control in front of the app


def client_ip() -> str:
    """
    Per-visitor key for rate limits. Order: the IP Streamlit reports; else the X-Forwarded-For entry added
    by our own proxy (counted from the RIGHT, since anything to its left can be typed by the visitor);
    else a random id kept in this browser session. Never a value shared by everyone.
    """
    try:
        ip = st.context.ip_address
        if ip:
            return str(ip)
    except Exception:
        pass
    if TRUSTED_PROXY_HOPS > 0:
        try:
            hops = [h.strip() for h in st.context.headers.get("X-Forwarded-For", "").split(",") if h.strip()]
            if len(hops) >= TRUSTED_PROXY_HOPS:
                return hops[-TRUSTED_PROXY_HOPS]
        except Exception:
            pass
    if "_visitor_id" not in st.session_state:
        st.session_state["_visitor_id"] = "session:" + uuid.uuid4().hex
    return st.session_state["_visitor_id"]


def duration_text(seconds: int) -> str:
    """Compact form for cards and menus: '4h 12m', '9m', '45s'."""
    h, m = divmod(max(0, seconds) // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m" if m else f"{max(1, seconds)}s"


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
_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*((?:[^()\s]|\([^()]*\))*)[^)]*\)")  # [text](url "title")
_AUTOLINK = re.compile(r"<\s*[a-z][a-z0-9+.-]*:[^>]*>", re.IGNORECASE)  # <javascript:...>, <https://...>


def _link_as_text(m) -> str:
    text, url = m.group(1).strip(), m.group(2)
    return f"{text} ({url})" if url.lower().startswith(("http://", "https://")) else text


def sanitize_markdown(text: str) -> str:
    """
    Model output can be steered by text inside an uploaded PDF (prompt injection). Remove remote
    images (silent data exfiltration via URL) and raw HTML, and turn every link into plain text:
    web links show their address, anything else (javascript:, data:, file:) keeps only its label.
    """
    text = _MD_IMAGE.sub("", text or "")
    text = _AUTOLINK.sub("", text)
    text = _HTML_TAG.sub("", text)
    return _MD_LINK.sub(_link_as_text, text)
