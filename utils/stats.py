"""
App-wide counters for the owner's Admin page (answers, speed, errors) and answer feedback.
In memory, like the rate limiter; feedback is also appended to data/feedback.jsonl
(git-ignored) so thumbs-down questions can be turned into evaluation test cases.
"""
import json
import os
import threading
import time
from collections import defaultdict, deque

import streamlit as st

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback.jsonl")
MAX_RECENT = 200  # recent feedback kept in memory for the Admin page


class AppStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.answers = defaultdict(lambda: {"count": 0, "seconds": 0.0, "errors": 0})  # per model
        self.feedback = deque(maxlen=MAX_RECENT)

    def record_answer(self, model: str, seconds: float, ok: bool):
        with self._lock:
            a = self.answers[model]
            if ok:
                a["count"] += 1
                a["seconds"] += seconds
            else:
                a["errors"] += 1

    def record_feedback(self, entry: dict):
        """entry: {"rating": 1|0, "question", "answer", "model", "sources"}; also appended to FEEDBACK_FILE."""
        entry = {**entry, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self._lock:
            self.feedback.appendleft(entry)
            try:
                os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
                with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass  # read-only hosts: the in-memory copy still feeds the Admin page

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime": time.time() - self.started,
                "answers": {m: dict(v) for m, v in self.answers.items()},
                "feedback": list(self.feedback),
            }


@st.cache_resource
def _shared_stats(version: int) -> AppStats:
    return AppStats()


def stats() -> AppStats:
    return _shared_stats(1)  # bump the number if AppStats' methods change


def is_admin(email: str) -> bool:
    """Owner-only pages: emails listed in ADMIN_EMAILS (comma-separated). Checked on the server."""
    admins = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
    return bool(email) and email.lower() in admins
