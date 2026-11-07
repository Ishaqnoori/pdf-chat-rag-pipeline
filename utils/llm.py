import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# llama-3.3-70b-versatile was shut down by Groq on 16 Aug 2026.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Daily question allowances per model: (per user, whole app). Each model has its own Groq quota,
# so set the app-wide numbers a little below the model's requests-per-day in the Groq console.
def _limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


MODEL_DAILY_LIMITS = {
    "openai/gpt-oss-120b": (_limit("LIMIT_MODEL_GPT_OSS_120B", 50), _limit("LIMIT_APP_MODEL_GPT_OSS_120B", 500)),
    "openai/gpt-oss-20b": (_limit("LIMIT_MODEL_GPT_OSS_20B", 100), _limit("LIMIT_APP_MODEL_GPT_OSS_20B", 1000)),
    "qwen/qwen3.6-27b": (_limit("LIMIT_MODEL_QWEN_27B", 50), _limit("LIMIT_APP_MODEL_QWEN_27B", 500)),
}

# Models offered in the UI: id -> (display name, short description)
AVAILABLE_MODELS = {
    "openai/gpt-oss-120b": ("GPT-OSS 120B", "Most thorough answers"),
    "openai/gpt-oss-20b": ("GPT-OSS 20B", "Fastest replies"),
    "qwen/qwen3.6-27b": ("Qwen 3.6 27B", "A second opinion"),
}


def default_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


# Small model used only to rewrite follow-up questions; on Groq each model has its own quota,
# so this keeps the main model's allowance for answers.
REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "2048"))  # caps a runaway answer's cost
MAX_REWRITE_TOKENS = 512  # reasoning models spend some of this thinking, so not too small


@lru_cache(maxsize=8)
def get_llm(model_name: str | None = None, max_tokens: int = MAX_ANSWER_TOKENS):
    """
    Returns a cached Groq chat model (one instance per model and token cap).
    max_retries=1: on a rate limit we'd rather switch to another model quickly than wait.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or .streamlit/secrets.toml.")

    return ChatGroq(model=model_name or default_model(), temperature=0.3, max_tokens=max_tokens, max_retries=1)


_RETRY_IN = re.compile(
    r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", re.IGNORECASE
)


def retry_after_seconds(error: Exception, default: int = 60) -> int:
    """Reads Groq's 'Please try again in 1h2m3.5s' (or '850ms') hint from a rate-limit error."""
    m = _RETRY_IN.search(str(error))
    if not m or not any(m.groups()):
        return default
    h, mins, secs, ms = (float(x) if x else 0.0 for x in m.groups())
    return max(1, int(h * 3600 + mins * 60 + secs + ms / 1000 + 0.999))


def is_rate_limit(error: Exception) -> bool:
    """True when a provider refused the call for quota/rate reasons (HTTP 429)."""
    text = f"{type(error).__name__} {error}".lower()
    return "ratelimit" in text.replace("_", "").replace(" ", "") or "429" in text or "rate limit" in text
