import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# llama-3.3-70b-versatile was shut down by Groq on 16 Aug 2026.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def _limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


# Every answer model in one place: id -> (menu name, description, provider, env-name suffix,
# questions per user per day, questions per day for the whole app). Each model has its own free quota,
# so keep the app-wide numbers a little below what the provider allows. Override any of them in .env
# with LIMIT_MODEL_<suffix> / LIMIT_APP_MODEL_<suffix>.
_MODELS = [
    ("openai/gpt-oss-120b", "GPT-OSS 120B", "Most thorough answers", "groq", "GPT_OSS_120B", 50, 500),
    ("openai/gpt-oss-20b", "GPT-OSS 20B", "Fastest replies", "groq", "GPT_OSS_20B", 100, 1000),
    ("qwen/qwen3.6-27b", "Qwen 3.6 27B", "A second opinion", "groq", "QWEN_27B", 50, 500),
    ("gemini-3.6-flash", "Gemini 3.6 Flash", "Google's newest Flash model", "google", "GEMINI_36_FLASH", 30, 200),
    ("gemini-2.5-flash", "Gemini 2.5 Flash", "Google, balanced and reliable", "google", "GEMINI_25_FLASH", 30, 200),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", "Google, lightest and quickest", "google",
     "GEMINI_25_FLASH_LITE", 50, 400),
]
AVAILABLE_MODELS = {m: (name, desc) for m, name, desc, *_ in _MODELS}
PROVIDERS = {m: provider for m, _, _, provider, *_ in _MODELS}
MODEL_DAILY_LIMITS = {
    m: (_limit(f"LIMIT_MODEL_{suffix}", per_user), _limit(f"LIMIT_APP_MODEL_{suffix}", per_app))
    for m, _, _, _, suffix, per_user, per_app in _MODELS
}


def default_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


# Small model used only to rewrite follow-up questions; on Groq each model has its own quota,
# so this keeps the main model's allowance for answers.
REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "2048"))  # caps a runaway answer's cost
MAX_REWRITE_TOKENS = 512  # reasoning models spend some of this thinking, so not too small


@lru_cache(maxsize=16)
def get_llm(model_name: str | None = None, max_tokens: int = MAX_ANSWER_TOKENS):
    """
    Returns a cached chat model for Groq or Google Gemini (one instance per model and token cap).
    max_retries=1: on a rate limit the app pauses that model rather than making the user wait.
    """
    model = model_name or default_model()
    if PROVIDERS.get(model) == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI  # same GOOGLE_API_KEY as embeddings

        return ChatGoogleGenerativeAI(model=model, temperature=0.3, max_output_tokens=max_tokens, max_retries=1)
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or .streamlit/secrets.toml.")
    return ChatGroq(model=model, temperature=0.3, max_tokens=max_tokens, max_retries=1)


_RETRY_IN = re.compile(  # Groq: "Please try again in 1h2m3.5s" or "850ms"
    r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", re.IGNORECASE
)
_RETRY_GOOGLE = re.compile(r"retry in ([\d.]+)s|retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", re.IGNORECASE)


def retry_after_seconds(error: Exception, default: int = 60) -> int:
    """Reads the provider's 'try again in …' hint from a rate-limit error (Groq or Google)."""
    text = str(error)
    google = _RETRY_GOOGLE.search(text)
    if google:
        return max(1, int(float(google.group(1) or google.group(2)) + 0.999))
    m = _RETRY_IN.search(text)
    if not m or not any(m.groups()):
        return default
    h, mins, secs, ms = (float(x) if x else 0.0 for x in m.groups())
    return max(1, int(h * 3600 + mins * 60 + secs + ms / 1000 + 0.999))


def is_rate_limit(error: Exception) -> bool:
    """True when a provider refused the call for quota/rate reasons (HTTP 429 / RESOURCE_EXHAUSTED)."""
    text = f"{type(error).__name__} {error}".lower()
    squashed = text.replace("_", "").replace(" ", "")
    return any(k in squashed for k in ("ratelimit", "resourceexhausted", "quotaexceeded")) or "429" in text


# A custom GROQ_MODEL from .env is offered too, with the default limits
if default_model() not in AVAILABLE_MODELS:
    AVAILABLE_MODELS[default_model()] = (default_model().split("/")[-1], "Set in your .env")
    PROVIDERS[default_model()] = "groq"
    MODEL_DAILY_LIMITS.setdefault(default_model(), (50, 500))
MODEL_IDS = list(AVAILABLE_MODELS)
