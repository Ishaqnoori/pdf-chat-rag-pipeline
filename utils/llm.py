import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# llama-3.3-70b-versatile was shut down by Groq on 16 Aug 2026.
# Override with GROQ_MODEL in .env if Groq retires this one too.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns a cached Groq chat model for generation.
    max_retries gives basic backoff on 429 rate-limit errors.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or .streamlit/secrets.toml.")

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0.3,
        max_retries=5,
    )
