import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# models/embedding-001 was shut down by Google on 30 Oct 2025.
# Override with GEMINI_EMBEDDING_MODEL in .env (e.g. models/gemini-embedding-2).
# NOTE: if you change this, delete faiss_index/ and re-upload — vectors from
# different embedding models are not compatible.
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


@lru_cache(maxsize=1)
def get_embeddings_model():
    """
    Returns a cached Google Generative AI embeddings model.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env or .streamlit/secrets.toml.")

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
