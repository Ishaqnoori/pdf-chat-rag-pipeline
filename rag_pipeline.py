import os
import shutil

from langchain_community.vectorstores import FAISS

from utils.embeddings import get_embeddings_model
from utils.loaders import load_and_split_pdf

INDEX_DIR = "faiss_index"


def build_vector_store(pdf_path: str, source_name: str | None = None):
    """
    Builds a FAISS vector store from a PDF and saves it to disk.
    """
    chunks = load_and_split_pdf(pdf_path, source_name=source_name)
    vector_store = FAISS.from_documents(chunks, get_embeddings_model())

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    vector_store.save_local(INDEX_DIR)

    return vector_store


def load_vector_store():
    """
    Loads the FAISS store from disk, or returns None if there isn't one.
    """
    if not os.path.exists(INDEX_DIR):
        return None
    # Safe here only because we create the index ourselves — never load an index from an untrusted source.
    return FAISS.load_local(INDEX_DIR, get_embeddings_model(), allow_dangerous_deserialization=True)


def get_retriever(vector_store=None, k: int = 4):
    """
    Returns an MMR retriever. Pass an in-memory vector_store to skip reloading from disk.
    """
    if vector_store is None:
        vector_store = load_vector_store()
    if vector_store is None:
        return None
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20})
