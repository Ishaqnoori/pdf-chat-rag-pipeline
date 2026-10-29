import hashlib
import os
import tempfile

import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory

st.set_page_config(page_title="PDF Chat RAG", page_icon="📄")

# --- Config: fall back to st.secrets if keys aren't in the environment ---
# Must run before the model helpers are first called.
for key in ["GOOGLE_API_KEY", "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_EMBEDDING_MODEL"]:
    if not os.getenv(key):
        try:
            os.environ[key] = st.secrets[key]
        except Exception:
            pass

from chain import get_conversational_chain  # noqa: E402
from rag_pipeline import build_vector_store, get_retriever  # noqa: E402
from utils.citations import extract_citations  # noqa: E402

st.title("📄 Chat with your PDF")
st.markdown("Powered by Groq, Gemini embeddings, and LangChain")

missing = [k for k in ["GOOGLE_API_KEY", "GROQ_API_KEY"] if not os.getenv(k)]
if missing:
    st.error(f"Missing API key(s): {', '.join(missing)}. Add them to `.env` or `.streamlit/secrets.toml`.")
    st.stop()

# --- Session state ---
st.session_state.setdefault("messages", [])
st.session_state.setdefault("chain", None)
st.session_state.setdefault("doc_hash", None)
st.session_state.setdefault("chat_history_store", InMemoryChatMessageHistory())


def get_session_history(session_id: str):
    return st.session_state.chat_history_store


def reset_chat():
    st.session_state["messages"] = []
    st.session_state.chat_history_store.clear()


@st.cache_resource(show_spinner=False)
def process_pdf(_file_bytes: bytes, filename: str, file_hash: str):
    """
    Builds the vector store once per unique file. Cached on the file's SHA-256
    (the leading underscore tells Streamlit not to hash the raw bytes). The old
    version cached on a random temp path, so the cache never hit.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name
    try:
        return build_vector_store(tmp_path, source_name=filename)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- Sidebar ---
with st.sidebar:
    st.header("Document & Settings")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF"):
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            with st.spinner("Processing document (chunking & embedding)..."):
                try:
                    vector_store = process_pdf(file_bytes, uploaded_file.name, file_hash)
                    st.session_state.chain = get_conversational_chain(
                        get_retriever(vector_store), get_session_history
                    )
                    if st.session_state.doc_hash != file_hash:
                        reset_chat()  # history from a different PDF would confuse the model
                    st.session_state.doc_hash = file_hash
                    st.success("PDF processed! You can start chatting.")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
    else:
        st.info("Please upload a PDF to begin.")
        st.session_state.chain = None
        st.session_state.doc_hash = None

    st.divider()
    if st.button("Clear chat"):
        reset_chat()
        st.rerun()

# --- Main chat area ---
if st.session_state.chain is not None:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: Page {', '.join(msg['sources'])}")

    user_query = st.chat_input("Ask a question about your document...")

    if user_query and user_query.strip():
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state["messages"].append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke(
                        {"question": user_query},
                        config={"configurable": {"session_id": "default"}},
                    )
                    answer = response["answer"]
                    sources = extract_citations(response["docs"])

                    st.write(answer)
                    if sources:
                        st.caption(f"Sources: Page {', '.join(sources)}")

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"An error occurred during generation: {e}")
elif uploaded_file is None:
    st.warning("Upload a document from the sidebar to chat with it.")
else:
    st.info("Click **Process PDF** in the sidebar.")
