import streamlit as st
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from rag_pipeline import build_vector_store, get_retriever
from chain import get_conversational_chain

st.set_page_config(page_title="PDF Chat RAG", page_icon="📄")

# --- Security and Config: Fallback to st.secrets ---
for key in ["GOOGLE_API_KEY", "GROQ_API_KEY"]:
    if not os.getenv(key):
        try:
            os.environ[key] = st.secrets[key]
        except Exception:
            pass # Will likely fail downstream if no key is provided

st.title("📄 Chat with your PDF")
st.markdown("Powered by Groq (Llama 3.3), Gemini (Embeddings), and LangChain")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False
    
# Store for RunnableWithMessageHistory
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = InMemoryChatMessageHistory()

def get_session_history(session_id: str):
    return st.session_state.chat_history_store

# --- Caching the Vector Index Build ---
@st.cache_resource
def process_pdf(file_path):
    """
    Builds the vector store. Uses Streamlit's cache_resource to prevent 
    rebuilding the index on every UI interaction.
    """
    return build_vector_store(file_path)

# --- Sidebar: File Upload & Controls ---
with st.sidebar:
    st.header("Document & Settings")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing document (chunking & embedding)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_pdf_path = tmp_file.name
                
                try:
                    process_pdf(tmp_pdf_path)
                    st.session_state.retriever_initialized = True
                    st.success("PDF processed successfully! You can now start chatting.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}. Please check your API keys.")
                finally:
                    if os.path.exists(tmp_pdf_path):
                        os.remove(tmp_pdf_path)
    else:
        st.info("Please upload a PDF to begin.")
        st.session_state.retriever_initialized = False
        
    st.divider()
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.session_state.chat_history_store.clear()
        st.rerun()

# --- Main Chat Area ---
if st.session_state.retriever_initialized:
    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: Page {', '.join(map(str, msg['sources']))}")

    # Chat Input
    user_query = st.chat_input("Ask a question about your document...")
    
    # Guard against empty or whitespace-only chat input
    if user_query and user_query.strip():
        # Display user question
        with st.chat_message("user"):
            st.write(user_query)
        
        st.session_state["messages"].append({"role": "user", "content": user_query})
        
        # Build the chain
        retriever = get_retriever()
        chain = get_conversational_chain(retriever, get_session_history)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # RunnableWithMessageHistory handles injecting and updating history natively
                    response = chain.invoke(
                        {"question": user_query},
                        config={"configurable": {"session_id": "default"}}
                    )
                    
                    answer = response["answer"]
                    docs = response["docs"]
                    
                    # Extract unique pages for citations
                    sources = list(set([doc.metadata.get("page", "Unknown") for doc in docs if doc.metadata.get("page") is not None]))
                    
                    st.write(answer)
                    if sources:
                        st.caption(f"Sources: Page {', '.join(map(str, sources))}")
                    
                    # Update UI history
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred during generation: {str(e)}")
else:
    if not uploaded_file:
        st.warning("Upload a document from the sidebar to chat with it.")
