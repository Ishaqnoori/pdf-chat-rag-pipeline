import streamlit as st
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage

from rag_pipeline import build_vector_store, get_retriever
from chain import get_conversational_chain

st.set_page_config(page_title="PDF Chat RAG", page_icon="📄")

# --- Security and Config: Fallback to st.secrets ---
if not os.getenv("GOOGLE_API_KEY"):
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass # Will likely fail downstream if no key is provided

st.title("📄 Chat with your PDF")
st.markdown("Powered by Google Gemini 1.5 Pro and LangChain")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False

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
                    st.error(f"Error processing PDF: {str(e)}. Please check your API key.")
                finally:
                    if os.path.exists(tmp_pdf_path):
                        os.remove(tmp_pdf_path)
    else:
        st.info("Please upload a PDF to begin.")
        st.session_state.retriever_initialized = False
        
    st.divider()
    if st.button("Clear chat"):
        st.session_state["messages"] = []
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
        chain = get_conversational_chain(retriever)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Transform messages into Langchain format for the chain
                    chat_history = []
                    for m in st.session_state["messages"][:-1]: # exclude the latest
                        if m["role"] == "user":
                            chat_history.append(HumanMessage(content=m["content"]))
                        else:
                            chat_history.append(AIMessage(content=m["content"]))

                    response = chain.invoke({
                        "question": user_query,
                        "chat_history": chat_history
                    })
                    
                    answer = response["answer"]
                    docs = response["docs"]
                    
                    # Extract unique pages for citations
                    sources = list(set([doc.metadata.get("page", "Unknown") for doc in docs if doc.metadata.get("page") is not None]))
                    
                    st.write(answer)
                    if sources:
                        st.caption(f"Sources: Page {', '.join(map(str, sources))}")
                    
                    # Update history
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
