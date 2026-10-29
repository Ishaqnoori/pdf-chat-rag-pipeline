"""
FastAPI backend for the PDF Chat RAG pipeline (for a separate frontend, e.g. Vite).
Run:  uvicorn main:app --reload --port 8000
"""
import os
import shutil
import tempfile
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel, Field

from chain import get_conversational_chain
from rag_pipeline import build_vector_store, get_retriever
from utils.citations import extract_citations

load_dotenv()

app = FastAPI(title="PDF Chat RAG API")

# Comma-separated list, e.g. "http://localhost:5173,https://myapp.com"
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory state (single-document, single-process demo) ---
chat_histories: dict[str, InMemoryChatMessageHistory] = {}
state = {"chain": None, "filename": None}
state_lock = threading.Lock()


def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


def _get_chain():
    """Builds the chain once and reuses it; also recovers an index saved by a previous run."""
    with state_lock:
        if state["chain"] is None:
            retriever = get_retriever()
            if retriever is None:
                return None
            state["chain"] = get_conversational_chain(retriever, get_session_history)
        return state["chain"]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "document": state["filename"]}


# Plain `def` (not async): these call blocking network/CPU code, so FastAPI runs them in a threadpool
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    tmp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_pdf_path = tmp_file.name

        vector_store = build_vector_store(tmp_pdf_path, source_name=file.filename)

        with state_lock:
            state["chain"] = get_conversational_chain(get_retriever(vector_store), get_session_history)
            state["filename"] = file.filename
            # Old conversations refer to the previous document
            chat_histories.clear()

        return {"status": "success", "message": f"Successfully processed {file.filename}"}
    except ValueError as e:  # e.g. scanned PDF with no text
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)


@app.post("/chat")
def chat_with_pdf(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    chain = _get_chain()
    if chain is None:
        raise HTTPException(status_code=400, detail="No PDF processed yet. Please upload a PDF first.")

    try:
        response = chain.invoke(
            {"question": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return {"answer": response["answer"], "sources": extract_citations(response["docs"])}


@app.delete("/clear")
def clear_history(session_id: str):
    if session_id in chat_histories:
        chat_histories[session_id].clear()
        return {"status": "success"}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
