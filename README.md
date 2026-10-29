# PDF Chat RAG Pipeline

Chat with a PDF using LangChain, FAISS, Gemini embeddings, and Groq for generation.
Two ways to run it: a **Streamlit app** (`app.py`) or a **FastAPI backend** (`main.py`) for a separate frontend.

## Features
- Upload a PDF → chunk → embed → FAISS index
- Conversational follow-ups (questions are rewritten using chat history)
- Page-number citations for every answer
- Generation: Groq `openai/gpt-oss-120b` (configurable via `GROQ_MODEL`)
- Embeddings: Gemini `models/gemini-embedding-001` (configurable via `GEMINI_EMBEDDING_MODEL`)

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` and `GROQ_API_KEY`
(or put them in `.streamlit/secrets.toml` for the Streamlit app).

## Run

**Streamlit UI**
```bash
streamlit run app.py
```

**FastAPI backend**
```bash
uvicorn main:app --reload --port 8000
```
Docs at http://localhost:8000/docs. Endpoints:

| Method | Path | Body / Params | Returns |
|---|---|---|---|
| GET | `/health` | – | status + current document |
| POST | `/upload` | multipart `file` (PDF) | success message |
| POST | `/chat` | `{"session_id": "...", "message": "..."}` | `{"answer": "...", "sources": ["1","3"]}` |
| DELETE | `/clear?session_id=...` | – | status |

Set `ALLOWED_ORIGINS` in `.env` if your frontend isn't on `http://localhost:5173`.

## Notes
- If you change the embedding model, delete `faiss_index/` and re-upload — old vectors won't match.
- Scanned/image-only PDFs have no text layer; run OCR first.
- The backend keeps one document and chat history in memory — fine for a demo, not for multi-user production.
