# Ask your PDF

![tests](https://github.com/Ishaqnoori/pdf-chat-rag-pipeline/actions/workflows/tests.yml/badge.svg)

Chat with any PDF. Upload a document, ask questions in plain language, and get answers drawn
only from that document, as text, tables or charts.

**Stack:** Streamlit · LangChain · FAISS · Gemini embeddings · Groq (GPT-OSS / Qwen) · Firebase Auth

```
PDF → validate → chunk → Gemini embeddings + BM25 keyword index (in memory)
    → hybrid retrieval (Reciprocal Rank Fusion) → Groq LLM (streamed) → formatted answer + source pages
```

## Features
- ChatGPT-style interface that follows your browser's light/dark setting
- Attach a PDF with **+** (up to 5 per chat, searched together), pick the model from the top-left menu
- Follow-up questions remember the conversation
- Answers stream in word by word, formatted as text, lists, tables, or bar/line/pie charts
- Source pages under every answer, plus three suggested follow-up questions
- Hybrid search: vector similarity + BM25 keywords, so exact IDs, names and codes are found
- Scanned PDFs work too: pages without a text layer are read with Gemini's document vision (OCR)
- Regenerate an answer, export a chat as Markdown; switch, rename or delete chats in **Recents**
- Built-in evaluation script for retrieval and answer quality: see [eval/README.md](eval/README.md)
- 👍/👎 feedback on answers, an owner-only **Admin** page (`ADMIN_EMAILS`), automated tests on every push
- Accounts with Firebase (sign up, email verification, password reset) or **Continue with Google**
- Security hardening: see [SECURITY.md](SECURITY.md)

## Setup (Windows)

1. Install Python 3.12 or 3.13 (tick "Add python.exe to PATH").
2. Copy `.env.example` to `.env` and fill in the three keys:
   - `GOOGLE_API_KEY`: https://aistudio.google.com → API keys
   - `GROQ_API_KEY`: https://console.groq.com → API Keys
   - `FIREBASE_API_KEY`: Firebase console → Project settings → Your apps → Config → `apiKey`
     (enable **Authentication → Sign-in method → Email/Password** first)
3. Optional, for **Continue with Google**: create an OAuth client in Google Auth Platform (redirect URI `http://localhost:8501/oauth2callback`), then double-click **`setup_google_login.bat`** and paste the Client ID and secret.
4. Double-click **`run.bat`**. It creates the environment, installs packages and opens the app.

Manual start: `venv\Scripts\python -m streamlit run app.py`

## Deploy (Streamlit Community Cloud)
1. Push to GitHub with `update_github.ps1` (it blocks commits that contain secrets).
2. On share.streamlit.io, create an app from this repo with `app.py` as the entry point.
3. In **Advanced settings → Secrets**, paste the three keys (see `.streamlit/secrets.toml.example`).

## Docker

```bash
docker build -t ask-your-pdf .
docker run -p 8501:8501 --env-file .env ask-your-pdf
```
Secrets are passed at run time and are never copied into the image (see `.dockerignore`).
For Google sign-in, mount your secrets file: `-v "%cd%\.streamlit\secrets.toml:/app/.streamlit/secrets.toml:ro"`.

## Tests

```bash
venv\Scripts\python -m pip install pytest
venv\Scripts\python -m pytest -q
```
GitHub Actions runs them, plus a dependency vulnerability scan, on every push.

## Usage limits

The free Groq and Gemini tiers have per-minute and per-day quotas shared by everyone using the app.
To keep them from running out:

| Limit | Default | `.env` setting |
|---|---|---|
| Questions per user | 30 per 10 min, 100 per day | `LIMIT_QUESTIONS_PER_USER_DAY` |
| PDF uploads per user | 10 per hour, 20 per day | `LIMIT_UPLOADS_PER_USER_DAY` |
| Questions, whole app | 20 per minute, 800 per day | `LIMIT_APP_QUESTIONS_PER_MINUTE`, `LIMIT_APP_QUESTIONS_PER_DAY` |
| Uploads, whole app | 150 per day | `LIMIT_APP_UPLOADS_PER_DAY` |
| Questions per model, per user | GPT-OSS 120B 50, GPT-OSS 20B 100, Qwen 50 per day | `LIMIT_MODEL_GPT_OSS_120B`, `LIMIT_MODEL_GPT_OSS_20B`, `LIMIT_MODEL_QWEN_27B` |
| Questions per model, whole app | 500 / 1,000 / 500 per day | `LIMIT_APP_MODEL_GPT_OSS_120B`, `LIMIT_APP_MODEL_GPT_OSS_20B`, `LIMIT_APP_MODEL_QWEN_27B` |
| Scanned PDFs (OCR) | 5 per user, 40 for the app, per day | `LIMIT_SCANS_PER_USER_DAY`, `LIMIT_APP_SCANS_PER_DAY` |
| Answer length | 2,048 tokens | `MAX_ANSWER_TOKENS` |

Token savings: only the last 3 question/answer pairs are sent to the model, follow-up questions are
rewritten by the small `gpt-oss-20b` model (separate quota), and identical PDFs are embedded only once.
Each session (and each New chat) starts on the model with the most capacity left, unless the user picked one. If a user uses up their allowance on any one model, all models are paused for that user until it resets (the model menu is locked), so switching can't bypass limits. Models are never swapped automatically mid-request: when the chosen model is out of quota the question is refused with its reset time. If the provider rate-limits a model, it is paused for everyone until the provider's reset time, and the user's question is refunded.
Counters are kept in memory, so they reset when the app restarts.

## Project structure
```
app.py                  UI, login gate, chat flow
chain.py                Conversational RAG chain and answer-format prompt
rag_pipeline.py         Builds the per-PDF search index
utils/retrieval.py      Hybrid retriever (BM25 + vectors, RRF)
utils/auth.py           Login, sign-up, sessions, rate limits
utils/firebase_auth.py  Firebase Auth REST client
utils/security.py       Rate limiter, validation, output sanitising
utils/loaders.py        PDF validation and chunking
utils/ocr.py            Text recognition for scanned PDFs (Gemini)
utils/render.py         Markdown + chart rendering
utils/llm.py            Groq models
utils/embeddings.py     Gemini embeddings
assets/style.css        Interface styles
eval/run_eval.py        Retrieval / answer evaluation
utils/stats.py          App-wide stats and answer feedback (Admin page)
tests/                  Automated tests (run in GitHub Actions)
```

## Limitations
- Scanned PDFs are limited to 50 pages / 14 MB and 5 per user per day (OCR costs a vision call).
- Conversations aren't saved; they end when you log out or close the tab.
- PDF text is sent to Google (embeddings) and Groq (answers) for processing.
