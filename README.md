# PDF Chat RAG Pipeline

A Streamlit application that allows you to chat with a PDF document using Google's Gemini models, LangChain, and FAISS.

## Features
- Upload PDF documents
- Automatic chunking and indexing using FAISS
- Conversational interface to ask questions about the document
- Powered by Google Gemini (`gemini-1.5-pro` for chat, `models/embedding-001` for embeddings)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key:
   - Create a `.env` file in the root directory and add:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```
   - Alternatively, for Streamlit secrets, copy `.streamlit/secrets.toml` and update the key.

4. Run the app:
   ```bash
   streamlit run app.py
   ```
