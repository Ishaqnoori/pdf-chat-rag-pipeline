import io

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Upload limits (item 16): keep processing time and API cost bounded
MAX_PAGES = 300
MAX_CHUNKS = 2000


def validate_pdf(data: bytes) -> None:
    """Rejects anything that isn't a readable, unencrypted PDF within the page limit."""
    if not data.startswith(b"%PDF-"):
        raise ValueError("This file isn't a valid PDF.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        raise ValueError("This PDF is damaged or can't be opened.")
    if reader.is_encrypted:
        raise ValueError("This PDF is password-protected. Remove the password and upload it again.")
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"This PDF has {len(reader.pages)} pages. The limit is {MAX_PAGES}.")


def _ocr_documents(file_path: str, allow_ocr) -> list:
    """Scanned PDF: read its text with OCR (Gemini), if the caller's usage limits allow it."""
    from utils.ocr import MAX_OCR_PAGES, ocr_pdf

    with open(file_path, "rb") as f:
        data = f.read()
    if len(PdfReader(io.BytesIO(data)).pages) > MAX_OCR_PAGES:
        raise ValueError(f"This looks like a scanned PDF. Scans can be up to {MAX_OCR_PAGES} pages.")
    if allow_ocr is not None:
        ok, message = allow_ocr()
        if not ok:
            raise ValueError(message)
    pages = ocr_pdf(data)
    if not pages:
        raise ValueError("No readable text found in this PDF, even with text recognition.")
    # page is 0-indexed like PyPDFLoader's, so citations stay consistent
    return [Document(page_content=t, metadata={"page": n - 1, "ocr": True}) for n, t in pages]


def load_and_split_pdf(file_path: str, source_name: str, chunk_size: int = 800, chunk_overlap: int = 100,
                       allow_ocr=None):
    """
    Loads a PDF from disk and splits it into chunks. Raises ValueError with a user-safe message.
    allow_ocr: optional callable returning (ok, message), checked before spending OCR quota.
    """
    documents = [d for d in PyPDFLoader(file_path).load() if d.page_content and d.page_content.strip()]
    if not documents:
        documents = _ocr_documents(file_path, allow_ocr)

    for d in documents:
        d.metadata["source"] = source_name  # never expose the server's temp path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError("This PDF has too much text to process. Try a shorter document.")
    return chunks
