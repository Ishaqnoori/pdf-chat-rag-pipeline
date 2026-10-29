from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(file_path: str, chunk_size: int = 800, chunk_overlap: int = 100,
                       source_name: str | None = None):
    """
    Loads a PDF and splits it into chunks.

    Args:
        file_path: Path to the PDF on disk (often a temp file).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks to keep context.
        source_name: Display name to store in metadata instead of the temp path.

    Returns:
        list[Document]: The chunks.

    Raises:
        ValueError: If no extractable text was found (e.g. a scanned PDF).
    """
    documents = PyPDFLoader(file_path).load()

    # Drop pages with no text so a scanned/image-only PDF fails clearly
    documents = [d for d in documents if d.page_content and d.page_content.strip()]
    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. It may be scanned/image-only — "
            "run OCR on it first."
        )

    if source_name:
        for d in documents:
            d.metadata["source"] = source_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
