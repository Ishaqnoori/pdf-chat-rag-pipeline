from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.embeddings import get_embeddings_model
from utils.loaders import load_and_split_pdf
from utils.retrieval import BM25, DocIndex, HybridRetriever, tokenize


def build_index(pdf_path: str, source_name: str, allow_ocr=None) -> DocIndex:
    """
    Builds the in-memory search index for one PDF (vectors + keywords). Nothing is
    written to disk, so a document's text never outlives the session that uploaded it.
    """
    chunks = load_and_split_pdf(pdf_path, source_name=source_name, allow_ocr=allow_ocr)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i  # shared id so both searches can be fused
    vector_store = FAISS.from_documents(chunks, get_embeddings_model())
    bm25 = BM25([tokenize(c.page_content) for c in chunks])
    return DocIndex(chunks=chunks, vector_store=vector_store, bm25=bm25)


def get_retriever(index: DocIndex, k: int = 4, mode: str = "hybrid") -> HybridRetriever:
    return HybridRetriever(index=index, k=k, mode=mode)


def merge_indexes(indexes: list[DocIndex]) -> DocIndex:
    """
    Combines several PDFs into one searchable index for a multi-document chat.
    Reuses the vectors each PDF already has (no embedding calls); only the small
    keyword index is rebuilt, which takes milliseconds.
    """
    if len(indexes) == 1:
        return indexes[0]
    chunks, pairs = [], []
    for idx in indexes:
        vectors = idx.vector_store.index.reconstruct_n(0, idx.vector_store.index.ntotal)
        for chunk, vector in zip(idx.chunks, vectors):
            copy = Document(page_content=chunk.page_content, metadata={**chunk.metadata, "chunk_id": len(chunks)})
            chunks.append(copy)
            pairs.append((copy.page_content, vector.tolist()))
    vector_store = FAISS.from_embeddings(pairs, get_embeddings_model(), metadatas=[c.metadata for c in chunks])
    return DocIndex(chunks=chunks, vector_store=vector_store, bm25=BM25([tokenize(c.page_content) for c in chunks]))
