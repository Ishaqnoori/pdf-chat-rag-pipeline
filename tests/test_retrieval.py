from langchain_core.documents import Document

from utils.retrieval import BM25, DocIndex, HybridRetriever, tokenize

TEXTS = [
    "The total amount due is 348 rupees",
    "Invoice number INV-2026/01 issued by Amazon",
    "Shipping address in Hyderabad",
    "Marketplace fee of 5 rupees",
    "Gaming mouse with RGB lights",
]
CHUNKS = [Document(page_content=t, metadata={"chunk_id": i}) for i, t in enumerate(TEXTS)]


class MissesExactIds:
    """Fake vector store that ranks the invoice-number chunk last (a typical vector-search miss)."""
    def similarity_search(self, query, k):
        return [CHUNKS[i] for i in (0, 3, 2, 4, 1)][:k]


def make_index():
    return DocIndex(chunks=CHUNKS, vector_store=MissesExactIds(), bm25=BM25([tokenize(t) for t in TEXTS]))


def test_tokenize_keeps_identifiers_together():
    assert tokenize("Invoice INV-2026/01 total 348.00") == ["invoice", "inv-2026/01", "total", "348.00"]


def test_bm25_scores_exact_term_highest():
    scores = BM25([tokenize(t) for t in TEXTS]).get_scores(tokenize("INV-2026/01"))
    assert scores.argmax() == 1 and scores[1] > 0


def test_bm25_unknown_term_scores_zero():
    assert BM25([tokenize(t) for t in TEXTS]).get_scores(["zzz"]).sum() == 0


def test_hybrid_finds_what_vector_search_misses():
    vector = HybridRetriever(index=make_index(), k=2, mode="vector").invoke("invoice INV-2026/01")
    hybrid = HybridRetriever(index=make_index(), k=2).invoke("invoice INV-2026/01")
    assert CHUNKS[1] not in vector
    assert CHUNKS[1] in hybrid
