"""
Hybrid retrieval: vector search (meaning) + BM25 keyword search (exact terms such as
invoice numbers, names, codes), merged with Reciprocal Rank Fusion.

Cost per question: one embedding call (same as before) plus a BM25 scan over at most
MAX_CHUNKS chunks, which takes a few milliseconds. The BM25 index is built once per PDF.
"""
import re
from dataclasses import dataclass

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Keeps tokens like "inv-2026/01" or "3.5" together so exact identifiers match
_TOKEN = re.compile(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*")
RRF_K = 60  # standard Reciprocal Rank Fusion constant


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 keyword scoring (k1=1.5, b=0.75), the same maths as the rank_bm25 package."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if len(corpus) else 0.0
        self.term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for doc in corpus:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        n = len(corpus)
        self.idf = {t: float(np.log((n - df + 0.5) / (df + 0.5) + 1.0)) for t, df in doc_freq.items()}

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.term_freqs))
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avg_len or 1.0))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.term_freqs], dtype=float)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


@dataclass
class DocIndex:
    """Everything needed to search one PDF: chunks, their vectors and a keyword index."""
    chunks: list[Document]
    vector_store: object  # FAISS
    bm25: BM25


class HybridRetriever(BaseRetriever):
    index: DocIndex
    k: int = 4
    fetch_k: int = 20
    mode: str = "hybrid"  # "hybrid" or "vector" (the eval script compares both)

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vector_hits = self.index.vector_store.similarity_search(query, k=self.fetch_k)
        if self.mode == "vector":
            return vector_hits[: self.k]

        rankings = [[d.metadata["chunk_id"] for d in vector_hits]]
        tokens = tokenize(query)
        if tokens:
            scores = self.index.bm25.get_scores(tokens)
            top = np.argsort(scores)[::-1][: self.fetch_k]
            rankings.append([int(i) for i in top if scores[i] > 0])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[: self.k]
        return [self.index.chunks[i] for i in best]
