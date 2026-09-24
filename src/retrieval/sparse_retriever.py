"""
Sparse retriever — BM25 lexical search over the document chunks.

Wraps the BM25 index into the same retrieve() interface as the dense retriever
so both can be used interchangeably by the hybrid retriever.
"""

from __future__ import annotations

import logging
import time

from src.indexing.bm25_index import BM25Index
from src.models import RetrievalResult

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    BM25-based sparse retrieval.

    Args:
        bm25_index: BM25Index instance.

    Usage:
        retriever = SparseRetriever(bm25_index)
        result = retriever.retrieve("What is BM25?", top_k=10)
    """

    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve chunks using BM25 lexical matching.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            RetrievalResult with scored candidates.
        """
        start = time.time()

        candidates = self.bm25_index.search(query=query, top_k=top_k)

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            query=query,
            method="bm25",
            candidates=candidates,
            latency_ms=elapsed_ms,
            metadata={
                "top_k": top_k,
                "results_returned": len(candidates),
                "bm25_k1": self.bm25_index.k1,
                "bm25_b": self.bm25_index.b,
            },
        )
