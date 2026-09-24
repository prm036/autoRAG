"""
Dense retriever — vector similarity search over the embedding index.

Wraps the embedding model and vector store into a single retrieve() interface
that the hybrid retriever and adaptive controller can call.
"""

from __future__ import annotations

import logging
import time

from src.embeddings.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.models import RetrievalResult, ScoredChunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retrieval using embedding similarity.

    Args:
        embedder: Embedder instance for encoding queries.
        vector_store: VectorStore instance for similarity search.

    Usage:
        retriever = DenseRetriever(embedder, vector_store)
        result = retriever.retrieve("What is RAG?", top_k=10)
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """
        Retrieve the most semantically similar chunks to the query.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            RetrievalResult with scored candidates.
        """
        start = time.time()

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Search the vector store
        candidates = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
        )

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            query=query,
            method="dense",
            candidates=candidates,
            latency_ms=elapsed_ms,
            metadata={
                "embedding_model": self.embedder.model_name,
                "top_k": top_k,
                "results_returned": len(candidates),
            },
        )
