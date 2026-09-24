"""
BM25 sparse index — traditional lexical retrieval over document chunks.

Maintains a BM25 index using rank_bm25 for term-frequency-based retrieval.
Complements dense retrieval by handling exact keyword matches, names,
numbers, and technical terminology that embedding models may miss.

The index is serializable (pickle) for persistence between sessions.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

from src.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Index:
    """
    BM25-based sparse retrieval index.

    Args:
        persist_path: File path for saving/loading the index.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Usage:
        index = BM25Index()
        index.add_chunks(chunks)
        results = index.search("what is retrieval augmented generation?", top_k=10)
        index.save()
        index = BM25Index.load("data/bm25_index.pkl")
    """

    def __init__(
        self,
        persist_path: str = "data/bm25_index.pkl",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.persist_path = persist_path
        self.k1 = k1
        self.b = b

        self._bm25 = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Build the BM25 index from a list of chunks.

        Note: rank_bm25 builds the index from scratch on initialization,
        so this replaces any existing index.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            Number of chunks indexed.
        """
        from rank_bm25 import BM25Okapi

        if not chunks:
            logger.warning("No chunks to index")
            return 0

        start = time.time()

        self._chunks = list(chunks)
        self._tokenized_corpus = [_tokenize(chunk.text) for chunk in self._chunks]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"Built BM25 index over {len(self._chunks)} chunks in {elapsed_ms:.0f}ms "
            f"(k1={self.k1}, b={self.b})"
        )

        return len(self._chunks)

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """
        Search the BM25 index for the most relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of ScoredChunk objects sorted by BM25 score (highest first).
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index is empty — call add_chunks() first")
            return []

        start = time.time()

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score (descending)
        import numpy as np

        top_indices = np.argsort(scores)[::-1][:top_k]

        elapsed_ms = (time.time() - start) * 1000

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue  # Skip zero-score results

            chunk = self._chunks[idx]
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(scores[idx]),
                    source="bm25",
                    rank=rank + 1,
                )
            )

        logger.debug(
            f"BM25 search returned {len(results)} results in {elapsed_ms:.0f}ms"
        )

        return results

    def save(self, path: str | None = None) -> None:
        """Save the BM25 index and chunks to disk."""
        save_path = path or self.persist_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index ({len(self._chunks)} chunks) to {save_path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a BM25 index from disk."""
        from rank_bm25 import BM25Okapi

        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls(persist_path=path, k1=data["k1"], b=data["b"])
        index._chunks = data["chunks"]
        index._tokenized_corpus = data["tokenized_corpus"]
        index._bm25 = BM25Okapi(
            index._tokenized_corpus,
            k1=index.k1,
            b=index.b,
        )

        logger.info(f"Loaded BM25 index ({len(index._chunks)} chunks) from {path}")
        return index

    def count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunks)

    def info(self) -> dict:
        """Return metadata about the BM25 index."""
        return {
            "method": "BM25Okapi",
            "count": self.count(),
            "k1": self.k1,
            "b": self.b,
            "persist_path": self.persist_path,
        }
