"""
Embedding model wrapper — generates vector representations of text.

Wraps sentence-transformers models and provides:
- Batch encoding with progress tracking
- Metadata recording (model name, dimensions, latency)
- Configurable device (CPU/CUDA/MPS)
- Normalized embeddings for cosine similarity
"""

from __future__ import annotations

import logging
import time

import numpy as np
from tqdm import tqdm

from src.models import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates dense embeddings for text using sentence-transformers.

    Args:
        model_name: HuggingFace model name (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        device: Compute device ("cpu", "cuda", "mps").
        batch_size: Number of texts to encode at once.
        normalize: Whether to L2-normalize embeddings (required for cosine similarity).

    Usage:
        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = embedder.embed(["Hello world", "How are you?"])
        chunks = embedder.embed_chunks(chunks)  # Attaches embeddings in-place
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        # Lazy-load the model to avoid import overhead when not needed
        self._model = None
        self._dimensions: int | None = None

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name} (device={self.device})")
            start = time.time()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            elapsed = time.time() - start
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded in {elapsed:.1f}s — "
                f"dimensions={self._dimensions}, device={self.device}"
            )
        return self._model

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        if self._dimensions is None:
            _ = self.model  # trigger lazy load
        return self._dimensions

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of text strings into dense vectors.

        Args:
            texts: List of strings to embed.
            show_progress: Show a tqdm progress bar.

        Returns:
            np.ndarray of shape (len(texts), dimensions).
        """
        if not texts:
            return np.array([])

        start = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"Embedded {len(texts)} texts in {elapsed_ms:.0f}ms "
            f"({elapsed_ms / len(texts):.1f}ms/text) — "
            f"model={self.model_name}, dims={self.dimensions}"
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a 1D vector."""
        return self.embed([query])[0]

    def embed_chunks(
        self, chunks: list[Chunk], show_progress: bool = True
    ) -> list[Chunk]:
        """
        Embed all chunks and attach the embedding to each Chunk object.

        Modifies chunks in-place (sets chunk.embedding) and returns them.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed(texts, show_progress=show_progress)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        logger.info(f"Attached embeddings to {len(chunks)} chunks")
        return chunks

    def info(self) -> dict:
        """Return metadata about the embedding model."""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    embedder = Embedder()

    sample_texts = [
        "What is retrieval-augmented generation?",
        "BM25 is a sparse retrieval algorithm based on term frequency.",
        "Dense retrieval uses neural embeddings for semantic search.",
        "Cross-encoder reranking improves passage relevance scoring.",
    ]

    print(f"\nModel: {embedder.model_name}")
    print(f"Dimensions: {embedder.dimensions}")

    vectors = embedder.embed(sample_texts)
    print(f"Shape: {vectors.shape}")

    # Show pairwise similarities
    print("\nPairwise cosine similarities:")
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            sim = np.dot(vectors[i], vectors[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}")
            print(f"    '{sample_texts[i][:60]}...'")
            print(f"    '{sample_texts[j][:60]}...'")
