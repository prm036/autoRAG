"""
Cross-encoder reranker — second-stage relevance scoring.

After first-stage retrieval (BM25/dense/hybrid), the reranker takes the
top-N candidate chunks and scores each (query, passage) pair using a
cross-encoder model. This produces much higher-quality relevance scores
than bi-encoder similarity because the cross-encoder attends to both
the query and passage simultaneously.

Pipeline:
    First-stage retrieval (top 20-50) → Cross-Encoder Reranker → Top 5-10 → LLM
"""

from __future__ import annotations

import logging
import time

from src.models import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranking for retrieved candidates.

    Args:
        model_name: HuggingFace cross-encoder model name.
        device: Compute device ("cpu", "cuda", "mps").
        batch_size: Number of pairs to score at once.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # Lazy-load the model
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                f"Loading cross-encoder: {self.model_name} (device={self.device})"
            )
            start = time.time()
            self._model = CrossEncoder(self.model_name, device=self.device)
            elapsed = time.time() - start
            logger.info(f"Cross-encoder loaded in {elapsed:.1f}s")

        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query: The original query.
            candidates: List of ScoredChunk from first-stage retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            Top-k ScoredChunk objects re-scored and re-ordered by the
            cross-encoder, with source set to "reranker".
        """
        if not candidates:
            return []

        start = time.time()

        # Create (query, passage) pairs for the cross-encoder
        pairs = [(query, sc.chunk.text) for sc in candidates]

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Pair scores with candidates and sort by cross-encoder score
        scored_pairs = list(zip(scores, candidates))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (ce_score, original) in enumerate(scored_pairs[:top_k], 1):
            reranked.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=float(ce_score),
                    source="reranker",
                    rank=rank,
                )
            )

        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"Reranked {len(candidates)} candidates → top {len(reranked)} "
            f"in {elapsed_ms:.0f}ms (model={self.model_name})"
        )

        # Log rank changes for analysis
        if logger.isEnabledFor(logging.DEBUG):
            for r in reranked:
                # Find original rank
                orig_rank = next(
                    (c.rank for c in candidates if c.chunk.chunk_id == r.chunk.chunk_id),
                    "?",
                )
                logger.debug(
                    f"  Rank {orig_rank} → {r.rank} "
                    f"(CE score: {r.score:.4f}): "
                    f"{r.chunk.text[:80]}..."
                )

        return reranked

    def info(self) -> dict:
        """Return metadata about the reranker."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }
