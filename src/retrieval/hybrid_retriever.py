"""
Hybrid retriever — combines sparse (BM25) and dense retrieval using
Reciprocal Rank Fusion (RRF).

RRF is a simple, parameter-light fusion method that merges ranked lists
from multiple retrievers without requiring score normalization:

    RRF_score(d) = Σ  1 / (k + rank_i(d))   for each ranker i

where k is a constant (typically 60) that controls how much weight is
given to top-ranked vs. lower-ranked documents.

This module also preserves per-source retrieval results for analysis.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from src.models import RetrievalResult, ScoredChunk
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and dense retrieval via RRF.

    Args:
        sparse_retriever: SparseRetriever (BM25) instance.
        dense_retriever: DenseRetriever instance.
        rrf_k: RRF constant (default 60). Higher values reduce the impact
               of rank position differences.
        sparse_top_k: Number of candidates to retrieve from BM25.
        dense_top_k: Number of candidates to retrieve from dense search.

    Usage:
        hybrid = HybridRetriever(sparse_retriever, dense_retriever)
        result = hybrid.retrieve("What is RAG?", top_k=20)
    """

    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
        sparse_top_k: int = 30,
        dense_top_k: int = 30,
    ):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k

    def retrieve(self, query: str, top_k: int = 20) -> RetrievalResult:
        """
        Retrieve candidates using both BM25 and dense retrieval, then
        fuse results with Reciprocal Rank Fusion.

        Args:
            query: The search query.
            top_k: Number of final fused results to return.

        Returns:
            RetrievalResult with RRF-fused scored candidates.
        """
        start = time.time()

        # Retrieve from both sources
        sparse_result = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)
        dense_result = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)

        # Apply RRF fusion
        fused = self._reciprocal_rank_fusion(
            sparse_candidates=sparse_result.candidates,
            dense_candidates=dense_result.candidates,
            top_k=top_k,
        )

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            query=query,
            method="hybrid_rrf",
            candidates=fused,
            latency_ms=elapsed_ms,
            metadata={
                "rrf_k": self.rrf_k,
                "sparse_candidates": len(sparse_result.candidates),
                "dense_candidates": len(dense_result.candidates),
                "sparse_latency_ms": sparse_result.latency_ms,
                "dense_latency_ms": dense_result.latency_ms,
                "fused_candidates": len(fused),
            },
        )

    def _reciprocal_rank_fusion(
        self,
        sparse_candidates: list[ScoredChunk],
        dense_candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """
        Merge two ranked lists using RRF.

        RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))

        Documents that appear in only one list get a contribution of 0
        from the other.
        """
        # Map chunk_id → (chunk, rrf_score, sources)
        chunk_scores: dict[str, dict] = {}

        # Process sparse (BM25) results
        for sc in sparse_candidates:
            cid = sc.chunk.chunk_id
            rrf_score = 1.0 / (self.rrf_k + sc.rank)

            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk": sc.chunk,
                    "rrf_score": 0.0,
                    "sources": [],
                    "bm25_rank": sc.rank,
                    "bm25_score": sc.score,
                }

            chunk_scores[cid]["rrf_score"] += rrf_score
            chunk_scores[cid]["sources"].append("bm25")

        # Process dense results
        for sc in dense_candidates:
            cid = sc.chunk.chunk_id
            rrf_score = 1.0 / (self.rrf_k + sc.rank)

            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "chunk": sc.chunk,
                    "rrf_score": 0.0,
                    "sources": [],
                    "dense_rank": sc.rank,
                    "dense_score": sc.score,
                }

            chunk_scores[cid]["rrf_score"] += rrf_score
            chunk_scores[cid]["sources"].append("dense")
            chunk_scores[cid]["dense_rank"] = sc.rank
            chunk_scores[cid]["dense_score"] = sc.score

        # Sort by RRF score (descending) and take top_k
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        # Build ScoredChunk results
        results = []
        for rank, entry in enumerate(sorted_chunks, 1):
            source_str = "+".join(sorted(set(entry["sources"])))
            results.append(
                ScoredChunk(
                    chunk=entry["chunk"],
                    score=entry["rrf_score"],
                    source=f"rrf({source_str})",
                    rank=rank,
                )
            )

        logger.debug(
            f"RRF fusion: {len(sparse_candidates)} sparse + "
            f"{len(dense_candidates)} dense → {len(results)} fused candidates"
        )

        return results

    def retrieve_with_breakdown(
        self, query: str, top_k: int = 20
    ) -> dict[str, RetrievalResult]:
        """
        Retrieve using all methods and return results separately for comparison.

        Returns a dict with keys: "sparse", "dense", "hybrid".
        """
        sparse_result = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)
        dense_result = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)

        fused = self._reciprocal_rank_fusion(
            sparse_result.candidates,
            dense_result.candidates,
            top_k=top_k,
        )

        hybrid_result = RetrievalResult(
            query=query,
            method="hybrid_rrf",
            candidates=fused,
        )

        return {
            "sparse": sparse_result,
            "dense": dense_result,
            "hybrid": hybrid_result,
        }
