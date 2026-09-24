"""
Retrieval evaluation metrics — measures how well the retrieval pipeline
finds relevant documents.

Implements:
- Recall@K: What fraction of relevant documents are in the top K?
- MRR (Mean Reciprocal Rank): How high is the first relevant document?
- nDCG@K: Discounted cumulative gain accounting for graded relevance.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Compute Recall@K — fraction of relevant documents found in top K.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff rank.

    Returns:
        Recall score in [0, 1].
    """
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    hits = top_k & relevant_ids
    return len(hits) / len(relevant_ids)


def mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank — 1/rank of the first relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.

    Returns:
        MRR score in [0, 1]. Returns 0 if no relevant document is found.
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Compute nDCG@K — normalized discounted cumulative gain.

    Supports binary relevance (relevant_ids) or graded relevance
    (relevance_scores dict mapping doc_id → score).

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Cutoff rank.
        relevance_scores: Optional graded relevance mapping.

    Returns:
        nDCG score in [0, 1].
    """
    def _dcg(ids: list[str], limit: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(ids[:limit]):
            if relevance_scores:
                rel = relevance_scores.get(doc_id, 0.0)
            else:
                rel = 1.0 if doc_id in relevant_ids else 0.0
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
        return dcg

    # Actual DCG
    dcg = _dcg(retrieved_ids, k)

    # Ideal DCG (perfect ranking)
    if relevance_scores:
        ideal_ids = sorted(relevant_ids, key=lambda x: relevance_scores.get(x, 0), reverse=True)
    else:
        ideal_ids = list(relevant_ids)
    idcg = _dcg(ideal_ids, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Run all retrieval metrics and return a comprehensive results dict.

    Args:
        retrieved_ids: Ordered list of retrieved IDs.
        relevant_ids: Ground-truth relevant IDs.
        k_values: List of K values for Recall@K and nDCG@K.

    Returns:
        Dict with all metric results.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    results: dict[str, Any] = {
        "mrr": mrr(retrieved_ids, relevant_ids),
        "total_retrieved": len(retrieved_ids),
        "total_relevant": len(relevant_ids),
    }

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return results
