"""
System-level metrics — measures operational performance.

Tracks latency breakdowns, token usage, and retrieval efficiency
for comparing RAG configurations at the systems level.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models import RAGResponse

logger = logging.getLogger(__name__)


def compute_system_metrics(response: RAGResponse) -> dict[str, Any]:
    """
    Extract system-level metrics from a RAGResponse.

    Returns a dict containing latency breakdowns, token counts,
    and retrieval statistics.
    """
    return {
        "routing_decision": response.routing_decision.value,
        "retrieval_iterations": response.retrieval_iterations,
        "total_latency_ms": response.total_latency_ms,
        "retrieval_latency_ms": response.retrieval_latency_ms,
        "reranking_latency_ms": response.reranking_latency_ms,
        "generation_latency_ms": response.generation_latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.input_tokens + response.output_tokens,
        "sources_used": len(response.sources),
        "sub_questions": len(response.sub_questions),
    }


def aggregate_system_metrics(
    responses: list[RAGResponse],
) -> dict[str, Any]:
    """
    Aggregate system metrics across multiple responses.

    Returns mean, min, max, and total for each metric.
    """
    if not responses:
        return {}

    metrics = [compute_system_metrics(r) for r in responses]

    # Numeric fields to aggregate
    numeric_fields = [
        "total_latency_ms",
        "retrieval_latency_ms",
        "reranking_latency_ms",
        "generation_latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "retrieval_iterations",
        "sources_used",
    ]

    aggregated: dict[str, Any] = {
        "num_queries": len(responses),
    }

    for field in numeric_fields:
        values = [m[field] for m in metrics]
        aggregated[field] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "total": sum(values),
        }

    # Route distribution
    route_counts: dict[str, int] = {}
    for m in metrics:
        route = m["routing_decision"]
        route_counts[route] = route_counts.get(route, 0) + 1
    aggregated["route_distribution"] = route_counts

    return aggregated
