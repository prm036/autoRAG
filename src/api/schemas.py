"""
Pydantic schemas for the FastAPI endpoints.

Defines request/response models with full typing for the RAG API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., description="The user's question.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration overrides for this query.",
    )
    force_strategy: str | None = Field(
        None,
        description=(
            "Force a specific routing strategy: 'direct', 'single_hop', "
            "or 'multi_hop'. If None, the adaptive router decides."
        ),
    )


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    directory: str = Field(..., description="Path to the directory of documents to ingest.")


# =============================================================================
# Response Models
# =============================================================================

class SourceChunk(BaseModel):
    """A source chunk referenced in the answer."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str  # "bm25", "dense", "rrf", "reranker"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryMetrics(BaseModel):
    """Execution metrics for a query."""

    routing_decision: str
    retrieval_iterations: int
    total_latency_ms: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    generation_latency_ms: float
    input_tokens: int
    output_tokens: int


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    question: str
    sources: list[SourceChunk]
    metrics: QueryMetrics
    sub_questions: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    documents_parsed: int
    chunks_created: int
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    version: str
    components: dict[str, Any]
