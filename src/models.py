"""
Core data models shared across all RAG components.

These Pydantic models define the data structures that flow through the pipeline,
ensuring type safety and clear interfaces between components.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Document & Chunk Models
# =============================================================================

class DocumentMetadata(BaseModel):
    """Metadata associated with a source document."""

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    source: str = ""          # File path or URL
    format: str = ""          # e.g., "pdf", "txt", "md"
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A raw document before chunking."""

    metadata: DocumentMetadata
    text: str


class Chunk(BaseModel):
    """A chunk of text extracted from a document, with full provenance."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str                  # Parent document ID
    text: str
    position: int = 0           # Chunk index within the document
    start_char: int = 0         # Character offset in original text
    end_char: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Populated after embedding
    embedding: list[float] | None = None


class ScoredChunk(BaseModel):
    """A chunk with a retrieval/reranking score attached."""

    chunk: Chunk
    score: float
    source: str = ""            # e.g., "bm25", "dense", "rrf", "reranker"
    rank: int = 0


# =============================================================================
# Retrieval Models
# =============================================================================

class RetrievalResult(BaseModel):
    """Result of a retrieval operation (any method)."""

    query: str
    method: str                  # "bm25", "dense", "hybrid", "hybrid+rerank"
    candidates: list[ScoredChunk]
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Generation Models
# =============================================================================

class LLMResponse(BaseModel):
    """Response from the LLM, including usage metadata."""

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class RoutingDecision(str, Enum):
    """Query routing decision from the Adaptive RAG controller."""

    DIRECT = "direct"
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class RAGResponse(BaseModel):
    """Complete response from the RAG system, with full observability."""

    answer: str
    query: str
    routing_decision: RoutingDecision
    sources: list[ScoredChunk] = Field(default_factory=list)

    # Multi-hop details
    sub_questions: list[str] = Field(default_factory=list)
    retrieval_iterations: int = 0

    # Metrics
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Execution trace for debugging
    trace: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Configuration Model
# =============================================================================

class Config(BaseModel):
    """Runtime configuration loaded from YAML. Provides typed access."""

    # This is intentionally loose — we load the YAML dict and access
    # sub-configs via helper methods. A stricter version would have
    # nested Pydantic models for each section.
    _raw: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, raw: dict[str, Any]):
        super().__init__()
        object.__setattr__(self, "_raw", raw)

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Access config values. e.g., config.get('chunking', 'chunk_size')."""
        section_data = self._raw.get(section, {})
        if key is None:
            return section_data if section_data else default
        return section_data.get(key, default)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw or {})
