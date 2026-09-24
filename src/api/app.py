"""
FastAPI application — serves the RAG system as a REST API.

Endpoints:
- POST /query  — Submit a question, get an answer with sources and metrics
- POST /ingest — Ingest documents into the knowledge base
- GET  /health — Health check with component status
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException

from src.adaptive.multi_hop import MultiHopPipeline
from src.adaptive.router import QueryRouter
from src.adaptive.single_hop import SingleHopPipeline
from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from src.embeddings.embedder import Embedder
from src.generation.context_builder import ContextBuilder
from src.generation.llm import LLMClient
from src.generation.prompts import direct_generation_prompt
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.cleaner import TextCleaner
from src.ingestion.parser import DocumentParser
from src.models import Config, RAGResponse, RoutingDecision
from src.observability.logger import QueryLogger
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

# =============================================================================
# Application Factory
# =============================================================================

def create_app(config_path: str = "config/default.yaml") -> FastAPI:
    """
    Create and configure the FastAPI application.

    Initializes all RAG pipeline components from the config file.
    """
    app = FastAPI(
        title="Adaptive Hybrid RAG API",
        description=(
            "End-to-end Adaptive RAG system with hybrid sparse/dense retrieval, "
            "cross-encoder reranking, and multi-hop retrieval."
        ),
        version="0.1.0",
    )

    # Load configuration
    config = Config.from_yaml(config_path)

    # Initialize components
    components = _init_components(config)
    app.state.components = components
    app.state.config = config

    # Register routes
    _register_routes(app)

    return app


def _init_components(config: Config) -> dict[str, Any]:
    """Initialize all RAG pipeline components from config."""
    # Ingestion
    parser = DocumentParser(
        supported_formats=config.get("ingestion", "supported_formats")
    )
    cleaner = TextCleaner()
    chunker = Chunker(
        strategy=config.get("chunking", "strategy", "sentence"),
        chunk_size=config.get("chunking", "chunk_size", 512),
        chunk_overlap=config.get("chunking", "chunk_overlap", 50),
        min_chunk_size=config.get("chunking", "min_chunk_size", 100),
    )

    # Embeddings
    embedder = Embedder(
        model_name=config.get("embedding", "model_name"),
        device=config.get("embedding", "device", "cpu"),
        batch_size=config.get("embedding", "batch_size", 64),
        normalize=config.get("embedding", "normalize", True),
    )

    # Indexing
    vector_store = VectorStore(
        persist_directory=config.get("vector_store", "persist_directory"),
        collection_name=config.get("vector_store", "collection_name"),
        similarity_metric=config.get("vector_store", "similarity_metric", "cosine"),
    )
    bm25_index = BM25Index(
        persist_path=config.get("bm25", "persist_path"),
        k1=config.get("bm25", "k1", 1.5),
        b=config.get("bm25", "b", 0.75),
    )

    # Retrieval
    dense_retriever = DenseRetriever(embedder, vector_store)
    sparse_retriever = SparseRetriever(bm25_index)
    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        rrf_k=config.get("retrieval", "rrf_k", 60),
        sparse_top_k=config.get("retrieval", "sparse_top_k", 30),
        dense_top_k=config.get("retrieval", "dense_top_k", 30),
    )

    # Reranking
    reranker = None
    if config.get("reranking", "enabled", True):
        reranker = Reranker(
            model_name=config.get("reranking", "model_name"),
            device=config.get("embedding", "device", "cpu"),
            batch_size=config.get("reranking", "batch_size", 32),
        )

    # Generation
    llm_config = config.get("llm") or {}
    provider = llm_config.get("provider", "openai")
    provider_config = llm_config.get(provider, {})

    llm_client = LLMClient(
        provider=provider,
        model_name=llm_config.get("model_name", "gpt-4o-mini"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 1024),
        **provider_config,
    )

    context_builder = ContextBuilder(
        max_contexts=config.get("context", "max_contexts", 5),
        max_context_tokens=config.get("context", "max_context_tokens", 3000),
        ordering=config.get("context", "ordering", "relevance"),
        dedup_threshold=config.get("context", "dedup_threshold", 0.9),
    )

    # Adaptive RAG pipelines
    rerank_top_k = config.get("reranking", "top_k", 5)
    retrieval_top_k = config.get("retrieval", "hybrid_top_k", 30)

    router = QueryRouter(llm_client)
    single_hop = SingleHopPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm_client=llm_client,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    multi_hop = MultiHopPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm_client=llm_client,
        max_hops=config.get("adaptive", "max_hops", 3),
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )

    # Observability
    query_logger = QueryLogger(
        log_dir=config.get("observability", "log_dir", "logs/"),
        log_format=config.get("observability", "log_format", "json"),
    )

    return {
        "parser": parser,
        "cleaner": cleaner,
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "dense_retriever": dense_retriever,
        "sparse_retriever": sparse_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "llm_client": llm_client,
        "context_builder": context_builder,
        "router": router,
        "single_hop": single_hop,
        "multi_hop": multi_hop,
        "query_logger": query_logger,
    }


# =============================================================================
# Routes
# =============================================================================

def _register_routes(app: FastAPI) -> None:
    """Register API endpoints."""

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Process a question through the Adaptive RAG pipeline."""
        c = app.state.components

        try:
            # Determine strategy
            if request.force_strategy:
                decision = RoutingDecision(request.force_strategy)
            else:
                decision = c["router"].route(request.question)

            # Execute the appropriate pipeline
            if decision == RoutingDecision.DIRECT:
                system_prompt, user_prompt = direct_generation_prompt(request.question)
                llm_response = c["llm_client"].generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                response = RAGResponse(
                    answer=llm_response.text,
                    query=request.question,
                    routing_decision=RoutingDecision.DIRECT,
                    generation_latency_ms=llm_response.latency_ms,
                    total_latency_ms=llm_response.latency_ms,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                )

            elif decision == RoutingDecision.SINGLE_HOP:
                response = c["single_hop"].answer(request.question)

            elif decision == RoutingDecision.MULTI_HOP:
                response = c["multi_hop"].answer(request.question)

            else:
                raise ValueError(f"Unknown routing decision: {decision}")

            # Log the query
            c["query_logger"].log(response)

            # Convert to API response
            return QueryResponse(
                answer=response.answer,
                question=response.query,
                sources=[
                    SourceChunk(
                        chunk_id=s.chunk.chunk_id,
                        doc_id=s.chunk.doc_id,
                        text=s.chunk.text,
                        score=s.score,
                        source=s.source,
                        metadata=s.chunk.metadata,
                    )
                    for s in response.sources
                ],
                metrics=QueryMetrics(
                    routing_decision=response.routing_decision.value,
                    retrieval_iterations=response.retrieval_iterations,
                    total_latency_ms=response.total_latency_ms,
                    retrieval_latency_ms=response.retrieval_latency_ms,
                    reranking_latency_ms=response.reranking_latency_ms,
                    generation_latency_ms=response.generation_latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ),
                sub_questions=response.sub_questions,
                trace=response.trace,
            )

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest):
        """Ingest documents from a directory into the knowledge base."""
        c = app.state.components

        try:
            # Parse documents
            docs = c["parser"].parse_directory(request.directory)

            # Clean
            for doc in docs:
                doc.text = c["cleaner"].clean(doc.text)

            # Chunk
            chunks = c["chunker"].chunk_documents(docs)

            # Embed
            chunks = c["embedder"].embed_chunks(chunks)

            # Index
            num_indexed = c["vector_store"].add_chunks(chunks)
            c["bm25_index"].add_chunks(chunks)
            c["bm25_index"].save()

            return IngestResponse(
                documents_parsed=len(docs),
                chunks_created=len(chunks),
                chunks_indexed=num_indexed,
                message=f"Successfully ingested {len(docs)} documents "
                        f"({len(chunks)} chunks)",
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        c = app.state.components

        return HealthResponse(
            status="healthy",
            version="0.1.0",
            components={
                "vector_store": c["vector_store"].info(),
                "bm25_index": c["bm25_index"].info(),
                "embedder": c["embedder"].info(),
                "llm": c["llm_client"].info(),
                "reranker": c["reranker"].info() if c["reranker"] else None,
            },
        )


# =============================================================================
# Default app instance (for uvicorn)
# =============================================================================

app = create_app()
