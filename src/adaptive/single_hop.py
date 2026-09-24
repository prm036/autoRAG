"""
Single-hop RAG pipeline — standard retrieve → rerank → generate.

Orchestrates the full single-step RAG flow:
1. Hybrid retrieval (BM25 + Dense + RRF)
2. Cross-encoder reranking
3. Context construction
4. LLM generation

This is the workhorse pipeline for straightforward factual questions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.generation.context_builder import ContextBuilder
from src.generation.llm import LLMClient
from src.generation.prompts import rag_generation_prompt
from src.models import RAGResponse, RoutingDecision, ScoredChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class SingleHopPipeline:
    """
    Single-step RAG: retrieve once → rerank → build context → generate.

    Args:
        hybrid_retriever: HybridRetriever for first-stage retrieval.
        reranker: Reranker for second-stage scoring (optional).
        context_builder: ContextBuilder for assembling LLM context.
        llm_client: LLMClient for answer generation.
        retrieval_top_k: Number of candidates from first-stage retrieval.
        rerank_top_k: Number of candidates after reranking.

    Usage:
        pipeline = SingleHopPipeline(hybrid_retriever, reranker, context_builder, llm)
        response = pipeline.answer("What is BM25?")
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker | None,
        context_builder: ContextBuilder,
        llm_client: LLMClient,
        retrieval_top_k: int = 30,
        rerank_top_k: int = 5,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm_client = llm_client
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def answer(self, query: str) -> RAGResponse:
        """
        Run the full single-hop RAG pipeline.

        Args:
            query: The user's question.

        Returns:
            RAGResponse with answer, sources, and execution metrics.
        """
        total_start = time.time()
        trace: list[dict[str, Any]] = []

        # Step 1: Hybrid retrieval
        retrieval_start = time.time()
        retrieval_result = self.hybrid_retriever.retrieve(
            query, top_k=self.retrieval_top_k
        )
        retrieval_ms = (time.time() - retrieval_start) * 1000

        trace.append({
            "step": "retrieval",
            "method": retrieval_result.method,
            "candidates": len(retrieval_result.candidates),
            "latency_ms": retrieval_ms,
        })

        candidates = retrieval_result.candidates

        # Step 2: Reranking (optional)
        reranking_ms = 0.0
        if self.reranker and candidates:
            rerank_start = time.time()
            candidates = self.reranker.rerank(
                query, candidates, top_k=self.rerank_top_k
            )
            reranking_ms = (time.time() - rerank_start) * 1000

            trace.append({
                "step": "reranking",
                "input_candidates": len(retrieval_result.candidates),
                "output_candidates": len(candidates),
                "latency_ms": reranking_ms,
            })

        # Step 3: Context construction
        context_str, used_chunks = self.context_builder.build(query, candidates)

        trace.append({
            "step": "context_construction",
            "chunks_used": len(used_chunks),
            "context_length": len(context_str),
        })

        # Step 4: LLM generation
        gen_start = time.time()
        system_prompt, user_prompt = rag_generation_prompt(query, context_str)
        llm_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        gen_ms = (time.time() - gen_start) * 1000

        trace.append({
            "step": "generation",
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "latency_ms": gen_ms,
        })

        total_ms = (time.time() - total_start) * 1000

        return RAGResponse(
            answer=llm_response.text,
            query=query,
            routing_decision=RoutingDecision.SINGLE_HOP,
            sources=used_chunks,
            retrieval_iterations=1,
            retrieval_latency_ms=retrieval_ms,
            reranking_latency_ms=reranking_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=total_ms,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            trace=trace,
        )
