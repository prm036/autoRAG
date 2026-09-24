"""
Multi-hop RAG pipeline — iterative retrieval with query decomposition.

For complex questions that require connecting multiple pieces of evidence:
1. Decompose the complex query into sub-questions
2. For each sub-question: retrieve → rerank → extract evidence
3. Accumulate evidence across steps
4. Synthesize a final answer from all gathered evidence

The pipeline maintains state across iterations and enforces a maximum
number of hops to prevent runaway retrieval loops.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.generation.context_builder import ContextBuilder
from src.generation.llm import LLMClient
from src.generation.prompts import (
    intermediate_answer_prompt,
    multi_hop_synthesis_prompt,
    query_decomposition_prompt,
)
from src.models import RAGResponse, RoutingDecision, ScoredChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class MultiHopPipeline:
    """
    Multi-hop RAG: decompose → (retrieve → reason)* → synthesize.

    Args:
        hybrid_retriever: HybridRetriever for each retrieval step.
        reranker: Reranker for second-stage scoring (optional).
        context_builder: ContextBuilder for assembling per-step context.
        llm_client: LLMClient for decomposition, intermediate answers, and synthesis.
        max_hops: Maximum number of retrieval iterations.
        retrieval_top_k: Candidates per retrieval step.
        rerank_top_k: Candidates after reranking per step.

    Usage:
        pipeline = MultiHopPipeline(hybrid, reranker, context_builder, llm, max_hops=3)
        response = pipeline.answer("Who founded the company that built the first...")
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker | None,
        context_builder: ContextBuilder,
        llm_client: LLMClient,
        max_hops: int = 3,
        retrieval_top_k: int = 30,
        rerank_top_k: int = 5,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm_client = llm_client
        self.max_hops = max_hops
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def answer(self, query: str) -> RAGResponse:
        """
        Run the full multi-hop RAG pipeline.

        Args:
            query: The user's complex question.

        Returns:
            RAGResponse with answer, evidence chain, sources, and metrics.
        """
        total_start = time.time()
        trace: list[dict[str, Any]] = []
        all_sources: list[ScoredChunk] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_retrieval_ms = 0.0
        total_reranking_ms = 0.0

        # Step 1: Decompose query into sub-questions
        decomp_start = time.time()
        sub_questions = self._decompose_query(query)
        decomp_ms = (time.time() - decomp_start) * 1000

        trace.append({
            "step": "decomposition",
            "sub_questions": sub_questions,
            "latency_ms": decomp_ms,
        })

        logger.info(
            f"Decomposed query into {len(sub_questions)} sub-questions: "
            f"{sub_questions}"
        )

        # Step 2: Iterative retrieval for each sub-question
        evidence_chain: list[dict] = []
        accumulated_evidence = ""

        for hop, sub_q in enumerate(sub_questions[: self.max_hops], 1):
            hop_start = time.time()

            # Retrieve
            retrieval_start = time.time()
            retrieval_result = self.hybrid_retriever.retrieve(
                sub_q, top_k=self.retrieval_top_k
            )
            retrieval_ms = (time.time() - retrieval_start) * 1000
            total_retrieval_ms += retrieval_ms

            candidates = retrieval_result.candidates

            # Rerank
            reranking_ms = 0.0
            if self.reranker and candidates:
                rerank_start = time.time()
                candidates = self.reranker.rerank(
                    sub_q, candidates, top_k=self.rerank_top_k
                )
                reranking_ms = (time.time() - rerank_start) * 1000
                total_reranking_ms += reranking_ms

            # Build context for this sub-question
            context_str, used_chunks = self.context_builder.build(sub_q, candidates)
            all_sources.extend(used_chunks)

            # Generate intermediate answer
            system_prompt, user_prompt = intermediate_answer_prompt(
                sub_question=sub_q,
                context=context_str,
                previous_evidence=accumulated_evidence if accumulated_evidence else None,
            )
            intermediate_response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            total_input_tokens += intermediate_response.input_tokens
            total_output_tokens += intermediate_response.output_tokens

            # Record this hop's evidence
            evidence_entry = {
                "sub_question": sub_q,
                "evidence": context_str[:500],  # Truncate for logging
                "answer": intermediate_response.text,
            }
            evidence_chain.append(evidence_entry)

            # Accumulate evidence for next hop
            accumulated_evidence += (
                f"\n\nSub-question: {sub_q}\n"
                f"Answer: {intermediate_response.text}"
            )

            hop_ms = (time.time() - hop_start) * 1000
            trace.append({
                "step": f"hop_{hop}",
                "sub_question": sub_q,
                "candidates_retrieved": len(retrieval_result.candidates),
                "candidates_reranked": len(candidates),
                "retrieval_ms": retrieval_ms,
                "reranking_ms": reranking_ms,
                "hop_latency_ms": hop_ms,
                "intermediate_answer": intermediate_response.text[:200],
            })

            logger.info(
                f"Hop {hop}/{len(sub_questions)}: '{sub_q[:60]}...' → "
                f"'{intermediate_response.text[:80]}...' ({hop_ms:.0f}ms)"
            )

        # Step 3: Synthesize final answer
        synthesis_start = time.time()
        system_prompt, user_prompt = multi_hop_synthesis_prompt(
            query, evidence_chain
        )
        synthesis_response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        synthesis_ms = (time.time() - synthesis_start) * 1000

        total_input_tokens += synthesis_response.input_tokens
        total_output_tokens += synthesis_response.output_tokens

        trace.append({
            "step": "synthesis",
            "latency_ms": synthesis_ms,
            "input_tokens": synthesis_response.input_tokens,
            "output_tokens": synthesis_response.output_tokens,
        })

        total_ms = (time.time() - total_start) * 1000

        # Deduplicate sources
        seen_ids = set()
        unique_sources = []
        for src in all_sources:
            if src.chunk.chunk_id not in seen_ids:
                seen_ids.add(src.chunk.chunk_id)
                unique_sources.append(src)

        return RAGResponse(
            answer=synthesis_response.text,
            query=query,
            routing_decision=RoutingDecision.MULTI_HOP,
            sources=unique_sources,
            sub_questions=sub_questions,
            retrieval_iterations=len(sub_questions),
            retrieval_latency_ms=total_retrieval_ms,
            reranking_latency_ms=total_reranking_ms,
            generation_latency_ms=synthesis_ms,
            total_latency_ms=total_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            trace=trace,
        )

    def _decompose_query(self, query: str) -> list[str]:
        """
        Use the LLM to decompose a complex query into sub-questions.

        Returns a list of sub-question strings.
        """
        system_prompt, user_prompt = query_decomposition_prompt(query)
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        # Parse numbered sub-questions from the response
        sub_questions = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering (e.g., "1.", "1)", "- ")
            import re
            cleaned = re.sub(r"^[\d]+[.)]\s*", "", line)
            cleaned = re.sub(r"^[-•]\s*", "", cleaned)
            if cleaned:
                sub_questions.append(cleaned)

        # Fallback: if decomposition fails, use the original query
        if not sub_questions:
            logger.warning(
                f"Query decomposition produced no sub-questions. "
                f"Using original query."
            )
            sub_questions = [query]

        return sub_questions[: self.max_hops]
