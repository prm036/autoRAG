"""
Query router — classifies query complexity for adaptive RAG routing.

Uses the LLM to classify each query into one of three strategies:
- DIRECT: Answer from LLM's parametric knowledge (no retrieval)
- SINGLE_HOP: Standard retrieve-once → rerank → generate
- MULTI_HOP: Iterative retrieval with query decomposition

Based on the Adaptive-RAG paper approach of using a classifier to
determine retrieval complexity per query.
"""

from __future__ import annotations

import logging
import time

from src.generation.llm import LLMClient
from src.generation.prompts import query_classification_prompt
from src.models import RoutingDecision

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Routes queries to the appropriate RAG strategy based on complexity.

    Args:
        llm_client: LLMClient instance for classification.

    Usage:
        router = QueryRouter(llm_client)
        decision = router.route("What is the capital of France?")
        # → RoutingDecision.DIRECT
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def route(self, query: str) -> RoutingDecision:
        """
        Classify a query and return the routing decision.

        Args:
            query: The user's question.

        Returns:
            RoutingDecision enum value.
        """
        start = time.time()

        system_prompt, user_prompt = query_classification_prompt(query)
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        decision = self._parse_decision(response.text)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"Query routed to {decision.value} in {elapsed_ms:.0f}ms: "
            f"'{query[:80]}...'"
        )
        logger.debug(f"Router LLM response: {response.text}")

        return decision

    def _parse_decision(self, response_text: str) -> RoutingDecision:
        """
        Parse the LLM's classification response into a RoutingDecision.

        Extracts the category from the first line of the response.
        Falls back to SINGLE_HOP if parsing fails (safe default).
        """
        first_line = response_text.strip().split("\n")[0].strip().upper()

        if "MULTI_HOP" in first_line or "MULTI-HOP" in first_line:
            return RoutingDecision.MULTI_HOP
        elif "DIRECT" in first_line:
            return RoutingDecision.DIRECT
        elif "SINGLE_HOP" in first_line or "SINGLE-HOP" in first_line or "SINGLE" in first_line:
            return RoutingDecision.SINGLE_HOP
        else:
            logger.warning(
                f"Could not parse routing decision from: '{first_line}'. "
                f"Defaulting to SINGLE_HOP."
            )
            return RoutingDecision.SINGLE_HOP
