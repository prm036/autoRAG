"""
Context builder — assembles retrieved chunks into LLM context.

Handles ordering, deduplication, truncation to token budget, and
source reference formatting. The output is a single context string
ready to be inserted into a generation prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models import ScoredChunk

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Assembles retrieved/reranked chunks into a formatted context for the LLM.

    Args:
        max_contexts: Maximum number of chunks to include.
        max_context_tokens: Approximate token budget (chars / 4 as heuristic).
        ordering: "relevance" (highest score first) or "document_order"
                  (sorted by doc_id + position).
        dedup_threshold: Jaccard similarity threshold for deduplication
                         (0.0 = no dedup, 1.0 = exact match only).

    Usage:
        builder = ContextBuilder(max_contexts=5)
        context_str, used_chunks = builder.build(query, scored_chunks)
    """

    def __init__(
        self,
        max_contexts: int = 5,
        max_context_tokens: int = 3000,
        ordering: str = "relevance",
        dedup_threshold: float = 0.9,
    ):
        self.max_contexts = max_contexts
        self.max_context_tokens = max_context_tokens
        self.ordering = ordering
        self.dedup_threshold = dedup_threshold

    def build(
        self,
        query: str,
        candidates: list[ScoredChunk],
    ) -> tuple[str, list[ScoredChunk]]:
        """
        Build a formatted context string from scored chunks.

        Args:
            query: The user query (for potential future use in context adaptation).
            candidates: Ranked list of ScoredChunk objects.

        Returns:
            (context_string, used_chunks) — the formatted context and the
            chunks that were actually included.
        """
        if not candidates:
            return "", []

        # 1. Deduplicate
        deduped = self._deduplicate(candidates)

        # 2. Limit to max_contexts
        selected = deduped[: self.max_contexts]

        # 3. Order
        if self.ordering == "document_order":
            selected = sorted(
                selected,
                key=lambda sc: (sc.chunk.doc_id, sc.chunk.position),
            )

        # 4. Truncate to token budget
        selected = self._truncate_to_budget(selected)

        # 5. Format
        context_str = self._format_context(selected)

        logger.info(
            f"Built context: {len(candidates)} candidates → "
            f"{len(deduped)} after dedup → {len(selected)} selected "
            f"({len(context_str)} chars)"
        )

        return context_str, selected

    def _deduplicate(self, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        Remove near-duplicate chunks using Jaccard similarity on word sets.

        Keeps the first (highest-ranked) version of near-duplicates.
        """
        if self.dedup_threshold <= 0:
            return candidates

        kept: list[ScoredChunk] = []
        kept_word_sets: list[set[str]] = []

        for sc in candidates:
            words = set(sc.chunk.text.lower().split())

            is_dup = False
            for existing_words in kept_word_sets:
                if not words or not existing_words:
                    continue
                intersection = len(words & existing_words)
                union = len(words | existing_words)
                jaccard = intersection / union if union > 0 else 0

                if jaccard >= self.dedup_threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(sc)
                kept_word_sets.append(words)

        if len(kept) < len(candidates):
            logger.debug(
                f"Deduplicated: {len(candidates)} → {len(kept)} chunks "
                f"(threshold={self.dedup_threshold})"
            )

        return kept

    def _truncate_to_budget(
        self, candidates: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """
        Truncate the candidate list to fit within the token budget.

        Uses a rough heuristic: 1 token ≈ 4 characters.
        """
        max_chars = self.max_context_tokens * 4
        selected = []
        total_chars = 0

        for sc in candidates:
            chunk_chars = len(sc.chunk.text)
            if total_chars + chunk_chars > max_chars:
                break
            selected.append(sc)
            total_chars += chunk_chars

        return selected

    def _format_context(self, candidates: list[ScoredChunk]) -> str:
        """
        Format selected chunks into a numbered context string.

        Each chunk is labeled with its source reference for grounding.
        """
        if not candidates:
            return ""

        parts = []
        for i, sc in enumerate(candidates, 1):
            title = sc.chunk.metadata.get("title", "Unknown")
            source = sc.chunk.metadata.get("source", "")
            source_ref = f"[Source: {title}]" if title else ""

            parts.append(
                f"[Context {i}] {source_ref}\n{sc.chunk.text}"
            )

        return "\n\n".join(parts)
