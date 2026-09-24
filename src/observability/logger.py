"""
Structured query logger — records execution traces for every query.

Logs the complete execution path including routing decision, retrieval
method, candidates, reranking, latency breakdowns, and token usage.
Supports JSON and text output formats.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import RAGResponse

logger = logging.getLogger(__name__)


class QueryLogger:
    """
    Structured per-query execution logger.

    Writes detailed execution information for every query processed
    by the RAG system. Useful for debugging, analysis, and understanding
    why the system produced a particular answer.

    Args:
        log_dir: Directory for log files.
        log_format: "json" for structured logs, "text" for human-readable.

    Usage:
        qlogger = QueryLogger(log_dir="logs/")
        qlogger.log(response)
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        log_format: str = "json",
    ):
        self.log_dir = log_dir
        self.log_format = log_format

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jsonl" if log_format == "json" else "log"
        self.log_file = os.path.join(log_dir, f"queries_{timestamp}.{ext}")

        logger.info(f"Query logger initialized: {self.log_file}")

    def log(self, response: RAGResponse) -> None:
        """
        Log a complete RAG response with execution details.

        Args:
            response: The RAGResponse to log.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": response.query,
            "routing_decision": response.routing_decision.value,
            "retrieval_iterations": response.retrieval_iterations,
            "sources_used": len(response.sources),
            "sub_questions": response.sub_questions,
            "latency": {
                "total_ms": response.total_latency_ms,
                "retrieval_ms": response.retrieval_latency_ms,
                "reranking_ms": response.reranking_latency_ms,
                "generation_ms": response.generation_latency_ms,
            },
            "tokens": {
                "input": response.input_tokens,
                "output": response.output_tokens,
                "total": response.input_tokens + response.output_tokens,
            },
            "answer_preview": response.answer[:200],
            "source_chunks": [
                {
                    "chunk_id": s.chunk.chunk_id,
                    "doc_id": s.chunk.doc_id,
                    "score": s.score,
                    "source": s.source,
                    "text_preview": s.chunk.text[:100],
                }
                for s in response.sources[:5]
            ],
            "trace": response.trace,
        }

        if self.log_format == "json":
            self._write_json(entry)
        else:
            self._write_text(entry)

    def _write_json(self, entry: dict[str, Any]) -> None:
        """Write a JSON Lines entry."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _write_text(self, entry: dict[str, Any]) -> None:
        """Write a human-readable log entry."""
        lines = [
            f"\n{'='*70}",
            f"[{entry['timestamp']}] Query: {entry['query']}",
            f"  Route: {entry['routing_decision']} | "
            f"Iterations: {entry['retrieval_iterations']} | "
            f"Sources: {entry['sources_used']}",
            f"  Latency: total={entry['latency']['total_ms']:.0f}ms, "
            f"retrieval={entry['latency']['retrieval_ms']:.0f}ms, "
            f"reranking={entry['latency']['reranking_ms']:.0f}ms, "
            f"generation={entry['latency']['generation_ms']:.0f}ms",
            f"  Tokens: in={entry['tokens']['input']}, "
            f"out={entry['tokens']['output']}, "
            f"total={entry['tokens']['total']}",
            f"  Answer: {entry['answer_preview']}",
        ]

        with open(self.log_file, "a") as f:
            f.write("\n".join(lines) + "\n")

    def get_log_path(self) -> str:
        """Return the path to the current log file."""
        return self.log_file
