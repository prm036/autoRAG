"""
Experiment runner — orchestrates systematic ablation experiments.

Runs the RAG pipeline across multiple configurations and collects
retrieval, generation, and system metrics for each. Outputs structured
results for comparison tables and analysis.

Configurations tested:
- LLM-only (no retrieval)
- BM25-only
- Dense-only
- Hybrid (BM25 + Dense + RRF)
- Hybrid + Reranking
- Single-hop RAG
- Multi-hop RAG
- Adaptive RAG
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import RAGResponse, RoutingDecision

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs controlled ablation experiments across RAG configurations.

    Args:
        output_dir: Directory for saving experiment results.

    Usage:
        runner = ExperimentRunner(output_dir="experiments/results")
        runner.add_config("hybrid+rerank", pipeline_fn)
        results = runner.run(eval_queries)
        runner.save_results(results)
    """

    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = output_dir
        self.configs: dict[str, Any] = {}

    def add_config(
        self,
        name: str,
        pipeline_fn,
        description: str = "",
    ) -> None:
        """
        Register a RAG configuration for evaluation.

        Args:
            name: Configuration name (e.g., "hybrid+rerank").
            pipeline_fn: Callable(query: str) → RAGResponse.
            description: Human-readable description.
        """
        self.configs[name] = {
            "pipeline_fn": pipeline_fn,
            "description": description,
        }
        logger.info(f"Registered config: {name} — {description}")

    def run(
        self,
        eval_data: list[dict[str, Any]],
        configs_to_run: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run all registered configurations against the evaluation dataset.

        Args:
            eval_data: List of {"query": str, "answer": str,
                       "relevant_chunk_ids": list[str]} dicts.
            configs_to_run: Subset of config names to run. None = all.

        Returns:
            Nested dict: {config_name: {query_idx: {metrics...}}}.
        """
        configs = configs_to_run or list(self.configs.keys())
        all_results: dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_queries": len(eval_data),
                "configs": configs,
            },
            "results": {},
        }

        for config_name in configs:
            if config_name not in self.configs:
                logger.warning(f"Config '{config_name}' not registered, skipping")
                continue

            logger.info(f"\n{'='*60}\nRunning config: {config_name}\n{'='*60}")
            pipeline_fn = self.configs[config_name]["pipeline_fn"]

            config_results = []
            for i, sample in enumerate(eval_data):
                query = sample["query"]
                ground_truth = sample.get("answer", "")
                relevant_ids = set(sample.get("relevant_chunk_ids", []))

                logger.info(
                    f"  [{i+1}/{len(eval_data)}] {config_name}: '{query[:60]}...'"
                )

                try:
                    start = time.time()
                    response: RAGResponse = pipeline_fn(query)
                    elapsed_ms = (time.time() - start) * 1000

                    # Compute metrics
                    result = self._evaluate_response(
                        response=response,
                        ground_truth=ground_truth,
                        relevant_ids=relevant_ids,
                    )
                    result["query"] = query
                    result["elapsed_ms"] = elapsed_ms
                    result["error"] = None

                except Exception as e:
                    logger.error(f"  Error on query '{query[:40]}...': {e}")
                    result = {
                        "query": query,
                        "error": str(e),
                    }

                config_results.append(result)

            all_results["results"][config_name] = config_results

            # Log summary for this config
            self._log_config_summary(config_name, config_results)

        return all_results

    def _evaluate_response(
        self,
        response: RAGResponse,
        ground_truth: str,
        relevant_ids: set[str],
    ) -> dict[str, Any]:
        """Evaluate a single RAG response against ground truth."""
        from src.evaluation.generation_metrics import evaluate_generation
        from src.evaluation.retrieval_metrics import evaluate_retrieval
        from src.evaluation.system_metrics import compute_system_metrics

        result: dict[str, Any] = {}

        # Retrieval metrics
        if response.sources:
            retrieved_ids = [s.chunk.chunk_id for s in response.sources]
            if relevant_ids:
                result["retrieval"] = evaluate_retrieval(
                    retrieved_ids, relevant_ids
                )

        # Generation metrics
        if ground_truth:
            result["generation"] = evaluate_generation(
                predicted=response.answer,
                ground_truth=ground_truth,
            )

        # System metrics
        result["system"] = compute_system_metrics(response)

        # Answer text for inspection
        result["answer"] = response.answer[:500]
        result["routing_decision"] = response.routing_decision.value
        result["retrieval_iterations"] = response.retrieval_iterations

        return result

    def _log_config_summary(
        self, config_name: str, results: list[dict]
    ) -> None:
        """Log aggregate summary for a config's results."""
        valid = [r for r in results if r.get("error") is None]
        errors = len(results) - len(valid)

        if not valid:
            logger.info(f"  {config_name}: all {len(results)} queries failed")
            return

        # Average generation metrics
        gen_metrics = [r["generation"] for r in valid if "generation" in r]
        if gen_metrics:
            avg_f1 = sum(m["f1"] for m in gen_metrics) / len(gen_metrics)
            avg_em = sum(m["exact_match"] for m in gen_metrics) / len(gen_metrics)
            logger.info(f"  {config_name} generation: F1={avg_f1:.3f}, EM={avg_em:.3f}")

        # Average system metrics
        sys_metrics = [r["system"] for r in valid if "system" in r]
        if sys_metrics:
            avg_latency = sum(m["total_latency_ms"] for m in sys_metrics) / len(sys_metrics)
            avg_tokens = sum(m["total_tokens"] for m in sys_metrics) / len(sys_metrics)
            logger.info(
                f"  {config_name} system: avg_latency={avg_latency:.0f}ms, "
                f"avg_tokens={avg_tokens:.0f}"
            )

        if errors:
            logger.warning(f"  {config_name}: {errors} errors out of {len(results)}")

    def save_results(
        self, results: dict[str, Any], filename: str | None = None
    ) -> str:
        """Save experiment results to a JSON file."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{timestamp}.json"

        output_path = str(Path(self.output_dir) / filename)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved experiment results to {output_path}")
        return output_path

    def results_to_table(self, results: dict[str, Any]) -> str:
        """
        Format experiment results as a markdown comparison table.

        Returns a string containing a markdown table comparing all configs.
        """
        configs = list(results["results"].keys())
        if not configs:
            return "No results to display."

        header = "| Configuration | F1 | EM | Avg Latency (ms) | Avg Tokens | Errors |"
        separator = "|---|---|---|---|---|---|"
        rows = [header, separator]

        for config_name in configs:
            config_results = results["results"][config_name]
            valid = [r for r in config_results if r.get("error") is None]
            errors = len(config_results) - len(valid)

            if not valid:
                rows.append(f"| {config_name} | — | — | — | — | {errors} |")
                continue

            gen = [r["generation"] for r in valid if "generation" in r]
            sys = [r["system"] for r in valid if "system" in r]

            avg_f1 = sum(m["f1"] for m in gen) / len(gen) if gen else 0
            avg_em = sum(m["exact_match"] for m in gen) / len(gen) if gen else 0
            avg_lat = sum(m["total_latency_ms"] for m in sys) / len(sys) if sys else 0
            avg_tok = sum(m["total_tokens"] for m in sys) / len(sys) if sys else 0

            rows.append(
                f"| {config_name} | {avg_f1:.3f} | {avg_em:.3f} | "
                f"{avg_lat:.0f} | {avg_tok:.0f} | {errors} |"
            )

        return "\n".join(rows)
