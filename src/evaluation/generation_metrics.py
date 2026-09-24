"""
Generation evaluation metrics — measures answer quality.

Implements:
- Exact Match (EM): Binary — does the answer exactly match the ground truth?
- F1 Score: Token-level overlap between predicted and ground truth answers.
- ROUGE-L: Longest common subsequence-based similarity.
- LLM-as-Judge: Uses the LLM to evaluate faithfulness and relevance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison: lowercase, strip, remove articles/punct."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(predicted: str, ground_truth: str) -> float:
    """
    Exact match after normalization.

    Returns 1.0 if the normalized answers match, 0.0 otherwise.
    """
    return 1.0 if normalize_answer(predicted) == normalize_answer(ground_truth) else 0.0


def token_f1(predicted: str, ground_truth: str) -> float:
    """
    Token-level F1 score between predicted and ground truth answers.

    Returns the harmonic mean of precision and recall over word tokens.
    """
    pred_tokens = normalize_answer(predicted).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def rouge_l(predicted: str, ground_truth: str) -> float:
    """
    ROUGE-L F1 score using the longest common subsequence.

    This is a pure Python implementation. For large-scale evaluation,
    consider using the `rouge-score` library.
    """
    pred_tokens = normalize_answer(predicted).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == truth_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n

    return 2 * precision * recall / (precision + recall)


def llm_judge_faithfulness(
    llm_client,
    query: str,
    context: str,
    answer: str,
) -> dict[str, Any]:
    """
    Use the LLM as a judge to evaluate answer faithfulness.

    Returns a dict with 'score' (1-5) and 'reasoning'.
    """
    from src.generation.prompts import faithfulness_judge_prompt

    system_prompt, user_prompt = faithfulness_judge_prompt(query, context, answer)
    response = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)

    return _parse_judge_response(response.text)


def llm_judge_relevance(
    llm_client,
    query: str,
    answer: str,
) -> dict[str, Any]:
    """
    Use the LLM as a judge to evaluate answer relevance.

    Returns a dict with 'score' (1-5) and 'reasoning'.
    """
    from src.generation.prompts import relevance_judge_prompt

    system_prompt, user_prompt = relevance_judge_prompt(query, answer)
    response = llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)

    return _parse_judge_response(response.text)


def _parse_judge_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM judge's SCORE + REASONING response."""
    score = 3  # Default middle score
    reasoning = response_text

    # Try to extract SCORE: N
    score_match = re.search(r"SCORE:\s*(\d+)", response_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(5, score))  # Clamp to [1, 5]

    # Try to extract REASONING: ...
    reasoning_match = re.search(
        r"REASONING:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return {"score": score, "reasoning": reasoning}


def evaluate_generation(
    predicted: str,
    ground_truth: str,
    llm_client=None,
    query: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Run all generation metrics and return comprehensive results.

    Args:
        predicted: The model's answer.
        ground_truth: The expected answer.
        llm_client: Optional LLM client for judge-based metrics.
        query: The original query (for LLM judge metrics).
        context: The retrieval context (for faithfulness evaluation).

    Returns:
        Dict with all metric results.
    """
    results: dict[str, Any] = {
        "exact_match": exact_match(predicted, ground_truth),
        "f1": token_f1(predicted, ground_truth),
        "rouge_l": rouge_l(predicted, ground_truth),
    }

    # Optional LLM-as-judge metrics
    if llm_client and query:
        if context:
            faithfulness = llm_judge_faithfulness(llm_client, query, context, predicted)
            results["faithfulness_score"] = faithfulness["score"]
            results["faithfulness_reasoning"] = faithfulness["reasoning"]

        relevance = llm_judge_relevance(llm_client, query, predicted)
        results["relevance_score"] = relevance["score"]
        results["relevance_reasoning"] = relevance["reasoning"]

    return results
