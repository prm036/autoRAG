"""
Prompt templates — structured prompts for all generation modes.

Separates prompt engineering from pipeline logic so prompts can be
iterated independently. Each function returns a (system_prompt, user_prompt)
tuple ready for the LLM client.
"""

from __future__ import annotations


# =============================================================================
# RAG Generation Prompts
# =============================================================================

def direct_generation_prompt(query: str) -> tuple[str, str]:
    """Prompt for direct LLM generation (no retrieval context)."""
    system = (
        "You are a knowledgeable assistant. Answer the user's question directly "
        "and accurately. If you are not confident in your answer, say so."
    )
    user = query
    return system, user


def rag_generation_prompt(query: str, context: str) -> tuple[str, str]:
    """
    Prompt for single-step RAG generation with retrieved context.

    Instructs the model to ground its answer in the provided evidence
    and avoid inventing unsupported information.
    """
    system = (
        "You are a precise question-answering assistant. Answer the user's question "
        "using ONLY the provided context. Follow these rules:\n"
        "1. Base your answer strictly on the information in the context.\n"
        "2. If the context does not contain enough information to answer, "
        "say 'I cannot find sufficient information in the provided documents.'\n"
        "3. Do not make up or infer information not present in the context.\n"
        "4. When possible, cite which context(s) support your answer.\n"
        "5. Be concise and direct."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return system, user


def multi_hop_synthesis_prompt(
    query: str,
    evidence_chain: list[dict],
) -> tuple[str, str]:
    """
    Prompt for synthesizing a final answer from multi-hop evidence.

    Args:
        query: The original complex question.
        evidence_chain: List of {"sub_question": ..., "evidence": ..., "answer": ...}
                        dicts from intermediate retrieval steps.
    """
    system = (
        "You are a precise question-answering assistant. You have been given a complex "
        "question that was broken into sub-questions. Evidence and intermediate answers "
        "were gathered for each sub-question. Synthesize a final answer using ALL the "
        "gathered evidence.\n\n"
        "Rules:\n"
        "1. Your answer must be grounded in the provided evidence.\n"
        "2. Connect the evidence from different steps to form a coherent answer.\n"
        "3. If evidence is contradictory or insufficient, acknowledge this.\n"
        "4. Be concise and direct."
    )

    # Format the evidence chain
    evidence_parts = []
    for i, step in enumerate(evidence_chain, 1):
        evidence_parts.append(
            f"Step {i}:\n"
            f"  Sub-question: {step.get('sub_question', 'N/A')}\n"
            f"  Evidence: {step.get('evidence', 'N/A')}\n"
            f"  Intermediate answer: {step.get('answer', 'N/A')}"
        )

    evidence_str = "\n\n".join(evidence_parts)
    user = (
        f"Original Question: {query}\n\n"
        f"Evidence Chain:\n{evidence_str}\n\n"
        f"Based on all the evidence above, provide a final comprehensive answer "
        f"to the original question.\n\nFinal Answer:"
    )
    return system, user


# =============================================================================
# Adaptive RAG Prompts
# =============================================================================

def query_classification_prompt(query: str) -> tuple[str, str]:
    """
    Prompt for classifying query complexity for adaptive routing.

    The LLM decides whether the query needs:
    - DIRECT: can be answered from general knowledge
    - SINGLE_HOP: needs one retrieval step from the knowledge base
    - MULTI_HOP: needs multiple retrieval steps / query decomposition
    """
    system = (
        "You are a query complexity classifier for a RAG system. "
        "Classify the given query into exactly one of these categories:\n\n"
        "DIRECT — The query is about general knowledge, common facts, or "
        "definitions that a language model can answer confidently without "
        "needing to search a knowledge base.\n\n"
        "SINGLE_HOP — The query requires retrieving specific information from "
        "the knowledge base, but a single retrieval step is sufficient. "
        "Examples: factual questions about a specific topic, looking up a "
        "specific detail.\n\n"
        "MULTI_HOP — The query is complex and requires connecting multiple "
        "pieces of information. It may involve comparison, reasoning across "
        "documents, or answering a question that depends on first finding "
        "intermediate information. Examples: 'What is the connection between X "
        "and Y?', 'Compare A and B', 'Who founded the company that...'\n\n"
        "Respond with ONLY the category name (DIRECT, SINGLE_HOP, or MULTI_HOP) "
        "on the first line, followed by a brief one-sentence justification."
    )
    user = f"Query: {query}"
    return system, user


def query_decomposition_prompt(query: str) -> tuple[str, str]:
    """
    Prompt for decomposing a complex query into sub-questions.

    Used by the multi-hop pipeline to break complex questions into
    simpler, independently retrievable sub-questions.
    """
    system = (
        "You are a query decomposition assistant. Break down the given complex "
        "question into 2-4 simpler sub-questions that, when answered individually, "
        "provide the information needed to answer the original question.\n\n"
        "Rules:\n"
        "1. Each sub-question should be self-contained and answerable independently.\n"
        "2. Order the sub-questions logically — earlier answers may inform later questions.\n"
        "3. Keep sub-questions specific and focused.\n"
        "4. Return ONLY the sub-questions, one per line, numbered.\n"
        "5. Do not include the original question in the list."
    )
    user = f"Complex question: {query}\n\nSub-questions:"
    return system, user


def intermediate_answer_prompt(
    sub_question: str,
    context: str,
    previous_evidence: str | None = None,
) -> tuple[str, str]:
    """
    Prompt for answering a sub-question using retrieved context,
    optionally considering previously gathered evidence.
    """
    system = (
        "You are answering a sub-question as part of a multi-step reasoning process. "
        "Answer the sub-question using the provided context. Be concise and factual. "
        "If the context doesn't contain the answer, say 'Not found in context.'"
    )

    user_parts = []
    if previous_evidence:
        user_parts.append(f"Previously gathered evidence:\n{previous_evidence}\n")
    user_parts.append(f"Context:\n{context}\n")
    user_parts.append(f"Sub-question: {sub_question}\n\nAnswer:")

    user = "\n".join(user_parts)
    return system, user


# =============================================================================
# Evaluation Prompts
# =============================================================================

def faithfulness_judge_prompt(
    query: str,
    context: str,
    answer: str,
) -> tuple[str, str]:
    """
    Prompt for LLM-as-judge evaluation of answer faithfulness.

    Determines whether the answer is supported by the retrieved context.
    """
    system = (
        "You are an impartial judge evaluating whether an answer is faithful to "
        "the provided context. An answer is faithful if ALL claims in the answer "
        "are directly supported by the context.\n\n"
        "Respond with:\n"
        "SCORE: <1-5> (1=completely unfaithful, 5=completely faithful)\n"
        "REASONING: <brief explanation>"
    )
    user = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        f"Is this answer faithful to the context?"
    )
    return system, user


def relevance_judge_prompt(query: str, answer: str) -> tuple[str, str]:
    """
    Prompt for LLM-as-judge evaluation of answer relevance.

    Determines whether the answer addresses the user's question.
    """
    system = (
        "You are an impartial judge evaluating whether an answer is relevant to "
        "the question asked. A relevant answer directly addresses what was asked.\n\n"
        "Respond with:\n"
        "SCORE: <1-5> (1=completely irrelevant, 5=perfectly relevant)\n"
        "REASONING: <brief explanation>"
    )
    user = (
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Is this answer relevant to the question?"
    )
    return system, user
