# 8. Adaptive RAG

The Adaptive RAG controller is the "brain" of AutoRAG. Instead of applying the same retrieval strategy to every query, it analyzes the query's complexity and routes it to the most appropriate pipeline. This is the core idea from the **Adaptive-RAG paper** (Jeong et al., NAACL 2024).

**Code**: `src/adaptive/router.py`, `src/adaptive/single_hop.py`, `src/adaptive/multi_hop.py`

---

## 8.1 The Core Insight

Not all queries are created equal:

```
"What is 2 + 2?"
    → Trivial. The LLM knows this. No retrieval needed.
    → Strategy: DIRECT

"Who is the CEO of NexGen Tech?"
    → Simple factual lookup. One retrieval step suffices.
    → Strategy: SINGLE_HOP

"Who is the CEO of the company that acquired QuantumFlow Innovations?"
    → Multi-step reasoning. Must first find which company, then who leads it.
    → Strategy: MULTI_HOP
```

Using multi-hop for every query is wasteful:
- 3-5x more latency (multiple retrieval + LLM calls)
- 3-5x more tokens (higher cost)
- Potentially worse accuracy for simple queries (more chances for the LLM to hallucinate)

Using single-hop for every query is insufficient:
- Multi-hop questions get incomplete or wrong answers
- The retriever doesn't know which intermediate entities to search for

### The Efficiency vs. Accuracy Tradeoff

From the Adaptive-RAG paper (Table 3):

| Strategy | Time/Query | Accuracy |
|---|---|---|
| No Retrieval | 0.35 sec | Low for complex queries |
| Single-step | 3.08 sec | Good for simple, poor for complex |
| Multi-step | 27.18 sec | Good for complex, wasteful for simple |
| **Adaptive** | **Variable** | **Good for ALL complexities** |

The adaptive approach spends 0.35s on trivial queries, 3s on moderate ones, and 27s only when truly needed for complex multi-hop queries. This is the optimal allocation of computational resources.

---

## 8.2 The Query Router

### How It Works

The router uses the LLM itself to classify query complexity:

```python
class QueryRouter:
    def route(self, query: str) -> RoutingDecision:
        # Build the classification prompt
        system_prompt, user_prompt = query_classification_prompt(query)

        # Ask the LLM to classify: A, B, or C
        response = self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)

        # Parse the response
        text = response.text.strip().upper()
        if "A" in text:
            return RoutingDecision.DIRECT
        elif "C" in text:
            return RoutingDecision.MULTI_HOP
        else:
            return RoutingDecision.SINGLE_HOP  # Default fallback
```

### Classification Criteria

The LLM is prompted to evaluate these signals:

| Signal | Indicates |
|---|---|
| General knowledge questions ("What is photosynthesis?") | DIRECT |
| Named entity lookup ("Who is the CEO of X?") | SINGLE_HOP |
| Comparative questions ("How does X compare to Y?") | SINGLE_HOP or MULTI_HOP |
| Bridge entities ("Who is the CEO of the company that...") | MULTI_HOP |
| Temporal reasoning ("Before X happened, what was...") | MULTI_HOP |
| Multi-document aggregation ("List all companies that...") | MULTI_HOP |

### Fallback Strategy

The router defaults to `SINGLE_HOP` if the LLM's response is ambiguous. This is a safe default because:
- It doesn't waste compute like MULTI_HOP
- It still retrieves context (unlike DIRECT which might hallucinate)
- Most real-world queries are moderate complexity

### Paper Comparison: Our Approach vs. the Paper's

| Aspect | Adaptive-RAG Paper | Our Implementation |
|---|---|---|
| Classifier | Fine-tuned T5-Large (770M params) | Zero-shot LLM prompt |
| Training | Requires labeled training data | No training needed |
| Speed | ~10ms (small dedicated model) | ~200-500ms (LLM inference) |
| Flexibility | Fixed to 3 classes it was trained on | Can be updated by changing the prompt |
| Accuracy | ~55% classification accuracy | Depends on LLM quality |

Our approach trades a small amount of latency for much greater simplicity and flexibility. In production, if routing latency became a bottleneck, you'd fine-tune a small classifier like the paper does.

---

## 8.3 Single-Hop Pipeline

The workhorse pipeline for standard factual questions.

### Flow

```
Query: "Who is the CEO of NexGen Tech?"
  │
  ▼
┌─────────────────┐
│ Hybrid Retrieval │ → 30 candidates (BM25 + Dense + RRF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Reranker      │ → 5 candidates (Cross-encoder re-scored)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Builder  │ → "Sarah Jenkins is the CEO..."
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation  │ → "The CEO of NexGen Tech is Sarah Jenkins"
└─────────────────┘
```

### Implementation

```python
class SingleHopPipeline:
    def answer(self, query: str) -> RAGResponse:
        # Step 1: Retrieve
        retrieval_result = self.hybrid_retriever.retrieve(query, top_k=30)

        # Step 2: Rerank
        candidates = self.reranker.rerank(query, retrieval_result.candidates, top_k=5)

        # Step 3: Build context
        context_str, used_chunks = self.context_builder.build(query, candidates)

        # Step 4: Generate
        system_prompt, user_prompt = rag_generation_prompt(query, context_str)
        llm_response = self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)

        # Step 5: Package response with metrics
        return RAGResponse(answer=llm_response.text, sources=used_chunks, ...)
```

### Metrics Tracked

Every single-hop call records:
- Retrieval latency (ms)
- Reranking latency (ms)
- Generation latency (ms)
- Total latency (ms)
- Input/output token counts
- Source chunks used
- Full execution trace

**Code**: See `src/adaptive/single_hop.py`

---

## 8.4 Multi-Hop Pipeline

The most complex pipeline, designed for questions requiring reasoning across multiple documents.

### Flow

```
Query: "Who is the CEO of the company that acquired QuantumFlow?"
  │
  ▼
┌──────────────────┐
│ Query Decomposer │ → ["Which company acquired QuantumFlow?",
│ (LLM)            │    "Who is the CEO of that company?"]
└────────┬─────────┘
         │
    ┌────┴─────────────────────────────────────┐
    ▼                                          ▼
┌─ HOP 1 ──────────────┐   ┌─ HOP 2 ──────────────────────────┐
│ Sub-Q: "Which company │   │ Sub-Q: "Who is the CEO of         │
│ acquired QuantumFlow?"│   │ NexGen Tech?"                      │
│                       │   │                                     │
│ Retrieve → Rerank     │   │ Retrieve → Rerank                  │
│ Context: "...acquired │   │ Context: "...CEO Sarah Jenkins..."  │
│ by NexGen Tech..."    │   │                                     │
│                       │   │ Evidence from Hop 1 is available    │
│ Answer: "NexGen Tech" │   │ Answer: "Sarah Jenkins"             │
└───────────┬───────────┘   └────────────────┬────────────────────┘
            │                                │
            └──────────┬─────────────────────┘
                       ▼
              ┌────────────────┐
              │   Synthesizer   │ → Combines all evidence
              │   (LLM)        │ → "Sarah Jenkins is the CEO of
              │                │    NexGen Tech, which acquired
              │                │    QuantumFlow in March 2024."
              └────────────────┘
```

### The Three Phases

#### Phase 1: Query Decomposition

The LLM breaks the complex question into sub-questions:

```python
def _decompose_query(self, query: str) -> list[str]:
    system_prompt, user_prompt = query_decomposition_prompt(query)
    response = self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)

    # Parse "1. ..." format into a list of strings
    sub_questions = parse_numbered_list(response.text)

    # Fallback: if decomposition fails, use original query
    if not sub_questions:
        sub_questions = [query]

    return sub_questions[:self.max_hops]  # Cap at max_hops
```

#### Phase 2: Iterative Retrieval + Reasoning

For each sub-question, we run the full retrieval pipeline and generate an intermediate answer. Crucially, each hop receives the **accumulated evidence** from previous hops:

```python
accumulated_evidence = ""

for hop, sub_q in enumerate(sub_questions):
    # Retrieve for this sub-question
    candidates = hybrid_retriever.retrieve(sub_q, top_k=30)
    candidates = reranker.rerank(sub_q, candidates, top_k=5)

    # Build context (includes previous evidence)
    context = context_builder.build(sub_q, candidates)

    # Generate intermediate answer WITH previous evidence
    intermediate = llm.generate(
        intermediate_answer_prompt(sub_q, context, accumulated_evidence)
    )

    # Accumulate for next hop
    accumulated_evidence += f"\nSub-question: {sub_q}\nAnswer: {intermediate.text}"
```

**Why pass previous evidence?** In Hop 2, the sub-question might be "Who is the CEO of NexGen Tech?" The LLM needs to know from Hop 1 that the relevant company is NexGen Tech (not some other company mentioned in the chunks).

#### Phase 3: Synthesis

After all hops complete, we synthesize the final answer:

```python
# Combine all evidence into a final prompt
system_prompt, user_prompt = multi_hop_synthesis_prompt(
    original_query, evidence_chain
)
final_answer = llm.generate(prompt=user_prompt, system_prompt=system_prompt)
```

### Max Hops

We cap the number of hops at `max_hops` (default: 3) to prevent:
- Runaway loops where the LLM keeps decomposing
- Excessive latency and cost
- Diminishing returns (most multi-hop questions need 2-3 hops)

### Deduplication

Across multiple hops, different sub-questions might retrieve the same chunk. We deduplicate by `chunk_id` before returning the final source list.

**Code**: See `src/adaptive/multi_hop.py`

---

## 8.5 The Full Adaptive Pipeline (API Level)

Here's how the FastAPI `/query` endpoint ties everything together:

```python
@app.post("/query")
async def query(request: QueryRequest):
    # Step 1: Route
    if request.force_strategy:
        decision = RoutingDecision(request.force_strategy)
    else:
        decision = router.route(request.question)

    # Step 2: Execute the chosen pipeline
    if decision == RoutingDecision.DIRECT:
        response = llm_client.generate(direct_prompt(request.question))
    elif decision == RoutingDecision.SINGLE_HOP:
        response = single_hop.answer(request.question)
    elif decision == RoutingDecision.MULTI_HOP:
        response = multi_hop.answer(request.question)

    # Step 3: Log and return
    query_logger.log(response)
    return response
```

The `force_strategy` parameter lets you override the router for testing and ablation experiments.

---

## 8.6 Interview-Ready Talking Points

> **Q: What is Adaptive RAG?**
> A: Instead of applying the same retrieval strategy to every query, Adaptive RAG classifies query complexity and routes to the appropriate pipeline — no retrieval for trivial questions, single-hop for simple lookups, multi-hop for complex reasoning. This optimizes both accuracy and computational cost.

> **Q: How does query decomposition work in multi-hop RAG?**
> A: We prompt the LLM to break a complex question into independent sub-questions. Each sub-question is answered through a full retrieve-rerank-generate cycle, with evidence accumulating across hops. Finally, all evidence is synthesized into a comprehensive answer.

> **Q: Why not always use multi-hop?**
> A: Multi-hop is 5-10x slower and more expensive than single-hop. For simple queries like "What is the capital of France?", the extra complexity adds no value and can actually hurt accuracy — more LLM calls mean more chances for hallucination.

> **Q: What happens if the router makes a wrong classification?**
> A: We default to SINGLE_HOP as a safe fallback. A complex question routed to single-hop will still get a partial answer (just from one retrieval step). A simple question routed to multi-hop wastes compute but still produces a correct answer.

---

**Next**: [Chapter 9 — Evaluation & Metrics →](09_evaluation.md)
