# 9. Evaluation & Metrics

Evaluation is how you prove your RAG system actually works. Without rigorous metrics, you're just hoping the LLM gives good answers. This chapter covers every metric implemented in AutoRAG, the math behind each one, and how to evaluate at scale.

**Code**: `src/evaluation/retrieval_metrics.py`, `src/evaluation/generation_metrics.py`, `src/evaluation/system_metrics.py`, `src/evaluation/experiment_runner.py`

---

## 9.1 The Three Layers of Evaluation

A RAG system has three layers that can each fail independently:

```
Layer 1: Did the RETRIEVER find the right documents?  → Retrieval Metrics
Layer 2: Did the LLM produce a correct ANSWER?        → Generation Metrics
Layer 3: Was the system fast and cost-effective?       → System Metrics
```

You need all three. A system with perfect retrieval but a hallucinating LLM is still broken. A system with perfect answers but 30-second latency is unusable in production.

---

## 9.2 Retrieval Metrics

These metrics measure whether the retrieval pipeline found the relevant documents. They require a ground truth set of "relevant" chunk/document IDs for each query.

### Recall@K

**Question**: Of all the relevant documents, what fraction did we find in our top K results?

```
                    |Retrieved@K ∩ Relevant|
Recall@K = ─────────────────────────────────
                      |Relevant|
```

**Example**:
- Relevant documents: {A, B, C, D}
- Retrieved top 5: [E, A, F, C, G]
- Retrieved ∩ Relevant = {A, C}
- Recall@5 = 2/4 = **0.50**

**Interpretation**: We found 50% of the relevant documents in our top 5. This is the most important retrieval metric because if relevant documents aren't retrieved, the LLM can't possibly use them.

**When to use**: Always. This is the primary retrieval metric.

### Mean Reciprocal Rank (MRR)

**Question**: How high up in our ranking is the *first* relevant document?

```
              1
MRR = ────────────────
        rank of first
        relevant result
```

**Example**:
- Retrieved: [E, A, F, C, G]
- Relevant: {A, B, C, D}
- First relevant = A at rank 2
- MRR = 1/2 = **0.50**

If the first relevant result is at rank 1: MRR = 1.0 (perfect)
If the first relevant result is at rank 10: MRR = 0.1 (poor)
If no relevant result is found: MRR = 0.0

**Interpretation**: For factoid QA where you only need one good document, MRR tells you how quickly you find it.

**When to use**: Single-hop QA where you need at least one relevant document fast.

### nDCG@K (Normalized Discounted Cumulative Gain)

**Question**: How good is our ranking overall, considering that top positions matter more?

This is the most sophisticated retrieval metric. It accounts for:
1. Whether relevant documents were retrieved (like Recall)
2. Where in the ranking they appear (top positions count more)
3. Graded relevance (some documents are more relevant than others)

#### Step 1: Discounted Cumulative Gain (DCG)

```
            K     relᵢ
DCG@K = Σ  ──────────── 
           i=1  log₂(i+1)
```

Where `relᵢ` is the relevance score of the document at position `i`. The `log₂(i+1)` term is the "discount" — it makes lower positions contribute less.

For binary relevance (relevant = 1, not relevant = 0):

| Position | Document | Relevant? | Discount | Contribution |
|---|---|---|---|---|
| 1 | E | No (0) | log₂(2) = 1.0 | 0/1.0 = 0.000 |
| 2 | A | Yes (1) | log₂(3) = 1.585 | 1/1.585 = 0.631 |
| 3 | F | No (0) | log₂(4) = 2.0 | 0/2.0 = 0.000 |
| 4 | C | Yes (1) | log₂(5) = 2.322 | 1/2.322 = 0.431 |
| 5 | G | No (0) | log₂(6) = 2.585 | 0/2.585 = 0.000 |

DCG@5 = 0.631 + 0.431 = **1.062**

#### Step 2: Ideal DCG (IDCG)

The best possible ranking (all relevant documents at the top):

| Position | Document | Relevant? | Discount | Contribution |
|---|---|---|---|---|
| 1 | A | Yes (1) | 1.0 | 1.000 |
| 2 | B | Yes (1) | 1.585 | 0.631 |
| 3 | C | Yes (1) | 2.0 | 0.500 |
| 4 | D | Yes (1) | 2.322 | 0.431 |
| 5 | — | — | — | — |

IDCG@5 = 1.000 + 0.631 + 0.500 + 0.431 = **2.562**

#### Step 3: Normalize

```
                DCG@K       1.062
nDCG@K = ────────────── = ─────── = 0.415
               IDCG@K      2.562
```

**Interpretation**: Our ranking achieves 41.5% of the ideal ranking quality.

**When to use**: When you want a single metric that captures both recall and ranking quality. Standard for academic benchmarks.

### Our Implementation

```python
def evaluate_retrieval(retrieved_ids, relevant_ids, k_values=[1, 3, 5, 10, 20]):
    return {
        "mrr": mrr(retrieved_ids, relevant_ids),
        "recall@1": recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg@5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
        # ... more k values
    }
```

**Code**: See `src/evaluation/retrieval_metrics.py`

---

## 9.3 Generation Metrics

These metrics compare the LLM's generated answer to the ground truth answer.

### Exact Match (EM)

**Question**: Does the generated answer exactly match the ground truth?

```
EM = 1 if normalize(predicted) == normalize(ground_truth) else 0
```

Before comparison, both strings are normalized:
- Lowercased
- Articles removed ("a", "an", "the")
- Punctuation removed
- Whitespace collapsed

**Example**:
- Predicted: "The CEO is Sarah Jenkins."
- Ground truth: "Sarah Jenkins"
- After normalization: "ceo is sarah jenkins" vs "sarah jenkins"
- EM = **0** (not exact match)

**Limitation**: Very strict. "Sarah Jenkins" vs "Jenkins, Sarah" scores 0 even though both are correct.

**When to use**: Quick binary check. The Adaptive-RAG paper uses this as a primary metric.

### Token-Level F1 Score

**Question**: How much do the predicted and ground truth answers overlap at the word level?

```
                    |predicted_tokens ∩ truth_tokens|
Precision = ───────────────────────────────────────────
                       |predicted_tokens|

                    |predicted_tokens ∩ truth_tokens|
Recall    = ───────────────────────────────────────────
                         |truth_tokens|

              2 × Precision × Recall
F1    = ──────────────────────────────
              Precision + Recall
```

**Example**:
- Predicted: "The CEO is Sarah Jenkins" → tokens: {the, ceo, is, sarah, jenkins}
- Ground truth: "Sarah Jenkins" → tokens: {sarah, jenkins}
- Common tokens: {sarah, jenkins}
- Precision = 2/5 = 0.4
- Recall = 2/2 = 1.0
- F1 = 2 × 0.4 × 1.0 / (0.4 + 1.0) = **0.571**

**Interpretation**: The answer contains the correct information (recall = 1.0) but also has extra words (precision = 0.4). F1 balances both.

**When to use**: The primary metric in the Adaptive-RAG paper. More forgiving than EM while still being token-level precise.

### ROUGE-L (Longest Common Subsequence)

**Question**: What's the longest subsequence of words that appears in both the predicted and ground truth answers?

A **subsequence** is a sequence that appears in the same order but not necessarily contiguously. For example, in "The big red cat", the subsequence "big cat" is valid (words appear in order, with "red" skipped).

```
                LCS(predicted, truth)
Precision = ─────────────────────────────
                   |predicted|

                LCS(predicted, truth)
Recall    = ─────────────────────────────
                     |truth|

              2 × Precision × Recall
ROUGE-L = ──────────────────────────────
              Precision + Recall
```

**LCS Computation** (Dynamic Programming):

Build a matrix where `dp[i][j]` = length of the longest common subsequence of `predicted[:i]` and `truth[:j]`:

```
if predicted[i] == truth[j]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

**Example**:
- Predicted: "the capital is paris" → [the, capital, is, paris]
- Ground truth: "paris is the capital of france" → [paris, is, the, capital, of, france]
- LCS: [is, capital] (length 2) or [the, capital] (length 2)
- Precision = 2/4 = 0.5
- Recall = 2/6 = 0.333
- ROUGE-L = 2 × 0.5 × 0.333 / (0.5 + 0.333) = **0.400**

**When to use**: Captures word order similarity better than F1 (which treats tokens as a bag). Good for longer, sentence-level answers.

### Accuracy (Substring Match)

**Question**: Does the predicted answer *contain* the ground truth as a substring?

```
Accuracy = 1 if ground_truth in predicted else 0
```

**Example**:
- Predicted: "The CEO of NexGen Tech is Sarah Jenkins, who took over in 2021."
- Ground truth: "Sarah Jenkins"
- "Sarah Jenkins" ∈ predicted → Accuracy = **1**

**When to use**: The Adaptive-RAG paper uses this alongside EM and F1. It's more lenient than EM — as long as the correct answer appears somewhere in the response, it counts.

---

## 9.4 LLM-as-a-Judge

### The Problem with Lexical Metrics

All the metrics above compare strings. They fail in cases like:

| Predicted | Ground Truth | EM | F1 | Correct? |
|---|---|---|---|---|
| "NYC" | "New York City" | 0 | 0 | ✅ Yes |
| "Dr. Thorne" | "Aris Thorne" | 0 | 0.5 | ✅ Yes |
| "1976-04-01" | "April 1, 1976" | 0 | 0.33 | ✅ Yes |

These are all correct answers that score poorly on lexical metrics because they use different surface forms.

### The Solution: Use an LLM to Judge

We prompt a separate LLM (or the same one) to act as an evaluator:

#### Faithfulness Judge

**Question**: Is the generated answer supported by the retrieved context, or did the model hallucinate?

```
Prompt:
  Given the context and the answer, evaluate whether the answer
  is faithfully grounded in the context.
  Score from 1 (completely hallucinated) to 5 (fully supported).

  Context: "QuantumFlow was founded in 2018 by Dr. Thorne..."
  Answer: "QuantumFlow was founded in 2018 by Dr. Aris Thorne."

  → SCORE: 5 (fully grounded in context)
```

#### Relevance Judge

**Question**: Does the generated answer actually address the user's question?

```
Prompt:
  Evaluate whether the answer is relevant to the question asked.
  Score from 1 (completely irrelevant) to 5 (directly answers the question).

  Question: "Who founded QuantumFlow?"
  Answer: "QuantumFlow was acquired by NexGen Tech in 2024."

  → SCORE: 1 (doesn't answer the question at all)
```

### When to Use LLM-as-Judge

- **At scale**: When you have 500+ questions and can't manually check each one
- **For semantic equivalence**: When lexical metrics would give false negatives
- **For hallucination detection**: Faithfulness scoring catches fabricated facts

### Limitations

- **Cost**: Each evaluation requires an LLM call (adds cost at scale)
- **Judge bias**: The LLM might be lenient or strict depending on the prompt
- **Circular reasoning**: Using the same model to generate and judge can mask errors

**Code**: See `src/evaluation/generation_metrics.py` — `llm_judge_faithfulness()` and `llm_judge_relevance()`

---

## 9.5 System Metrics

These measure operational performance — critical for production but often overlooked in academic papers.

| Metric | What It Measures | Why It Matters |
|---|---|---|
| Total latency (ms) | End-to-end response time | User experience |
| Retrieval latency (ms) | Time spent searching | Bottleneck identification |
| Reranking latency (ms) | Time in cross-encoder | Usually the slowest stage |
| Generation latency (ms) | Time in LLM | Cost and speed tradeoff |
| Input tokens | Tokens sent to LLM | Directly affects cost |
| Output tokens | Tokens generated by LLM | Directly affects cost |
| Retrieval iterations | Number of hops | Efficiency indicator |

### Aggregation

For experiment-level analysis, we compute **mean, min, max, and total** across all queries:

```python
def aggregate_system_metrics(responses):
    return {
        "total_latency_ms": {"mean": 1234, "min": 200, "max": 5000, "total": 61700},
        "input_tokens": {"mean": 1500, "min": 200, "max": 4000, "total": 75000},
        "route_distribution": {"direct": 5, "single_hop": 35, "multi_hop": 10},
    }
```

**Code**: See `src/evaluation/system_metrics.py`

---

## 9.6 The Experiment Runner

The `ExperimentRunner` orchestrates full ablation experiments:

### What It Does

1. Register multiple pipeline configurations (BM25-only, Dense-only, Hybrid, Hybrid+Rerank, Adaptive)
2. Run each configuration against the same evaluation dataset
3. Compute all retrieval, generation, and system metrics
4. Output a comparison table

### Usage

```python
runner = ExperimentRunner(output_dir="experiments/results")

# Register configurations
runner.add_config("bm25_only", bm25_pipeline_fn)
runner.add_config("dense_only", dense_pipeline_fn)
runner.add_config("hybrid", hybrid_pipeline_fn)
runner.add_config("hybrid+rerank", hybrid_rerank_pipeline_fn)
runner.add_config("adaptive", adaptive_pipeline_fn)

# Run experiments
results = runner.run(eval_data)

# Print comparison table
print(runner.results_to_table(results))
```

### Output

```
| Configuration   | F1    | EM    | Avg Latency (ms) | Avg Tokens | Errors |
|-----------------|-------|-------|-------------------|------------|--------|
| bm25_only       | 0.410 | 0.280 | 1200              | 800        | 0      |
| dense_only      | 0.450 | 0.310 | 1400              | 850        | 0      |
| hybrid          | 0.520 | 0.380 | 1600              | 900        | 0      |
| hybrid+rerank   | 0.580 | 0.430 | 1800              | 920        | 0      |
| adaptive        | 0.590 | 0.440 | 1500              | 880        | 0      |
```

This table is what you put on your resume and discuss in interviews. It shows that each component adds measurable value.

**Code**: See `src/evaluation/experiment_runner.py`

---

## 9.7 Metric Summary Reference Card

| Metric | Type | Range | Formula | Best For |
|---|---|---|---|---|
| Recall@K | Retrieval | [0, 1] | \|retrieved∩relevant\| / \|relevant\| | Did we find the right docs? |
| MRR | Retrieval | [0, 1] | 1/rank_of_first_relevant | How quickly do we find one? |
| nDCG@K | Retrieval | [0, 1] | DCG@K / IDCG@K | Overall ranking quality |
| Exact Match | Generation | {0, 1} | normalized(pred) == normalized(truth) | Strict correctness |
| F1 | Generation | [0, 1] | Harmonic mean of token P & R | Token overlap |
| ROUGE-L | Generation | [0, 1] | LCS-based F1 | Word order similarity |
| Accuracy | Generation | {0, 1} | truth ∈ predicted | Lenient correctness |
| Faithfulness | LLM Judge | [1, 5] | LLM evaluation | Hallucination detection |
| Relevance | LLM Judge | [1, 5] | LLM evaluation | Answer quality |

---

## 9.8 Interview-Ready Talking Points

> **Q: How do you evaluate a RAG system?**
> A: Three layers: (1) Retrieval metrics (Recall@K, MRR, nDCG) to check if the right documents were found; (2) Generation metrics (F1, EM, ROUGE-L) to check if the answer is correct; (3) System metrics (latency, tokens) to check operational efficiency. For scale, we also use LLM-as-Judge for faithfulness and relevance scoring.

> **Q: What is nDCG and how is it computed?**
> A: Normalized Discounted Cumulative Gain. It computes DCG (sum of relevance scores discounted by log of position), then normalizes by the ideal DCG (perfect ranking). Range is [0,1]. It rewards relevant documents ranked higher and penalizes relevant documents ranked lower.

> **Q: What's the problem with Exact Match for evaluating RAG?**
> A: It's too strict. "NYC" vs "New York City" scores 0. "Dr. Thorne" vs "Aris Thorne" scores 0. Both are correct. That's why we also use F1 (partial credit for overlapping tokens) and LLM-as-Judge (semantic equivalence checking).

> **Q: How do you evaluate 1000 questions without manual checking?**
> A: Automated lexical metrics (EM, F1) for ground-truth-available datasets, plus LLM-as-Judge for semantic evaluation. The judge LLM rates each answer on faithfulness (is it grounded in context?) and relevance (does it answer the question?) on a 1-5 scale. This scales to any dataset size.

> **Q: What is an ablation study in the context of RAG?**
> A: Systematically disabling or changing one component at a time to measure its impact. For example: BM25-only → +Dense → +RRF → +Reranking → +Adaptive routing. Each step should show measurable improvement. If a component doesn't improve metrics, it's unnecessary complexity.
