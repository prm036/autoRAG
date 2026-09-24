# 6. Reranking

Reranking is the second stage of our retrieval pipeline. After the first-pass retriever returns ~30 candidates, the reranker re-scores them with a much more powerful model to select the top ~5 most relevant chunks.

**Code**: `src/retrieval/reranker.py`

---

## 6.1 Why Rerank?

First-pass retrieval (BM25 + Dense) is designed for **speed** — it must search through thousands or millions of chunks in milliseconds. This speed comes at a cost: the scoring models are relatively simple.

- **BM25**: Scores based on word overlap only. No understanding of meaning.
- **Dense bi-encoder**: Independently embeds query and document. Cannot model fine-grained interactions between the query and document tokens.

A reranker uses a much more powerful model that sees the query and document **together**, enabling it to capture nuanced relevance signals that first-pass retrievers miss.

### Example: Why First-Pass Fails

**Query**: "When was the company that makes the iPhone founded?"

**Chunk A** (relevant): "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne."

**Chunk B** (misleading): "The iPhone was first released on June 29, 2007, revolutionizing the smartphone industry."

A bi-encoder might rank Chunk B higher because "iPhone" directly appears. A cross-encoder would understand that the question asks about the founding date, not the product release, and rank Chunk A higher.

---

## 6.2 Bi-Encoders vs Cross-Encoders

This is one of the most important distinctions in modern NLP. Understanding it deeply will serve you well in interviews.

### Bi-Encoder (Used in First-Pass Retrieval)

```
Query: "When was Apple founded?"    Document: "Apple was founded in 1976"
         │                                      │
         ▼                                      ▼
   ┌──────────┐                          ┌──────────┐
   │Transformer│                          │Transformer│
   │ Encoder   │                          │ Encoder   │
   └─────┬────┘                          └─────┬────┘
         │                                      │
         ▼                                      ▼
   Query Embedding                       Document Embedding
     [0.2, -0.4, ...]                    [0.3, -0.3, ...]
         │                                      │
         └──────────── cosine sim ──────────────┘
                         │
                     Score: 0.87
```

**Key Properties**:
- Query and document are encoded **independently** (in separate forward passes)
- Embeddings can be **pre-computed** and cached
- Comparison is a simple dot product → **O(1) per candidate**
- **Scales to millions** of documents because you only embed the query once

**Limitation**: Because the query and document never "see" each other during encoding, the model can't capture fine-grained token-level interactions. It produces a coarse semantic similarity score.

### Cross-Encoder (Used in Reranking)

```
Query: "When was Apple founded?"  +  Document: "Apple was founded in 1976"
                    │
                    ▼
        ┌─────────────────────┐
        │    [CLS] Query      │
        │    [SEP] Document   │    ← Concatenated as one input
        │    [SEP]            │
        │                     │
        │    Transformer      │    ← Full attention across ALL tokens
        │    Encoder          │       Query tokens attend to document tokens
        │    12 layers        │       and vice versa
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
            Classification Head
                   │
                   ▼
             Score: 0.94
```

**Key Properties**:
- Query and document are processed **together** as a single concatenated input
- Full **cross-attention** between every query token and every document token
- Produces a much more accurate relevance score
- **Cannot pre-compute** — must run a full forward pass for each (query, document) pair
- **O(N) per query** where N = number of candidates → does not scale to millions

### The Tradeoff

| Property | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Accuracy | Good | Excellent |
| Speed (per pair) | ~0.1ms (dot product) | ~5-10ms (full forward pass) |
| Scalability | Millions of docs | Tens of candidates |
| Pre-computation | ✅ Yes | ❌ No |
| Token interactions | ❌ Independent encoding | ✅ Full cross-attention |

### Why We Use Both

This is the beauty of the two-stage pipeline:

1. **Stage 1 (Bi-encoder)**: Embed the query once, dot product against 100K+ pre-computed document embeddings → retrieve ~30 candidates in ~5ms
2. **Stage 2 (Cross-encoder)**: Score each of the ~30 candidates against the query → select top ~5 in ~150ms

Total: ~155ms for extremely high-quality retrieval over a large corpus.

---

## 6.3 Our Reranking Model: ms-marco-MiniLM-L-6-v2

We use `cross-encoder/ms-marco-MiniLM-L-6-v2`.

| Property | Value |
|---|---|
| Architecture | 6-layer MiniLM cross-encoder |
| Parameters | ~22M |
| Trained on | MS MARCO passage ranking dataset |
| Task | Binary relevance classification (relevant / not relevant) |
| Output | Logit score (higher = more relevant) |

### Why This Model?

- **MS MARCO trained**: MS MARCO is the largest public passage ranking dataset (~500K queries, ~8.8M passages). Models trained on it generalize well to other domains.
- **MiniLM-L-6**: Only 6 transformer layers (vs 12 for BERT-base), so it's 2x faster while retaining ~95% of the accuracy.
- **Industry standard**: This exact model is used in production at companies like Pinecone, Weaviate, and countless RAG systems.

---

## 6.4 The Reranking Pipeline

```python
class Reranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=5):
        # Create (query, chunk_text) pairs
        pairs = [(query, sc.chunk.text) for sc in candidates]

        # Score all pairs in a single batch
        scores = self.model.predict(pairs, batch_size=32)

        # Sort by cross-encoder score (descending)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # Return top_k
        return [ScoredChunk(chunk=sc.chunk, score=score, source="reranker")
                for sc, score in scored[:top_k]]
```

### What Happens Internally

For each (query, chunk) pair:

1. Concatenate: `[CLS] query text [SEP] chunk text [SEP]`
2. Tokenize into subword tokens (WordPiece)
3. Forward pass through 6 transformer layers with full self-attention
4. Extract the `[CLS]` token's hidden state
5. Pass through a linear classification head → scalar relevance score

### Batch Processing

We score all 30 candidates in a single batch call. This is crucial for performance:
- **Without batching**: 30 × 5ms = 150ms (sequential)
- **With batching**: 30 pairs processed in parallel = ~50ms (with GPU) or ~200ms (CPU)

---

## 6.5 Configuration

```yaml
reranking:
  enabled: true
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5            # Number of chunks to keep after reranking
  batch_size: 32
```

### Why top_k = 5?

After reranking, we only keep the top 5 chunks for context construction. This is because:
- More chunks = more input tokens for the LLM = higher cost and latency
- Diminishing returns: the 6th most relevant chunk rarely changes the answer
- 3-5 is the standard in production RAG systems
- Each chunk is ~250 tokens, so 5 chunks ≈ 1,250 tokens of context (well within limits)

---

## 6.6 When to Skip Reranking

Reranking is optional. You might skip it when:
- **Latency is critical**: Reranking adds ~50-200ms per query
- **Corpus is small**: With <1000 chunks, BM25 + dense already gives good results
- **First-pass quality is high**: If your embedding model is very good and the queries are simple

In our system, you can disable it:
```yaml
reranking:
  enabled: false
```

---

## 6.7 Interview-Ready Talking Points

> **Q: What's the difference between a bi-encoder and a cross-encoder?**
> A: A bi-encoder encodes query and document independently into separate vectors and compares them with cosine similarity — fast but coarse. A cross-encoder processes query and document together as one concatenated input with full cross-attention — slow but much more accurate. We use bi-encoders for first-pass retrieval (scale to millions) and cross-encoders for reranking (precision on tens of candidates).

> **Q: Why not just use cross-encoders for everything?**
> A: Cross-encoders require a forward pass for every (query, document) pair. For 100K documents, that's 100K forward passes per query (~500 seconds). Bi-encoders pre-compute document embeddings once, then each query is a single encoding + dot products (~5ms).

> **Q: Why rerank at all? Isn't the hybrid retriever good enough?**
> A: Hybrid retrieval optimizes for recall (finding relevant documents). Reranking optimizes for precision (eliminating irrelevant ones). The cross-encoder captures fine-grained token interactions — like understanding that "founded" vs "founded in" vs "co-founded" matter differently depending on the query — that bi-encoders miss.

---

**Next**: [Chapter 7 — Generation & Prompting →](07_generation.md)
