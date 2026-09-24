# 5. Retrieval Pipeline

The retrieval pipeline is the search engine of AutoRAG. Given a user query, it finds the most relevant chunks from our knowledge base using a combination of sparse and dense retrieval, fused together with Reciprocal Rank Fusion.

**Code**: `src/retrieval/sparse_retriever.py`, `src/retrieval/dense_retriever.py`, `src/retrieval/hybrid_retriever.py`

---

## 5.1 The Two-Stage Retrieval Paradigm

Modern information retrieval uses a **two-stage** pipeline:

```
Stage 1: First-Pass Retrieval (Fast, Broad)
    Query → BM25 Index ──→ 30 candidates  ─┐
    Query → Vector Store ─→ 30 candidates  ─┤
                                             ├→ RRF Fusion → 30 merged candidates
                                             │
Stage 2: Reranking (Slow, Precise)                                 │
    30 candidates → Cross-Encoder → 5 final candidates ←──────────┘
```

**Stage 1** is designed for **high recall** — find as many potentially relevant chunks as possible, even if some are noisy. We use cheap, fast algorithms (BM25 + ANN vector search).

**Stage 2** is designed for **high precision** — from those candidates, identify the truly relevant ones. We use an expensive but accurate cross-encoder. (Covered in Chapter 6.)

---

## 5.2 Sparse Retrieval (BM25)

The sparse retriever wraps the BM25 index (built in Chapter 4) with a simple interface:

```python
class SparseRetriever:
    def retrieve(self, query: str, top_k: int = 30) -> RetrievalResult:
        scored_chunks = self.bm25_index.search(query, top_k)
        return RetrievalResult(query=query, method="bm25", candidates=scored_chunks)
```

### What "Sparse" Means

In information retrieval, "sparse" refers to the representation. A BM25 query is represented as a **sparse vector** where:
- Each dimension corresponds to a word in the vocabulary
- Most dimensions are 0 (the query only contains a few words)
- Non-zero entries are the BM25 weights

For a vocabulary of 100,000 words and a query of 5 words, only 5 out of 100,000 dimensions are non-zero — hence "sparse."

**Code**: See `src/retrieval/sparse_retriever.py`

---

## 5.3 Dense Retrieval

The dense retriever embeds the query into a vector and searches the ChromaDB vector store:

```python
class DenseRetriever:
    def retrieve(self, query: str, top_k: int = 30) -> RetrievalResult:
        query_embedding = self.embedder.embed_text(query)
        scored_chunks = self.vector_store.search(query_embedding, top_k)
        return RetrievalResult(query=query, method="dense", candidates=scored_chunks)
```

### What "Dense" Means

Every dimension of the 384-dim embedding vector has a non-zero value — hence "dense." This is in contrast to the sparse BM25 representation where most dimensions are zero.

### The Embedding Space

You can think of the vector store as a 384-dimensional space where every chunk occupies a point. When a query comes in:

1. The query is embedded into the same space
2. We find the K closest chunk-points to the query-point
3. "Closeness" is measured by cosine similarity

```
         ●c3
        /
    ●c1 ── ●Q (query)     ← c1 is the nearest neighbor
      \   /
       ●c2
              ●c4          ← c4 is far away (low similarity)
```

**Code**: See `src/retrieval/dense_retriever.py`

---

## 5.4 Why Hybrid Retrieval?

Neither sparse nor dense retrieval alone is sufficient:

### Case 1: BM25 Wins

**Query**: "What is the HNSW algorithm?"
- BM25 finds chunks containing the exact term "HNSW" → ✅ High relevance
- Dense might return chunks about "graph algorithms" or "nearest neighbor search" that don't mention HNSW specifically → ⚠️ Lower precision

### Case 2: Dense Wins

**Query**: "How to fix a slow database"
- BM25 looks for exact tokens "fix", "slow", "database" → ⚠️ Might miss relevant chunks that say "optimize query performance"
- Dense understands semantic similarity → ✅ Finds chunks about "database optimization" and "query performance tuning"

### Case 3: Both Contribute

**Query**: "BM25 ranking in information retrieval"
- BM25 finds chunks with exact match on "BM25" (rare term, high IDF)
- Dense finds chunks about "ranking algorithms" and "relevance scoring" that complement the BM25 results

**Empirical evidence**: The Adaptive-RAG paper and numerous benchmarks show that hybrid retrieval consistently outperforms either method alone by 5-15% on Recall@K.

---

## 5.5 Reciprocal Rank Fusion (RRF)

### The Problem

BM25 and dense retrieval produce scores on completely different scales:
- BM25 scores might range from 0 to 25
- Cosine similarity scores range from -1 to 1

You **cannot** simply average these scores. Even min-max normalization is problematic because score distributions are different.

### The Solution: RRF

Reciprocal Rank Fusion ignores the raw scores entirely. It only uses the **rank** (position in the sorted list). This makes it score-agnostic.

### The RRF Formula

For a document `d` retrieved by multiple systems, the RRF score is:

```
                    1
RRF(d) = Σ  ──────────────
         s∈S   k + rankₛ(d)
```

Where:
- `S` = set of retrieval systems (in our case: {BM25, Dense})
- `rankₛ(d)` = rank of document `d` in system `s`'s results (1-indexed)
- `k` = smoothing constant (default: **60**)

### Why k = 60?

The constant `k` controls how much we trust higher-ranked vs lower-ranked documents:

- **Small k (e.g., 1)**: Top-ranked documents get a massively higher score. The fusion heavily favors whatever each system ranks #1.
- **Large k (e.g., 1000)**: Scores are nearly flat across all ranks. Rank position barely matters.
- **k = 60**: The original RRF paper (Cormack et al., 2009) found that k=60 provides the best balance. It gives a meaningful boost to top-ranked documents while still considering lower-ranked ones.

### Worked Example

Two systems retrieve documents for the query "BM25 ranking":

```
BM25 Results:         Dense Results:
  Rank 1: Chunk_A       Rank 1: Chunk_C
  Rank 2: Chunk_B       Rank 2: Chunk_A
  Rank 3: Chunk_D       Rank 3: Chunk_E
  Rank 4: Chunk_C       Rank 4: Chunk_B
```

RRF scores (k=60):

```
Chunk_A:  1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252  ← HIGHEST
Chunk_C:  1/(60+4) + 1/(60+1) = 0.01563 + 0.01639 = 0.03202
Chunk_B:  1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03175
Chunk_D:  1/(60+3) + 0         = 0.01587               = 0.01587
Chunk_E:  0        + 1/(60+3)  = 0.01587               = 0.01587
```

**Result**: Chunk_A ranks highest because it appeared in **both** lists (rank 1 in BM25, rank 2 in Dense). Documents that appear in both systems get a natural boost — this is the key insight of RRF.

### Properties of RRF

1. **Score-agnostic**: Works regardless of the scale or distribution of input scores.
2. **Parameter-free** (almost): The only parameter is k, and k=60 works well universally.
3. **Rewards agreement**: Documents ranked highly by multiple systems score highest.
4. **Handles missing documents**: If a document appears in only one system's results, it still gets a score (just lower).

### Our Implementation

```python
class HybridRetriever:
    def retrieve(self, query, top_k=30):
        # Run both retrievers in parallel
        sparse_result = self.sparse_retriever.retrieve(query, self.sparse_top_k)
        dense_result = self.dense_retriever.retrieve(query, self.dense_top_k)

        # Compute RRF scores
        rrf_scores = {}
        for rank, sc in enumerate(sparse_result.candidates, 1):
            rrf_scores[sc.chunk.chunk_id] = 1.0 / (self.rrf_k + rank)

        for rank, sc in enumerate(dense_result.candidates, 1):
            chunk_id = sc.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score and return top_k
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return top_k results
```

**Code**: See `src/retrieval/hybrid_retriever.py`

---

## 5.6 Configuration

```yaml
retrieval:
  sparse_top_k: 30     # BM25 first-pass candidates
  dense_top_k: 30      # Dense first-pass candidates
  hybrid_top_k: 30     # After RRF fusion
  rrf_k: 60            # RRF smoothing constant (standard default)
```

### Why top_k = 30?

- Too low (e.g., 5): Might miss relevant documents, especially for ambiguous queries.
- Too high (e.g., 200): Returns too much noise, slowing down the reranker.
- 20-50 is the standard range. We use 30 as a good balance.

---

## 5.7 Interview-Ready Talking Points

> **Q: What is hybrid retrieval and why use it?**
> A: Hybrid retrieval combines sparse (BM25) and dense (vector) retrieval. BM25 excels at exact keyword matching (important for entity names and rare terms), while dense retrieval captures semantic similarity (synonyms, paraphrases). Together they achieve higher recall than either alone.

> **Q: Explain Reciprocal Rank Fusion.**
> A: RRF merges ranked lists from multiple retrieval systems using only rank positions, not raw scores. The formula is `1/(k + rank)` summed across systems. k=60 is the standard. It's elegant because it's score-agnostic — you don't need to normalize incompatible score distributions.

> **Q: Why not just normalize the scores and average them?**
> A: BM25 and cosine similarity have fundamentally different score distributions. BM25 scores are unbounded and sparse, while cosine similarity is bounded [-1, 1] and dense. Even with min-max normalization, the distributions don't align well. RRF avoids this entirely by using rank positions only.

> **Q: What's the two-stage retrieval paradigm?**
> A: Stage 1 (first-pass) uses fast, scalable methods (BM25 + ANN) to retrieve a broad set of ~30 candidates with high recall. Stage 2 (reranking) uses an expensive but accurate cross-encoder to re-score and select the top ~5 with high precision.

---

**Next**: [Chapter 6 — Reranking →](06_reranking.md)
