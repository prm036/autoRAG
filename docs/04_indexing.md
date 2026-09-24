# 4. Indexing

After chunks are embedded, they need to be stored in searchable data structures. AutoRAG maintains **two parallel indexes** — a dense vector store and a sparse BM25 index — so that we can later perform hybrid retrieval.

**Code**: `src/indexing/vector_store.py`, `src/indexing/bm25_index.py`

---

## 4.1 Why Two Indexes?

Dense (vector) retrieval and sparse (BM25) retrieval have complementary strengths:

| Property | BM25 (Sparse) | Dense (Vector) |
|---|---|---|
| Matches on | Exact keywords | Semantic meaning |
| "car" vs "automobile" | ❌ No match | ✅ Match |
| Rare entity names | ✅ Excellent | ⚠️ Sometimes weak |
| Typos / abbreviations | ⚠️ Partial | ⚠️ Partial |
| Speed (1M docs) | ~5ms | ~10-50ms |
| Storage | Inverted index (small) | Float vectors (larger) |

By maintaining both, we can fuse their results later (Chapter 5) to get the best of both worlds.

---

## 4.2 BM25: The Math Behind Sparse Retrieval

BM25 (Best Matching 25) is the industry-standard ranking function for keyword-based retrieval. It's what powers search engines like Elasticsearch and Solr under the hood.

### Building Blocks: TF and IDF

BM25 is built on two classic information retrieval concepts:

#### Term Frequency (TF)
How often does a term appear in a document? More occurrences → more relevant.

```
TF(t, d) = count of term t in document d
```

But raw TF has a problem: a term appearing 100 times isn't 100x more relevant than appearing once. BM25 uses **saturating TF** to dampen this:

```
                    f(t, d) × (k₁ + 1)
Saturated TF = ─────────────────────────────
                f(t, d) + k₁ × (1 - b + b × |d|/avgdl)
```

Where:
- `f(t, d)` = raw frequency of term `t` in document `d`
- `k₁` = saturation parameter (default: **1.5**)
- `b` = length normalization parameter (default: **0.75**)
- `|d|` = length of document `d` (in tokens)
- `avgdl` = average document length across the corpus

**What k₁ controls**: How quickly TF saturates. At k₁=0, all non-zero TFs are treated equally. At k₁=∞, there's no saturation (linear TF).

**What b controls**: How much document length affects scoring. At b=0, no length normalization (long documents aren't penalized). At b=1, full length normalization (long documents are heavily penalized). The default **b=0.75** is a moderate penalty.

#### Inverse Document Frequency (IDF)
How rare is a term across the entire corpus? Rarer terms are more informative.

```
                         N - n(t) + 0.5
IDF(t) = log ──────────────────────────────
                         n(t) + 0.5
```

Where:
- `N` = total number of documents in the corpus
- `n(t)` = number of documents containing term `t`

**Intuition**: The word "the" appears in every document, so IDF("the") ≈ 0 (not informative). The word "QuantumFlow" might appear in only 1 out of 10,000 documents, so IDF("QuantumFlow") is very high (very informative).

### The Full BM25 Formula

For a query `Q` with terms `q₁, q₂, ..., qₙ`, the BM25 score of document `d` is:

```
                      n
BM25(Q, d) = Σ   IDF(qᵢ) × ──────────────────────────────────────
                     i=1              f(qᵢ, d) × (k₁ + 1)
                              ─────────────────────────────────────
                              f(qᵢ, d) + k₁ × (1 - b + b × |d|/avgdl)
```

### Worked Example

Imagine a tiny corpus of 3 chunks:
```
Chunk A: "BM25 is a ranking function used in information retrieval"
Chunk B: "Dense retrieval uses neural embeddings for search"
Chunk C: "BM25 and dense retrieval can be combined in hybrid search"
```

Query: **"BM25 ranking"**

Step 1 — IDF:
- IDF("BM25") = log((3 - 2 + 0.5) / (2 + 0.5)) = log(0.6) ≈ -0.51 (but capped at 0)
- IDF("ranking") = log((3 - 1 + 0.5) / (1 + 0.5)) = log(1.67) ≈ 0.51

Step 2 — Saturated TF (for Chunk A, assuming avgdl = 9 tokens):
- f("BM25", A) = 1, |A| = 9 tokens
- Saturated TF = 1 × (1.5 + 1) / (1 + 1.5 × (1 - 0.75 + 0.75 × 9/9)) = 2.5 / 2.5 = 1.0

Step 3 — Score: BM25 = IDF("ranking") × Saturated_TF("ranking") + ...

The exact numbers will vary, but the key insight is that **Chunk A scores highest** because it contains both query terms, and "ranking" has a higher IDF (it's rarer than "BM25" in this corpus).

### Our Implementation

```python
class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.bm25 = None  # BM25Okapi instance from rank_bm25

    def add_chunks(self, chunks):
        tokenized = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Return top-k chunks ranked by BM25 score
```

We use the `rank_bm25` library which implements the standard Okapi BM25 formula. The index is serialized to disk via `pickle` for persistence.

**Code**: See `src/indexing/bm25_index.py`

---

## 4.3 ChromaDB: The Vector Store

### What is a Vector Store?

A vector store is a database optimized for storing and searching high-dimensional vectors. When you have 100,000 chunks each with a 384-dimension embedding, you need a way to quickly find the "nearest" vectors to a query embedding.

### The Naive Approach: Brute-Force Search

Compare the query vector against every stored vector:

```python
for each stored_vector in index:
    similarity = cosine(query_vector, stored_vector)
    if similarity > threshold:
        results.append(stored_vector)
```

This is **O(N × D)** where N = number of vectors and D = dimensions. For 1M vectors × 384 dims = 384M floating-point operations per query. Feasible but slow (~100ms).

### Approximate Nearest Neighbor (ANN) Search

For production systems, we use ANN algorithms that trade a tiny amount of accuracy for massive speed improvements:

| Algorithm | Used By | Speed | Accuracy |
|---|---|---|---|
| HNSW | ChromaDB, Qdrant, Weaviate | ~1-5ms | ~95-99% |
| IVF-PQ | FAISS | ~0.5-2ms | ~90-95% |
| ScaNN | Google Vertex AI | ~0.5-1ms | ~95-99% |

### HNSW: How ChromaDB Searches

ChromaDB uses **HNSW (Hierarchical Navigable Small World)** graphs internally. Here's the intuition:

1. **Build phase**: Organize vectors into a multi-layer graph. Higher layers have fewer nodes (like an express highway), lower layers have more nodes (like local streets).
2. **Search phase**: Start at the top layer, greedily navigate to the nearest node, then drop down to the next layer and repeat. This is like "zooming in" from a bird's-eye view.

```
Layer 3 (few nodes):     A ──────── B ──────── C
                              │
Layer 2:                 A ── D ── B ── E ── C
                              │         │
Layer 1:                 A D F B G E H C I J
                              │
Layer 0 (all nodes):     A D F K B L G E M H C N I J O P
```

To find the nearest neighbor of query Q:
1. Start at node A in Layer 3
2. Compare distances: A→Q vs B→Q. B is closer → move to B
3. Drop to Layer 2 at B. Compare B, E. E is closer → move to E
4. Drop to Layer 1 at E. Compare E, G, H. G is closer → move to G
5. Drop to Layer 0 at G. Fine-grained search around G → find the exact nearest neighbors

This takes **O(log N)** instead of O(N).

### Our ChromaDB Configuration

```python
class VectorStore:
    def __init__(self, persist_directory, collection_name, similarity_metric):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": similarity_metric}  # "cosine"
        )

    def add_chunks(self, chunks):
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def search(self, query_embedding, top_k=10):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # Returns chunks ranked by cosine similarity
```

ChromaDB handles HNSW construction, persistence to disk, and efficient querying automatically.

**Code**: See `src/indexing/vector_store.py`

---

## 4.4 Index Comparison Summary

| Feature | BM25 Index | ChromaDB Vector Store |
|---|---|---|
| **What's stored** | Tokenized text + inverted index | Float32 embedding vectors |
| **Search algorithm** | BM25 scoring formula | HNSW approximate nearest neighbor |
| **Input at query time** | Tokenized query string | Query embedding (384-dim vector) |
| **Strengths** | Exact keywords, rare entities | Semantic similarity, synonyms |
| **Weaknesses** | No semantic understanding | Can miss exact keyword matches |
| **Persistence** | Pickle file (~MBs) | SQLite + binary files (~MBs-GBs) |
| **Speed (10K docs)** | ~1-2ms | ~2-5ms |

---

## 4.5 Interview-Ready Talking Points

> **Q: What are k₁ and b in BM25?**
> A: k₁ (default 1.5) controls term frequency saturation — how quickly repeated occurrences of a word stop increasing the score. b (default 0.75) controls document length normalization — how much longer documents are penalized.

> **Q: Why use ChromaDB over FAISS?**
> A: ChromaDB provides a higher-level API with built-in persistence, metadata filtering, and collection management. FAISS gives raw performance but requires more engineering around it. For a project of this scale, ChromaDB's developer experience is the right tradeoff.

> **Q: What is HNSW?**
> A: Hierarchical Navigable Small World — a graph-based approximate nearest neighbor algorithm. It builds a multi-layer graph where you start searching at a coarse top layer and progressively refine at lower layers. It achieves ~95-99% recall in O(log N) time versus O(N) for brute force.

> **Q: Why maintain two separate indexes?**
> A: BM25 and dense retrieval have complementary strengths. BM25 excels at exact keyword matching (critical for entity names, code, and rare terms), while dense retrieval captures semantic meaning (synonyms, paraphrases). Hybrid fusion combines both for better overall recall.

---

**Next**: [Chapter 5 — Retrieval Pipeline →](05_retrieval.md)
