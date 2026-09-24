# 3. Embeddings & Vector Spaces

Embeddings are the bridge between human language and mathematical search. They convert text into dense numerical vectors so that **semantic similarity** can be computed with simple arithmetic.

**Code**: `src/embeddings/embedder.py`

---

## 3.1 What is an Embedding?

An embedding is a fixed-length vector of floating-point numbers that represents the "meaning" of a piece of text. For example:

```
"The cat sat on the mat"  →  [0.12, -0.45, 0.78, 0.03, ..., -0.22]  (384 dimensions)
"A kitten was on the rug" →  [0.11, -0.43, 0.76, 0.05, ..., -0.20]  (384 dimensions)
"Stock market crashed"    →  [0.89, 0.12, -0.67, 0.45, ..., 0.33]   (384 dimensions)
```

Notice that the first two vectors are very similar (close numbers), while the third is very different. That's the magic: **semantically similar texts produce nearby vectors**.

### Why Not Just Use Keywords?

Keyword search (like BM25) matches exact tokens. It would fail on:
- **Synonyms**: Searching "car" won't find documents about "automobile"
- **Paraphrasing**: "How to fix a bug" vs "debugging techniques"
- **Conceptual similarity**: "climate change effects" vs "global warming impact"

Embeddings capture meaning, not surface tokens. The embedding for "car" and "automobile" will be nearly identical.

---

## 3.2 How Embeddings Are Trained

Modern text embedding models are typically **transformer-based** neural networks trained with **contrastive learning**:

### The Contrastive Learning Objective

Given a training set of (query, positive_passage, negative_passage) triplets:

```
Loss = max(0, similarity(query, negative) - similarity(query, positive) + margin)
```

This pushes the model to:
- **Pull together** embeddings of semantically related texts (query ↔ positive)
- **Push apart** embeddings of unrelated texts (query ↔ negative)

### The Training Process

```
Input text: "Retrieval augmented generation combines search with LLMs"
    │
    ▼
┌─────────────────┐
│  Transformer     │  (e.g., BERT, MiniLM)
│  Encoder         │
│  12 layers       │
│  384-dim output  │
└────────┬────────┘
         │
         ▼
    Mean Pooling        ← Average all token embeddings into one vector
         │
         ▼
    L2 Normalize        ← Scale to unit length (important for cosine similarity)
         │
         ▼
    [0.12, -0.45, ...]  ← Final 384-dimensional embedding
```

---

## 3.3 Our Embedding Model: BGE-small-en-v1.5

We use `BAAI/bge-small-en-v1.5` from the Beijing Academy of AI (BAAI).

| Property | Value |
|---|---|
| Architecture | BERT-based transformer |
| Parameters | 33M |
| Dimensions | 384 |
| Max sequence length | 512 tokens |
| Training data | Large-scale text pairs (MS MARCO, NQ, etc.) |
| MTEB Benchmark rank | Top-tier for its size class |

### Why This Model?

- **Size vs quality tradeoff**: At 33M parameters, it's small enough to run on CPU in milliseconds, yet produces embeddings competitive with models 10x larger.
- **Industry standard**: Widely used in production RAG systems. Recognized by interviewers.
- **Normalized outputs**: Produces unit-length vectors, making cosine similarity equivalent to a simple dot product.

---

## 3.4 Similarity Metrics

Once we have embeddings, we need a way to measure how "close" two vectors are. There are three common metrics:

### Cosine Similarity (Our Default)

Measures the angle between two vectors, ignoring their magnitude:

```
                    A · B           Σ(aᵢ × bᵢ)
cos(A, B) = ─────────────── = ────────────────────
              ‖A‖ × ‖B‖      √Σ(aᵢ²) × √Σ(bᵢ²)
```

- Range: [-1, 1] (1 = identical direction, 0 = orthogonal, -1 = opposite)
- **When vectors are L2-normalized** (unit length), cosine similarity = dot product: `cos(A, B) = A · B`
- This is why we normalize our embeddings — it makes search faster.

### Euclidean Distance (L2)

Measures the straight-line distance between two points in vector space:

```
L2(A, B) = √Σ(aᵢ - bᵢ)²
```

- Range: [0, ∞) — lower is more similar
- For normalized vectors: `L2² = 2 - 2·cos(A,B)`, so it's equivalent to cosine similarity.

### Inner Product (Dot Product)

```
IP(A, B) = Σ(aᵢ × bᵢ)
```

- Range: (-∞, ∞) — higher is more similar
- For normalized vectors: identical to cosine similarity
- Fastest to compute (no square root needed)

### Why Cosine Similarity?

For normalized vectors, all three metrics produce the same ranking. We default to cosine because:
1. It's scale-invariant (robust to varying document lengths)
2. It's the most widely understood metric in the industry
3. Our embedding model produces normalized vectors, so cosine = dot product = fast

---

## 3.5 The Embedding Pipeline in Code

```python
class Embedder:
    def __init__(self, model_name, device, batch_size, normalize):
        self.model = SentenceTransformer(model_name, device=device)
        # Model is lazy-loaded on first use

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed a batch of chunks efficiently."""
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, batch_size=64, normalize_embeddings=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        return chunks
```

### Batch Encoding

Embedding one chunk at a time is slow because of GPU/CPU overhead per call. Instead, we encode in **batches** (default: 64 chunks at once). The model processes all 64 texts in a single forward pass through the transformer, which is dramatically faster.

---

## 3.6 The Curse of Dimensionality

### Why 384 Dimensions?

Each dimension in the embedding captures some abstract "feature" of the text. More dimensions = more expressive power, but:

| Dimensions | Model Example | Storage per vector | Search speed |
|---|---|---|---|
| 128 | Very small models | 512 bytes | Fastest |
| 384 | bge-small, MiniLM | 1.5 KB | Fast |
| 768 | bge-base, BERT-base | 3 KB | Moderate |
| 1024 | bge-large | 4 KB | Slower |
| 1536 | OpenAI ada-002 | 6 KB | Slowest |

384 dimensions is the sweet spot for local/CPU deployment. It's enough to capture rich semantics while keeping memory usage and search latency low.

### What About 1 Million Documents?

With 384-dim float32 vectors:
- 1M vectors × 1.5 KB = **1.5 GB** of raw vector data
- This fits comfortably in RAM on any modern machine
- Search over 1M vectors takes ~10-50ms with ANN indexing (covered in Chapter 4)

---

## 3.7 Interview-Ready Talking Points

> **Q: What is a text embedding?**
> A: A fixed-length numerical vector that captures the semantic meaning of text. Similar texts produce nearby vectors in the embedding space.

> **Q: How is cosine similarity computed?**
> A: It's the dot product of two vectors divided by the product of their magnitudes. For unit-normalized vectors, it simplifies to just the dot product. Range is [-1, 1].

> **Q: Why use a small embedding model instead of OpenAI's?**
> A: Cost and latency. A 33M-parameter local model embeds thousands of chunks per second for free, while API-based models charge per token and add network latency. For our use case, the quality difference is negligible.

> **Q: What's the difference between bi-encoder embeddings and cross-encoder scores?**
> A: Bi-encoders embed query and document independently (fast, scalable to millions). Cross-encoders process query and document together as a pair (slow, but much more accurate). We use bi-encoders for initial retrieval and cross-encoders for reranking. This is covered in detail in Chapter 6.

---

**Next**: [Chapter 4 — Indexing →](04_indexing.md)
