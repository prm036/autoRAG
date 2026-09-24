# AutoRAG Documentation

A comprehensive, self-contained guide to every component, algorithm, and design decision in the AutoRAG system.

## Reading Order

| # | Page | What You'll Learn |
|---|---|---|
| 1 | [System Overview](01_overview.md) | Architecture, data flow, and the adaptive routing paradigm |
| 2 | [Ingestion Pipeline](02_ingestion.md) | Document parsing, text cleaning, and chunking strategies |
| 3 | [Embeddings & Vector Spaces](03_embeddings.md) | How text becomes numbers, sentence-transformers, cosine similarity |
| 4 | [Indexing](04_indexing.md) | BM25 math (TF-IDF, Okapi BM25), ChromaDB vector store, ANN search |
| 5 | [Retrieval Pipeline](05_retrieval.md) | Dense vs sparse retrieval, hybrid fusion with RRF, the full math |
| 6 | [Reranking](06_reranking.md) | Bi-encoders vs cross-encoders, why reranking matters |
| 7 | [Generation & Prompting](07_generation.md) | LLM integration, context construction, prompt engineering |
| 8 | [Adaptive RAG](08_adaptive_rag.md) | Query routing, single-hop pipeline, multi-hop decomposition |
| 9 | [Evaluation & Metrics](09_evaluation.md) | EM, F1, ROUGE-L, MRR, Recall@K, nDCG, LLM-as-Judge |

## Quick Reference

```
User Query
    │
    ▼
┌─────────────┐
│ Query Router │──► Classifies complexity: DIRECT / SINGLE_HOP / MULTI_HOP
└─────┬───────┘
      │
      ├── DIRECT ──────────────► LLM generates answer from parametric knowledge
      │
      ├── SINGLE_HOP ──► Hybrid Retrieval → Reranking → Context → LLM
      │
      └── MULTI_HOP ───► Decompose → Loop(Retrieve → Reason) → Synthesize
```
