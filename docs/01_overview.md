# 1. System Overview

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by giving them access to external knowledge at inference time. Instead of relying solely on what the model memorized during training (its **parametric knowledge**), we first **retrieve** relevant documents from a knowledge base, then **augment** the LLM's input with those documents so it can **generate** a grounded, factual answer.

### Why RAG Exists

LLMs have two fundamental problems:

1. **Knowledge Cutoff**: They only know what was in their training data. Ask about something that happened after training, and they'll either hallucinate or say they don't know.
2. **Hallucination**: Even for knowledge within their training window, LLMs sometimes confidently generate plausible-sounding but factually incorrect information.

RAG solves both problems by turning the LLM from a "closed-book exam" into an "open-book exam" — the model can look things up before answering.

### The RAG Equation

At its simplest, RAG can be expressed as:

```
answer = LLM(query + retrieved_context)
```

Where:
- `query` is the user's question
- `retrieved_context` is the relevant text pulled from a knowledge base
- `LLM()` generates a natural language answer conditioned on both

---

## The Problem with Simple RAG

The equation above assumes a single retrieval step, but real-world questions vary wildly in complexity:

| Query Type | Example | What's Needed |
|---|---|---|
| **Trivial** | "What is 2 + 2?" | No retrieval — the LLM knows this |
| **Simple factual** | "Who is the CEO of NexGen Tech?" | One retrieval step, one document |
| **Multi-hop** | "Who is the CEO of the company that acquired QuantumFlow?" | Multiple retrieval steps, reasoning across documents |

Using the same strategy for all three is wasteful:
- Running full retrieval for trivial questions wastes compute.
- A single retrieval step for multi-hop questions produces incomplete or wrong answers because the needed information is spread across multiple documents.

This is the core insight from the **Adaptive-RAG paper** (Jeong et al., 2024): we need a system that **adapts its strategy** based on query complexity.

---

## AutoRAG Architecture

AutoRAG implements the Adaptive-RAG paradigm with a full production-ready pipeline:

```
                    ┌──────────────────────────────────────────────────┐
                    │              INGESTION (Offline)                 │
                    │                                                  │
                    │  Documents → Parser → Cleaner → Chunker         │
                    │                          │                       │
                    │                    ┌─────┴──────┐               │
                    │                    ▼            ▼                │
                    │              Embedder      BM25 Index            │
                    │                    │                              │
                    │                    ▼                              │
                    │              ChromaDB                             │
                    └──────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────┐
                    │              QUERY TIME (Online)                  │
                    │                                                   │
                    │  User Query                                       │
                    │      │                                            │
                    │      ▼                                            │
                    │  ┌────────────┐                                   │
                    │  │Query Router│ → Classifies: DIRECT/SINGLE/MULTI │
                    │  └─────┬──────┘                                   │
                    │        │                                          │
                    │   ┌────┼────────────────────┐                    │
                    │   ▼    ▼                    ▼                    │
                    │ DIRECT  SINGLE-HOP        MULTI-HOP              │
                    │   │    Hybrid Retrieval    Decompose query        │
                    │   │      ├─ BM25           For each sub-query:   │
                    │   │      ├─ Dense            ├─ Hybrid Retrieval │
                    │   │      └─ RRF Fusion       ├─ Rerank           │
                    │   │    Rerank (Cross-Enc)    └─ Generate partial │
                    │   │    Build Context        Synthesize all       │
                    │   │    Generate Answer                           │
                    │   ▼         ▼                    ▼               │
                    │   └─────────┴────────────────────┘               │
                    │                    │                              │
                    │                    ▼                              │
                    │              RAGResponse                          │
                    │   (answer + sources + metrics + trace)            │
                    └──────────────────────────────────────────────────┘
```

---

## Data Flow: The Life of a Query

Let's trace a concrete multi-hop question through the system:

**Query**: *"Who is the CEO of the company that acquired QuantumFlow Innovations?"*

### Step 1: Routing (`src/adaptive/router.py`)
The Query Router sends the question to the LLM with a classification prompt. The LLM recognizes this requires connecting two facts (acquisition + CEO), so it returns `MULTI_HOP`.

### Step 2: Decomposition (`src/adaptive/multi_hop.py`)
The Multi-Hop Pipeline asks the LLM to break the question into sub-questions:
1. "Which company acquired QuantumFlow Innovations?"
2. "Who is the CEO of [that company]?"

### Step 3: First Hop — Retrieval
For sub-question 1, the Hybrid Retriever runs:
- **BM25** searches the token index for "QuantumFlow" + "acquired" → finds chunks from `quantumflow.md`
- **Dense** encodes the sub-question into a vector and searches ChromaDB → also finds chunks about the acquisition
- **RRF Fusion** merges both result lists into a single ranked list
- **Reranker** (cross-encoder) re-scores the top candidates for precise relevance

The top chunk contains: *"QuantumFlow Innovations was fully acquired by NexGen Tech for $1.2 Billion."*

### Step 4: First Hop — Intermediate Answer
The LLM reads the retrieved context and generates: *"NexGen Tech acquired QuantumFlow Innovations."*

### Step 5: Second Hop — Retrieval
For sub-question 2 ("Who is the CEO of NexGen Tech?"), the same retrieval process runs. This time it finds chunks from `nexgen.md` containing: *"The company is currently led by CEO Sarah Jenkins."*

### Step 6: Second Hop — Intermediate Answer
The LLM generates: *"Sarah Jenkins is the CEO of NexGen Tech."*

### Step 7: Synthesis
The Multi-Hop Pipeline sends the original question and all accumulated evidence to the LLM for a final synthesis:

> **Answer**: "Sarah Jenkins is the CEO of NexGen Tech, the company that acquired QuantumFlow Innovations in March 2024 for $1.2 Billion."

### Step 8: Response
The system packages the answer, source chunks, latency breakdown, token usage, and execution trace into a `RAGResponse` and returns it to the user.

---

## Key Design Decisions

| Decision | Our Choice | Why |
|---|---|---|
| **Retrieval Strategy** | Hybrid (BM25 + Dense) | BM25 excels at exact keyword matches; Dense excels at semantic similarity. Together they cover more ground than either alone. |
| **Fusion Method** | Reciprocal Rank Fusion (RRF) | Score-agnostic, parameter-free, and well-studied. Doesn't require normalizing incompatible score distributions. |
| **Reranking** | Cross-encoder | Far more accurate than bi-encoders for relevance scoring, at the cost of speed. We only apply it to the top candidates. |
| **Routing** | LLM-based classifier | More flexible than rule-based routing. Can understand query semantics and intent. |
| **Multi-hop** | Query decomposition | Explicit decomposition is more interpretable and debuggable than implicit chain-of-thought approaches. |

---

## Code Map

| Component | Directory | Key Files |
|---|---|---|
| Data models | `src/` | `models.py` |
| Ingestion | `src/ingestion/` | `parser.py`, `cleaner.py`, `chunker.py` |
| Embeddings | `src/embeddings/` | `embedder.py` |
| Indexing | `src/indexing/` | `vector_store.py`, `bm25_index.py` |
| Retrieval | `src/retrieval/` | `dense_retriever.py`, `sparse_retriever.py`, `hybrid_retriever.py`, `reranker.py` |
| Generation | `src/generation/` | `llm.py`, `context_builder.py`, `prompts.py` |
| Adaptive RAG | `src/adaptive/` | `router.py`, `single_hop.py`, `multi_hop.py` |
| Evaluation | `src/evaluation/` | `retrieval_metrics.py`, `generation_metrics.py`, `system_metrics.py`, `experiment_runner.py` |
| Observability | `src/observability/` | `logger.py` |
| API | `src/api/` | `app.py`, `schemas.py` |

---

**Next**: [Chapter 2 — Ingestion Pipeline →](02_ingestion.md)
