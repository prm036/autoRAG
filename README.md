# Adaptive Hybrid RAG for Multi-Hop Question Answering

An end-to-end Adaptive Retrieval-Augmented Generation system with hybrid sparse/dense retrieval, cross-encoder reranking, multi-hop retrieval, and systematic evaluation.

## Architecture

```
Documents → Parsing/Cleaning → Chunking → Embeddings → Vector Index + BM25 Index
    → Hybrid Retrieval (RRF) → Cross-Encoder Reranking → Context Selection → LLM Generation
```

The **Adaptive RAG controller** examines each query and routes it to:
- **Direct LLM** — for simple questions answerable from general knowledge
- **Single-hop RAG** — retrieve once → rerank → generate
- **Multi-hop RAG** — decompose → iterate (retrieve → reason) → synthesize

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[all]"
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY=your-key-here
# OR
export GEMINI_API_KEY=your-key-here
```

### 3. Ingest Documents

Place your documents in `data/raw/`, then:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/raw/"}'
```

### 4. Start the API

```bash
uvicorn src.api.app:app --reload
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

## Project Structure

```
src/
├── ingestion/          # Document parsing, cleaning, chunking
├── embeddings/         # Embedding model wrapper
├── indexing/           # Vector store (ChromaDB) + BM25 index
├── retrieval/          # Dense, sparse, hybrid (RRF), reranker
├── generation/         # LLM client, context builder, prompts
├── adaptive/           # Query router, single-hop, multi-hop pipelines
├── evaluation/         # Retrieval, generation, and system metrics
├── observability/      # Structured query logging
└── api/                # FastAPI serving layer
```

## Docker

```bash
docker-compose up --build
```

## Configuration

All hyperparameters are in `config/default.yaml`:
- Chunking strategy and size
- Embedding model
- Retrieval top-K values and RRF constant
- Cross-encoder reranking model
- LLM provider and model
- Adaptive routing parameters

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Sparse Retrieval | BM25 (rank_bm25) |
| Vector Store | ChromaDB |
| Reranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| LLM | OpenAI / Gemini / Ollama |
| API | FastAPI |
| Containerization | Docker |

## License

MIT
