# 2. Ingestion Pipeline

The ingestion pipeline transforms raw documents (PDFs, text files, markdown) into indexed, searchable chunks. This is the **offline** phase — it happens once before any queries are processed.

```
Raw Files → Parser → Document → Cleaner → Cleaned Document → Chunker → Chunks[]
```

**Code**: `src/ingestion/parser.py`, `src/ingestion/cleaner.py`, `src/ingestion/chunker.py`

---

## 2.1 Document Parsing

### What It Does
The parser reads files from disk and converts them into our internal `Document` data structure. Different file formats require different parsing strategies.

### Supported Formats

| Format | Library | Notes |
|---|---|---|
| `.txt` | Built-in Python `open()` | Simplest case — just read the file |
| `.pdf` | `PyPDF2` | Extracts text page-by-page. Cannot handle scanned PDFs (those need OCR). |
| `.md` | Built-in Python `open()` | Read as plain text, preserving markdown structure |

### The Document Object
After parsing, every file becomes a `Document`:

```python
class Document(BaseModel):
    metadata: DocumentMetadata  # doc_id, title, source path, format
    text: str                    # The full text content
```

The `DocumentMetadata` object carries a UUID (`doc_id`) that uniquely identifies this document throughout the entire pipeline. This is crucial for **provenance tracking** — when the system cites a source, we can trace it all the way back to the original file.

**Code**: See `src/ingestion/parser.py` — the `parse_file()` and `parse_directory()` methods.

---

## 2.2 Text Cleaning

### Why Clean?
Raw text from PDFs and other documents is often messy:
- Extra whitespace and newlines from PDF column layouts
- Unicode artifacts (smart quotes, em-dashes, non-breaking spaces)
- Headers, footers, and page numbers mixed into the text

If we feed this noisy text into the embedding model, the embeddings will capture the noise rather than the meaning. The cleaner normalizes all of this.

### What the Cleaner Does

| Step | Before | After |
|---|---|---|
| Unicode normalization | `café` (two representations) | `café` (canonical form) |
| Whitespace collapse | `"Hello    \n\n   world"` | `"Hello world"` |
| Control character removal | `"text\x00\x01here"` | `"texthere"` |

**Code**: See `src/ingestion/cleaner.py` — the `clean()` method.

---

## 2.3 Chunking

### Why Chunk?

LLMs have a limited **context window** (the maximum amount of text they can process at once). Even GPT-4 with 128K tokens can't ingest an entire book. More importantly:

1. **Retrieval precision**: If you store entire documents, a search for "BM25 algorithm" might return a 50-page paper when only one paragraph is relevant. Smaller chunks mean more precise retrieval.
2. **Embedding quality**: Embedding models (like `bge-small-en-v1.5`) are trained on short passages (~256 tokens). If you feed them a 10,000-word document, the resulting embedding is a blurry average of all the topics in the document.

### The Chunking Dilemma

There's a fundamental tension:

- **Too small** → Chunks lose context. A chunk saying "He was born in 1990" is useless without knowing who "He" refers to.
- **Too large** → Chunks dilute relevance. A 5000-character chunk about many topics will weakly match any of them.

The sweet spot is typically **200-500 tokens** (roughly 800-2000 characters).

### Strategy 1: Fixed-Size Chunking

The simplest approach. Split text into blocks of exactly `chunk_size` characters, with `chunk_overlap` characters shared between consecutive chunks.

```
Document:  [AAAAAAAAAA|BBBBBBBBBB|CCCCCCCCCC]
                      ↕ overlap  ↕ overlap
Chunk 1:   [AAAAAAAAAA|BB]
Chunk 2:              [BB|BBBBBBBB|CC]
Chunk 3:                          [CC|CCCCCCCC]
```

**Pros**: Simple, predictable chunk sizes.
**Cons**: Can split mid-sentence or even mid-word.

### Strategy 2: Sentence-Aware Chunking (Our Default)

Uses NLTK's `sent_tokenize()` to first split text into sentences, then greedily packs sentences into chunks up to `chunk_size` characters. Overlap is achieved by carrying forward trailing sentences from the previous chunk.

```
Sentences: [S1, S2, S3, S4, S5, S6, S7, S8]

Chunk 1: [S1, S2, S3]        (total chars ≤ chunk_size)
Chunk 2: [S3, S4, S5, S6]    (S3 is the overlap from Chunk 1)
Chunk 3: [S6, S7, S8]        (S6 is the overlap from Chunk 2)
```

**Pros**: Never splits mid-sentence. Produces more semantically coherent chunks.
**Cons**: Chunk sizes vary slightly. Requires NLTK sentence tokenizer.

### Our Default Configuration

```yaml
chunking:
  strategy: "sentence"
  chunk_size: 1000        # ~200-250 tokens
  chunk_overlap: 200      # ~20% overlap
  min_chunk_size: 100     # Discard tiny trailing chunks
```

### Why Overlap Matters

Overlap ensures that information near chunk boundaries isn't lost. Consider this text:

> "Dr. Aris Thorne founded QuantumFlow Innovations in 2018. The company was acquired by NexGen Tech in 2024."

If the chunk boundary falls between these two sentences (no overlap), then no single chunk contains both the founder's name and the acquisition. With overlap, at least one chunk will contain both facts, which is critical for multi-hop reasoning.

### The Chunk Object

After chunking, each piece of text becomes a `Chunk` with full provenance:

```python
class Chunk(BaseModel):
    chunk_id: str       # Unique UUID
    doc_id: str         # Links back to parent Document
    text: str           # The actual text content
    position: int       # 0-indexed position within the document
    start_char: int     # Character offset in original document
    end_char: int       # Character offset end
    metadata: dict      # Inherited from the parent document
    embedding: list[float] | None  # Populated later by the Embedder
```

The `start_char` / `end_char` fields are important for **source attribution** — when the system tells the user "I found this in quantumflow.md", it can point to the exact character range.

**Code**: See `src/ingestion/chunker.py` — the `chunk_document()` method and the `_sentence_chunk()` / `_fixed_chunk()` strategies.

---

## 2.4 Interview-Ready Talking Points

> **Q: Why not just store whole documents?**
> A: Embedding quality degrades on long text, and retrieval precision drops because entire documents match many queries weakly. Chunking gives us precise, targeted retrieval.

> **Q: How do you choose chunk size?**
> A: It's empirical, but the standard is 200-500 tokens. Smaller chunks for factoid QA (where you need precise answers), larger chunks for summarization tasks (where you need broader context). Our default is ~250 tokens.

> **Q: Why sentence-aware chunking over fixed-size?**
> A: Fixed-size chunking can break mid-sentence, creating semantically incomplete fragments like "He was born in". Sentence-aware chunking preserves semantic boundaries.

> **Q: What's the purpose of chunk overlap?**
> A: Information near chunk boundaries would be split across two chunks and potentially lost during retrieval. Overlap ensures at least one chunk captures the complete context around the boundary.

---

**Next**: [Chapter 3 — Embeddings & Vector Spaces →](03_embeddings.md)
