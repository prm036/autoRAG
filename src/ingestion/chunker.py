"""
Document chunker — splits cleaned documents into retrievable chunks.

Implements two strategies:
1. **Fixed-size chunking**: splits by character count with configurable overlap.
2. **Sentence-aware chunking** (default): splits on sentence boundaries, grouping
   sentences until the chunk reaches the target size, with overlap.

Each chunk carries full provenance: chunk_id, doc_id, position index, and
character offsets so answers can be traced back to exact source locations.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Literal

from src.models import Chunk, Document

logger = logging.getLogger(__name__)

# Pre-compiled regex for sentence boundary detection.
# Handles: period/question/exclamation followed by whitespace and uppercase letter.
# Preserves abbreviations like "Dr.", "U.S." to some degree.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)


class Chunker:
    """
    Splits documents into chunks for indexing and retrieval.

    Args:
        strategy: "fixed" for character-based splitting, "sentence" for
                  sentence-boundary-aware splitting.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        min_chunk_size: Discard chunks shorter than this.

    Usage:
        chunker = Chunker(strategy="sentence", chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk_document(document)
        all_chunks = chunker.chunk_documents(documents)
    """

    def __init__(
        self,
        strategy: Literal["fixed", "sentence"] = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a single document into chunks.

        Returns a list of Chunk objects with full provenance metadata.
        """
        text = document.text
        if not text or not text.strip():
            logger.warning(
                f"Empty document: {document.metadata.doc_id} "
                f"({document.metadata.title})"
            )
            return []

        if self.strategy == "sentence":
            raw_chunks = self._chunk_by_sentences(text)
        elif self.strategy == "fixed":
            raw_chunks = self._chunk_fixed_size(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        # Build Chunk objects with provenance
        chunks = []
        for i, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=document.metadata.doc_id,
                text=chunk_text.strip(),
                position=i,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "title": document.metadata.title,
                    "source": document.metadata.source,
                    "strategy": self.strategy,
                    "chunk_size_config": self.chunk_size,
                    "overlap_config": self.chunk_overlap,
                },
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked '{document.metadata.title}': "
            f"{len(text)} chars → {len(chunks)} chunks "
            f"(strategy={self.strategy}, size={self.chunk_size}, "
            f"overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents and return all chunks in a flat list."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(
            f"Total: {len(documents)} documents → {len(all_chunks)} chunks"
        )
        return all_chunks

    # ---- Chunking strategies ----

    def _chunk_fixed_size(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text into fixed-size chunks with overlap.

        Returns: list of (chunk_text, start_char, end_char) tuples.
        """
        chunks = []
        start = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))

            if end >= text_len:
                break
            start += step

        return chunks

    def _chunk_by_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text on sentence boundaries, grouping sentences until the
        target chunk size is reached. Overlap is achieved by repeating
        trailing sentences from the previous chunk.

        Returns: list of (chunk_text, start_char, end_char) tuples.
        """
        # Split into sentences
        sentences = _SENTENCE_SPLIT_RE.split(text)
        if not sentences:
            return [(text, 0, len(text))]

        # Track character positions of each sentence
        sentence_positions: list[tuple[str, int, int]] = []
        pos = 0
        for sent in sentences:
            start = text.find(sent, pos)
            if start == -1:
                start = pos
            end = start + len(sent)
            sentence_positions.append((sent, start, end))
            pos = end

        # Group sentences into chunks
        chunks: list[tuple[str, int, int]] = []
        current_sentences: list[int] = []  # indices into sentence_positions
        current_length = 0

        for idx, (sent, s_start, s_end) in enumerate(sentence_positions):
            sent_len = len(sent)

            # If adding this sentence exceeds chunk_size and we have content,
            # finalize the current chunk
            if current_length + sent_len > self.chunk_size and current_sentences:
                chunk_start = sentence_positions[current_sentences[0]][1]
                chunk_end = sentence_positions[current_sentences[-1]][2]
                chunk_text = text[chunk_start:chunk_end]
                chunks.append((chunk_text, chunk_start, chunk_end))

                # Compute overlap: keep trailing sentences that fit in overlap budget
                overlap_sentences = []
                overlap_length = 0
                for prev_idx in reversed(current_sentences):
                    prev_len = len(sentence_positions[prev_idx][0])
                    if overlap_length + prev_len > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, prev_idx)
                    overlap_length += prev_len

                current_sentences = overlap_sentences
                current_length = overlap_length

            current_sentences.append(idx)
            current_length += sent_len

        # Don't forget the last chunk
        if current_sentences:
            chunk_start = sentence_positions[current_sentences[0]][1]
            chunk_end = sentence_positions[current_sentences[-1]][2]
            chunk_text = text[chunk_start:chunk_end]
            chunks.append((chunk_text, chunk_start, chunk_end))

        return chunks


if __name__ == "__main__":
    # Quick demo with sample text
    import sys

    logging.basicConfig(level=logging.INFO)

    sample_text = (
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with language model generation. The system first "
        "retrieves relevant documents from a knowledge base. Then it uses these "
        "documents as context for the language model. This approach helps reduce "
        "hallucinations and grounds the model's responses in factual evidence. "
        "BM25 is a classical sparse retrieval method based on term frequency. "
        "Dense retrieval uses learned vector representations to capture semantic "
        "similarity between queries and documents. Hybrid retrieval combines both "
        "approaches using techniques like Reciprocal Rank Fusion. Cross-encoder "
        "reranking further improves the quality of retrieved passages by scoring "
        "each query-document pair with a fine-tuned transformer model."
    )

    from src.models import Document, DocumentMetadata

    doc = Document(
        metadata=DocumentMetadata(title="RAG Overview", source="demo"),
        text=sample_text,
    )

    for strategy in ["fixed", "sentence"]:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy}")
        print(f"{'=' * 60}")

        chunker = Chunker(
            strategy=strategy,
            chunk_size=256,
            chunk_overlap=50,
            min_chunk_size=50,
        )
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            print(f"\n  Chunk {chunk.position} [{chunk.start_char}:{chunk.end_char}] "
                  f"({len(chunk.text)} chars):")
            print(f"  {chunk.text[:120]}...")
