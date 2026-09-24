"""
ChromaDB vector store — persistent vector index for dense retrieval.

Stores document chunk embeddings alongside metadata in ChromaDB for
efficient approximate nearest-neighbor search. Supports:
- Adding chunks with pre-computed embeddings
- Querying by embedding vector
- Persistence to disk
- Full metadata storage for traceability
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store for dense retrieval.

    Args:
        persist_directory: Path for persisting the database.
        collection_name: Name of the ChromaDB collection.
        similarity_metric: Distance metric — "cosine", "l2", or "ip".

    Usage:
        store = VectorStore(persist_directory="data/vector_store")
        store.add_chunks(chunks)  # chunks must have .embedding set
        results = store.search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "rag_chunks",
        similarity_metric: str = "cosine",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.similarity_metric = similarity_metric

        self._client = None
        self._collection = None

    @property
    def collection(self):
        """Lazy-initialize the ChromaDB client and collection."""
        if self._collection is None:
            import chromadb

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Initializing ChromaDB at '{self.persist_directory}' "
                f"(collection='{self.collection_name}', "
                f"metric='{self.similarity_metric}')"
            )

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.similarity_metric},
            )

            logger.info(
                f"Collection '{self.collection_name}' has "
                f"{self._collection.count()} vectors"
            )

        return self._collection

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """
        Add chunks with their embeddings to the vector store.

        Args:
            chunks: List of Chunk objects (must have .embedding set).
            batch_size: Number of chunks to insert per batch.

        Returns:
            Number of chunks added.
        """
        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            logger.warning("No chunks with embeddings to add")
            return 0

        start = time.time()

        # Insert in batches to avoid memory issues with large collections
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i : i + batch_size]

            ids = [c.chunk_id for c in batch]
            embeddings = [c.embedding for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "position": c.position,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "title": c.metadata.get("title", ""),
                    "source": c.metadata.get("source", ""),
                }
                for c in batch
            ]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"Added {len(valid_chunks)} chunks to vector store in {elapsed_ms:.0f}ms"
        )

        return len(valid_chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """
        Search for the most similar chunks to a query embedding.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of ScoredChunk objects, sorted by relevance (highest first).
        """
        start = time.time()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        elapsed_ms = (time.time() - start) * 1000

        scored_chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # ChromaDB returns distances; convert to similarity scores.
                # For cosine: similarity = 1 - distance
                if self.similarity_metric == "cosine":
                    score = 1.0 - distance
                elif self.similarity_metric == "l2":
                    score = 1.0 / (1.0 + distance)
                else:
                    score = -distance  # ip: higher is better, but chromadb negates

                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=metadata.get("doc_id", ""),
                    text=results["documents"][0][i] if results["documents"] else "",
                    position=metadata.get("position", 0),
                    start_char=metadata.get("start_char", 0),
                    end_char=metadata.get("end_char", 0),
                    metadata=metadata,
                )

                scored_chunks.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=score,
                        source="dense",
                        rank=i + 1,
                    )
                )

        logger.debug(
            f"Vector search returned {len(scored_chunks)} results in {elapsed_ms:.0f}ms"
        )

        return scored_chunks

    def count(self) -> int:
        """Return the number of vectors in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all vectors from the collection."""
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Cleared collection '{self.collection_name}'")

    def info(self) -> dict:
        """Return metadata about the vector store."""
        return {
            "provider": "chromadb",
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "similarity_metric": self.similarity_metric,
            "count": self.count() if self._collection else 0,
        }
