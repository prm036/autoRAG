"""
Document parser — loads documents from disk into a unified Document model.

Supports plain text (.txt), Markdown (.md), and PDF (.pdf) formats.
Each document is assigned a unique ID and metadata for downstream traceability.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.models import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Loads documents from files or directories into Document objects.

    Usage:
        parser = DocumentParser(supported_formats=[".txt", ".pdf", ".md"])
        docs = parser.parse_directory("data/raw/")
        doc = parser.parse_file("data/raw/example.txt")
    """

    def __init__(self, supported_formats: list[str] | None = None):
        self.supported_formats = supported_formats or [".txt", ".pdf", ".md"]

    def parse_file(self, file_path: str | Path) -> Document:
        """Parse a single file into a Document."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(
                f"Unsupported format '{suffix}'. Supported: {self.supported_formats}"
            )

        logger.info(f"Parsing document: {file_path}")

        if suffix == ".pdf":
            text = self._parse_pdf(file_path)
        elif suffix in (".txt", ".md"):
            text = self._parse_text(file_path)
        else:
            raise ValueError(f"No parser implemented for format: {suffix}")

        metadata = DocumentMetadata(
            title=file_path.stem,
            source=str(file_path),
            format=suffix.lstrip("."),
        )

        return Document(metadata=metadata, text=text)

    def parse_directory(self, dir_path: str | Path) -> list[Document]:
        """
        Parse all supported files in a directory (non-recursive).

        Returns a list of Document objects, one per file.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        documents = []
        files = sorted(dir_path.iterdir())

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.parse_file(file_path)
                    documents.append(doc)
                    logger.info(
                        f"  Parsed '{file_path.name}': {len(doc.text)} chars"
                    )
                except Exception as e:
                    logger.warning(f"  Failed to parse '{file_path.name}': {e}")

        logger.info(f"Parsed {len(documents)} documents from {dir_path}")
        return documents

    # ---- Format-specific parsers ----

    def _parse_text(self, file_path: Path) -> str:
        """Parse a plain text or markdown file."""
        return file_path.read_text(encoding="utf-8")

    def _parse_pdf(self, file_path: Path) -> str:
        """
        Parse a PDF file using PyPDF2.

        Falls back to a warning if PyPDF2 is not installed.
        """
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. "
                "Install it with: pip install PyPDF2"
            )

        reader = PdfReader(str(file_path))
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Clean up common PDF extraction artifacts
                import re
                cleaned = re.sub(r"\s*\n\s*", " ", text)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                pages.append(cleaned)

        return "\n\n".join(pages)


if __name__ == "__main__":
    # Quick smoke test
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.parser <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    parser = DocumentParser()

    if target.is_dir():
        docs = parser.parse_directory(target)
    else:
        docs = [parser.parse_file(target)]

    for doc in docs:
        print(f"\n--- {doc.metadata.title} ({doc.metadata.format}) ---")
        print(f"    Doc ID:  {doc.metadata.doc_id}")
        print(f"    Source:  {doc.metadata.source}")
        print(f"    Length:  {len(doc.text)} chars")
        print(f"    Preview: {doc.text[:200]}...")
