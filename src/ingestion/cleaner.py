"""
Text cleaner — normalizes raw document text before chunking.

Handles whitespace normalization, encoding issues, and common artifacts
from PDF extraction and web scraping. The cleaner is intentionally conservative:
it fixes formatting issues without altering content.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and normalizes raw document text.

    Usage:
        cleaner = TextCleaner()
        cleaned = cleaner.clean(raw_text)
    """

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to a text string.

        Steps (in order):
        1. Unicode normalization (NFC)
        2. Replace common typographic characters
        3. Normalize whitespace
        4. Remove control characters
        5. Collapse excessive blank lines
        6. Strip leading/trailing whitespace
        """
        if not text:
            return ""

        text = self._normalize_unicode(text)
        text = self._replace_typographic_chars(text)
        text = self._normalize_whitespace(text)
        text = self._remove_control_chars(text)
        text = self._collapse_blank_lines(text)
        text = text.strip()

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize to NFC form (composed characters)."""
        return unicodedata.normalize("NFC", text)

    def _replace_typographic_chars(self, text: str) -> str:
        """Replace common typographic characters with standard equivalents."""
        replacements = {
            "\u2018": "'",   # left single quote
            "\u2019": "'",   # right single quote
            "\u201c": '"',   # left double quote
            "\u201d": '"',   # right double quote
            "\u2013": "-",   # en dash
            "\u2014": "-",   # em dash
            "\u2026": "...", # ellipsis
            "\u00a0": " ",   # non-breaking space
            "\u200b": "",    # zero-width space
            "\ufeff": "",    # BOM
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace while preserving paragraph structure.

        - Replaces tabs with spaces
        - Collapses multiple spaces into one (within lines)
        - Preserves single newlines (paragraph boundaries)
        """
        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Collapse multiple spaces within lines (not newlines)
        text = re.sub(r"[^\S\n]+", " ", text)

        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove non-printable control characters (except newline and tab)."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def _collapse_blank_lines(self, text: str) -> str:
        """Collapse 3+ consecutive blank lines into 2 (preserve paragraph breaks)."""
        return re.sub(r"\n{3,}", "\n\n", text)


if __name__ == "__main__":
    # Quick demo
    sample = (
        "  This is   a\u00a0test   document.\n\n\n\n\n"
        "It has \u201csmart quotes\u201d and an em\u2014dash.\n"
        "Also\ttabs\tand   extra    spaces.\n"
    )
    cleaner = TextCleaner()
    cleaned = cleaner.clean(sample)
    print("=== Original ===")
    print(repr(sample))
    print("\n=== Cleaned ===")
    print(repr(cleaned))
    print("\n=== Readable ===")
    print(cleaned)
