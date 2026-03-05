"""Normalized exact-match string comparison."""

from __future__ import annotations

import re
import string
import unicodedata


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def exact_match(predicted: str, expected: str) -> bool:
    """Case-insensitive, punctuation-normalized comparison."""
    return _normalize(predicted) == _normalize(expected)
