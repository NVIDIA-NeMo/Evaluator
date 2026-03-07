"""Text-based scorers: exact and fuzzy string matching."""
from __future__ import annotations

import re
import string
import unicodedata

from nemo_evaluator.scoring.types import ScorerInput


def exact_match(sample: ScorerInput) -> dict:
    return {"correct": _normalize(sample.response) == _normalize(str(sample.target))}


def fuzzy_match(sample: ScorerInput) -> dict:
    pred = _normalize(sample.response)
    targets = sample.metadata.get("correct_answers", [str(sample.target)])
    correct = any(_normalize(t) in pred or pred in _normalize(t) for t in targets)
    return {"correct": correct, "extracted": sample.response.strip()[:200]}


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())
