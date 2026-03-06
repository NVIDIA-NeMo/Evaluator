"""Multiple-choice question answer extraction and scoring."""
from __future__ import annotations

import re

_LETTER_RE = re.compile(r"(?:^|\s)\(?([A-J])\)?(?:\s|[.,;:]|$)")
_ANSWER_LETTER_RE = re.compile(
    r"(?:answer|choice|option)\s*(?:is\s*:?|:)\s*\(?([A-Ja-j])\)?",
    re.IGNORECASE,
)
_BOXED_LETTER_RE = re.compile(r"\\boxed\{([A-J])\}")


def extract_mcq_answer(text: str) -> str | None:
    """Extract a multiple-choice letter (A-J) from model output.

    Priority: \\boxed{} > "answer is X" pattern > last standalone letter.
    """
    boxed = _BOXED_LETTER_RE.findall(text)
    if boxed:
        return boxed[-1]

    answer_match = _ANSWER_LETTER_RE.findall(text)
    if answer_match:
        return answer_match[-1].upper()

    letters = _LETTER_RE.findall(text)
    if letters:
        return letters[-1]

    return None


def mcq_score(response: str, expected: str) -> bool:
    extracted = extract_mcq_answer(response)
    if extracted is None:
        return False
    return extracted.upper() == expected.strip().upper()
