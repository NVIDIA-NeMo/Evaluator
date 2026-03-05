r"""Answer extraction from model responses.

Supports:
  - \boxed{...} (LaTeX)
  - "The answer is ..." pattern
  - Last numeric line
  - Final line fallback
"""

from __future__ import annotations

import re


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANSWER_IS_RE = re.compile(r"(?:the\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^[\s]*(-?[\d,]+\.?\d*)\s*$")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic wrapping from text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    return text.strip()


def extract_answer(text: str) -> str:
    r"""Extract a concise answer from model output.

    Priority: \boxed{} > "the answer is" > last numeric line > last non-empty line.
    Markdown formatting (bold, italic) is stripped before extraction.
    """
    text = _strip_markdown(text)

    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    answer_match = _ANSWER_IS_RE.search(text)
    if answer_match:
        return answer_match.group(1).strip()

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if _NUMERIC_RE.match(line):
            return line.replace(",", "").strip()

    return lines[-1] if lines else ""
