# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multiple-choice loglikelihood scorers (lm-evaluation-harness parity).

The :func:`multiple_choice_acc` scorer expects a per-row payload populated
by an upstream solver (typically :class:`LogprobRankingSolver`) that already
ran the candidate continuations and computed sum-loglikelihoods. The payload
reaches the scorer via :attr:`ScorerInput.metadata` under three keys:

- ``_mc_choices``: list of candidate continuation strings.
- ``_mc_choices_logprobs``: list of per-choice sum-loglikelihoods.
- ``_mc_choices_is_greedy``: list of booleans, True iff every continuation
  token equals the top-1 logprob token at its position.

The companion :func:`mcq_letter_extract` scorer scores the text-mode case
where the model produced a free-form response and we extract a single
letter (A-J) before comparing to ground truth.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from nemo_evaluator.environments.custom import scorer
from nemo_evaluator.scoring.metric import MetricOutputSpec

if TYPE_CHECKING:
    from nemo_evaluator.scoring.types import ScorerInput


_INT_TO_LETTER = {i: chr(ord("A") + i) for i in range(10)}


# Patterns ordered most-specific first; the first match wins.
_MCQ_LETTER_PATTERNS = (
    re.compile(r"\\boxed\{\s*([A-Ja-j])\s*\}"),
    re.compile(r"(?:answer\s+is\s*[:\-]?\s*\(?)\s*([A-Ja-j])\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?([A-Ja-j])\)", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-Ja-j])[\)\.\:]", re.IGNORECASE),
    re.compile(r"\b([A-Ja-j])\b"),
)


def _extract_letter(response: str | None) -> str:
    """Best-effort extract a single A-J letter from a free-form response."""
    text = (response or "").strip()
    if not text:
        return ""
    if text[0].upper() in "ABCDEFGHIJ" and (len(text) == 1 or not text[1].isalpha()):
        return text[0].upper()
    for pat in _MCQ_LETTER_PATTERNS:
        m = pat.search(text[:200])
        if m:
            return m.group(1).upper()
    return ""


@scorer(
    metric_type="multiple_choice_acc",
    outputs=[
        MetricOutputSpec.continuous_score(
            "acc",
            description="1.0 iff argmax of raw choice loglikelihoods matches gold (canonical MMLU metric).",
        ),
        MetricOutputSpec.continuous_score(
            "acc_norm",
            description="1.0 iff argmax of length-normalized loglikelihoods matches gold (ARC/BoolQ).",
        ),
        MetricOutputSpec.continuous_score(
            "acc_greedy",
            description="1.0 iff the highest-loglikelihood greedy-eligible choice matches gold.",
        ),
    ],
)
def multiple_choice_acc(sample: ScorerInput) -> dict:
    """Score multiple-choice loglikelihood ranking with lm-eval-harness metrics.

    Reads ``_mc_choices``, ``_mc_choices_logprobs``, ``_mc_choices_is_greedy``
    from :attr:`ScorerInput.metadata`. Returns ``{"acc": 0.0, "acc_norm":
    0.0, "acc_greedy": 0.0}`` when the payload is missing or malformed (e.g.,
    user wired the scorer to a benchmark that didn't go through the
    LogprobRankingSolver).

    Resolves the gold target heuristically — accepts integer index, single
    letter ``"A"..."Z"``, stringified integer, or verbatim choice text.
    Length-normalization uses character length to match lm-eval-harness's
    ``acc_norm``.
    """
    meta = sample.metadata or {}
    choices = meta.get("_mc_choices") or []
    logprobs = meta.get("_mc_choices_logprobs") or []
    is_greedy = meta.get("_mc_choices_is_greedy") or []

    zero = {"acc": 0.0, "acc_norm": 0.0, "acc_greedy": 0.0}
    if not choices or not logprobs or len(choices) != len(logprobs):
        return zero

    gold_idx = _resolve_gold_index(sample.target, choices)
    if gold_idx is None:
        return zero

    raw_argmax = max(range(len(logprobs)), key=lambda i: logprobs[i])

    norm_scores = [logprobs[i] / max(len(choices[i]), 1) for i in range(len(choices))]
    norm_argmax = max(range(len(norm_scores)), key=lambda i: norm_scores[i])

    greedy_argmax: int | None = None
    if is_greedy and any(is_greedy):
        greedy_indices = [i for i, g in enumerate(is_greedy) if g]
        if greedy_indices:
            greedy_argmax = max(greedy_indices, key=lambda i: logprobs[i])

    return {
        "acc": 1.0 if raw_argmax == gold_idx else 0.0,
        "acc_norm": 1.0 if norm_argmax == gold_idx else 0.0,
        "acc_greedy": 1.0 if greedy_argmax is not None and greedy_argmax == gold_idx else 0.0,
    }


@scorer(
    metric_type="mcq_letter_extract",
    outputs=[
        MetricOutputSpec.continuous_score(
            "correct",
            description="1.0 iff the letter extracted from the response matches the gold letter.",
        ),
        MetricOutputSpec.boolean(
            "parsed",
            description="True iff the response yielded a non-empty extracted letter.",
        ),
    ],
)
def mcq_letter_extract(sample: ScorerInput) -> dict:
    """Extract A-J from response and compare to target.

    Handles common response formats: ``"A"``, ``"A)"``, ``"The answer is B"``,
    ``"B. Because..."``, ``"(C)"``, ``"Option D"``, ``"\\boxed{E}"``.

    Targets may be a letter (``"A"..."J"``), an integer (``0..9``), the
    string form of an integer, or verbatim choice text. When the target is
    verbatim text and the row has letter-coded choice columns (``a``/``b``/
    ``c``/``d`` in ``ScorerInput.metadata``), it is matched against those.
    """
    predicted = _extract_letter(sample.response)

    raw = sample.target
    target_letter = ""
    if isinstance(raw, bool):
        raw = int(raw)
    if isinstance(raw, int):
        target_letter = _INT_TO_LETTER.get(raw, "")
    elif isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            target_letter = _INT_TO_LETTER.get(int(s), s.upper())
        elif s and s[0].upper() in "ABCDEFGHIJ" and len(s) <= 2:
            target_letter = s[0].upper()
        else:
            for letter, key in zip("ABCD", ("a", "b", "c", "d")):
                if (sample.metadata or {}).get(key, "").strip().lower() == s.lower():
                    target_letter = letter
                    break

    return {
        "correct": 1.0 if predicted and predicted == target_letter else 0.0,
        "parsed": bool(predicted),
    }


def _resolve_gold_index(target: object, choices: list[str]) -> int | None:
    """Map a heterogeneous target value to an index into choices."""
    if target is None:
        return None
    if isinstance(target, bool):
        i = int(target)
        return i if i < len(choices) else None
    if isinstance(target, int):
        return target if 0 <= target < len(choices) else None
    if isinstance(target, str):
        s = target.strip()
        if not s:
            return None
        if len(s) == 1 and s.upper().isalpha():
            idx = ord(s.upper()) - ord("A")
            if 0 <= idx < len(choices):
                return idx
        if s.lstrip("-").isdigit():
            try:
                idx = int(s)
                if 0 <= idx < len(choices):
                    return idx
            except ValueError:
                pass
        for i, c in enumerate(choices):
            if c.strip() == s:
                return i
        s_low = s.lower()
        for i, c in enumerate(choices):
            if c.strip().lower() == s_low:
                return i
    return None


__all__ = ["multiple_choice_acc", "mcq_letter_extract"]
