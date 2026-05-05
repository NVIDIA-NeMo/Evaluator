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
"""Text-based scorers: exact and fuzzy string matching, MCQ extraction."""

from __future__ import annotations

import re
import string
import unicodedata

from nemo_evaluator.scoring.types import ScorerInput

_MCQ_LETTER = re.compile(
    r"(?:"
    r"\\boxed\{([A-Za-z])\}"
    r"|(?:answer|ans)[\s:]*([A-Za-z])\b"
    r"|\(([A-Za-z])\)"
    r"|^([A-Za-z])[\.\)\s]*$"
    r")",
    re.IGNORECASE,
)


def extract_mcq_letter(text: str) -> str:
    """Extract a single option letter from common chat-model MCQ formats.

    Handles ``\\boxed{B}``, ``Answer: B``, ``(B)``, and bare ``B``.
    Prefers the last match so that explanations preceding the final
    answer don't interfere.  Returns the uppercase letter if found,
    otherwise the stripped input.
    """
    text = text.strip()
    matches = list(_MCQ_LETTER.finditer(text))
    if matches:
        m = matches[-1]
        return next(g for g in m.groups() if g is not None).upper()
    return text


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
