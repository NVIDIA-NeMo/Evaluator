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
"""Scoring: all scorer implementations and utilities.

Scorers are functions ``(ScorerInput) -> dict`` used by benchmarks.
Look up scorers by name via :func:`get_scorer`.
"""

from __future__ import annotations

from typing import Callable

from nemo_evaluator.scoring.contracts import (
    CorpusMetric,
    CorpusTemplateMetric,
    Metric,
    MetricInput,
    MetricOutput,
    MetricResult,
    MetricScore,
    MetricWithSecrets,
    Scorer,
    SecretRefLike,
    SecretResolver,
    SecretsMixin,
    TemplateMetric,
    get_metric,
    list_metrics,
    metric_as_scorer,
    register_metric,
    scorer_as_metric,
)
from nemo_evaluator.scoring.judge import (
    JudgeScoringConfig,
    build_judge_prompt,
    judge_score,
    needs_judge,
    parse_judge_response,
)
from nemo_evaluator.scoring.json_schema import extract_json, validate_json_schema
from nemo_evaluator.scoring.pattern import answer_line, multichoice_regex, numeric_match
from nemo_evaluator.scoring.code_execution import code_sandbox, code_sandbox_async
from nemo_evaluator.scoring.text import exact_match, extract_mcq_letter, fuzzy_match
from nemo_evaluator.scoring.types import ScorerInput

_SCORER_REGISTRY: dict[str, Callable[[ScorerInput], dict]] = {
    "exact_match": exact_match,
    "fuzzy_match": fuzzy_match,
    "multichoice_regex": multichoice_regex,
    "mcq_extract": lambda s: _mcq_extract(s),
    "answer_line": answer_line,
    "numeric_match": numeric_match,
}


def _mcq_extract(sample: ScorerInput) -> dict:
    extracted = extract_mcq_letter(sample.response)
    correct = extracted.upper() == str(sample.target).strip().upper()
    return {"correct": correct, "extracted": extracted}


def get_scorer(name: str) -> Callable[[ScorerInput], dict]:
    """Look up a scorer by name. Raises KeyError if unknown."""
    if name not in _SCORER_REGISTRY:
        raise KeyError(f"Unknown scorer {name!r}. Available: {sorted(_SCORER_REGISTRY)}")
    return _SCORER_REGISTRY[name]


def list_scorers() -> list[str]:
    """Return sorted list of registered scorer names."""
    return sorted(_SCORER_REGISTRY)


__all__ = [
    "ScorerInput",
    "get_scorer",
    "list_scorers",
    # Text
    "exact_match",
    "fuzzy_match",
    "extract_mcq_letter",
    # Pattern
    "multichoice_regex",
    "answer_line",
    "numeric_match",
    # Sandbox
    "code_sandbox",
    "code_sandbox_async",
    # Judge
    "needs_judge",
    "JudgeScoringConfig",
    "build_judge_prompt",
    "parse_judge_response",
    "judge_score",
    # JSON
    "extract_json",
    "validate_json_schema",
]
