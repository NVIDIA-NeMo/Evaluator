# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NELScorerMixin -- SDK metrics callable as NEL scorers.

NELScorerMixin maps ScorerInput -> (item, sample) dicts the SDK metrics expect:
- item = {"reference": scorer_input.target, **metadata}
- sample = {"output_text": scorer_input.response, "response": scorer_input.response}

So SDK metrics configured with templates like ``reference="{{ reference }}"``
and using default candidate (reads sample.output_text) work out of the box
against a NEL ScorerInput.
"""

from nemo_evaluator.scoring.types import ScorerInput
from nemo_evaluator.sdk.scoring.exact_match import ExactMatchMetric


def _metric() -> ExactMatchMetric:
    return ExactMatchMetric(reference="{{ reference }}")


def test_sdk_exact_match_metric_as_nel_scorer():
    """ExactMatchMetric accepts ScorerInput and returns a dict."""
    result = _metric()(ScorerInput(response="hello", target="hello"))

    assert isinstance(result, dict)
    assert "score" in result
    assert "metric_type" in result
    assert result["score"] == 1.0


def test_sdk_exact_match_negative_case():
    """Non-matching response scores 0."""
    result = _metric()(ScorerInput(response="hello", target="goodbye"))
    assert result["score"] == 0.0


def test_sdk_metric_preserves_metadata_passthrough():
    """Metadata from ScorerInput flows into item dict."""
    result = _metric()(
        ScorerInput(
            response="answer",
            target="answer",
            metadata={"problem_id": 42},
        )
    )
    assert result["score"] == 1.0
