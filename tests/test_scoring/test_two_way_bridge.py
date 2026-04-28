# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two-way Scorer ↔ Metric bridge.

Closes the loop with :func:`metric_as_scorer` (already exists). The reverse
direction :func:`scorer_as_metric` lets any function-style scorer be used
where the :class:`Metric` Protocol is expected — unblocks NMP's evaluator
service from running NEL-native function scorers without rewrites.
"""

from __future__ import annotations

import math
from typing import Literal

import pytest

from nemo_evaluator.scoring.contracts import (
    Metric,
    MetricInput,
    MetricResult,
    MetricScore,
    TemplateMetric,
    metric_as_scorer,
    scorer_as_metric,
)


# ---------------------------------------------------------------------------
# Basic wrapping — sync scorer
# ---------------------------------------------------------------------------


def _correct_scorer(input: MetricInput) -> dict:
    """Sync scorer returning {'correct': bool}."""
    return {"correct": str(input.response).strip() == str(input.target).strip()}


def test_wrapped_sync_scorer_satisfies_metric_protocol():
    m = scorer_as_metric(_correct_scorer, name="exact", score_names=["correct"])
    assert isinstance(m, Metric)
    assert m.type == "exact"
    assert m.score_names() == ["correct"]


async def test_wrapped_sync_scorer_compute_scores_positive():
    m = scorer_as_metric(_correct_scorer, name="exact", score_names=["correct"])
    result = await m.compute_scores(MetricInput(response="x", target="x"))
    assert isinstance(result, MetricResult)
    assert len(result.scores) == 1
    assert result.scores[0].name == "correct"
    assert result.scores[0].value == 1.0


async def test_wrapped_sync_scorer_compute_scores_negative():
    m = scorer_as_metric(_correct_scorer, name="exact", score_names=["correct"])
    result = await m.compute_scores(MetricInput(response="x", target="y"))
    assert result.scores[0].value == 0.0


# ---------------------------------------------------------------------------
# Async scorer
# ---------------------------------------------------------------------------


async def _async_scorer(input: MetricInput) -> dict:
    """Async scorer returning {'correct': bool, 'latency': float}."""
    return {"correct": True, "latency": 0.123}


async def test_wrapped_async_scorer_compute_scores():
    """An async function scorer is awaited inside compute_scores."""
    m = scorer_as_metric(_async_scorer, name="async_exact", score_names=["correct"])
    result = await m.compute_scores(MetricInput(response="x", target="x"))
    assert result.scores[0].value == 1.0


async def test_wrapped_async_scorer_multi_score():
    m = scorer_as_metric(
        _async_scorer, name="async_multi", score_names=["correct", "latency"]
    )
    result = await m.compute_scores(MetricInput(response="x", target="x"))
    names = [s.name for s in result.scores]
    values = [s.value for s in result.scores]
    assert names == ["correct", "latency"]
    assert values == [1.0, 0.123]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_empty_score_names_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        scorer_as_metric(_correct_scorer, name="x", score_names=[])


async def test_missing_key_raises_keyerror():
    """Scorer must emit every key in score_names."""

    def _partial(input: MetricInput) -> dict:
        return {"correct": True}  # missing "latency"

    m = scorer_as_metric(_partial, name="partial", score_names=["correct", "latency"])
    with pytest.raises(KeyError, match="did not emit 'latency'"):
        await m.compute_scores(MetricInput(response="x", target="x"))


async def test_non_numeric_value_raises_typeerror():
    """Values must be castable to float."""

    def _str_value(input: MetricInput) -> dict:
        return {"correct": "not a number"}

    m = scorer_as_metric(_str_value, name="bad", score_names=["correct"])
    with pytest.raises(TypeError, match="non-numeric value"):
        await m.compute_scores(MetricInput(response="x", target="x"))


async def test_non_dict_return_raises_typeerror():
    """Scorer must return a dict, not e.g. a tuple."""

    def _wrong_return(input: MetricInput) -> tuple:
        return (1.0,)

    m = scorer_as_metric(_wrong_return, name="bad", score_names=["correct"])
    with pytest.raises(TypeError, match="returned tuple, expected dict"):
        await m.compute_scores(MetricInput(response="x", target="x"))


# ---------------------------------------------------------------------------
# Round-trip property: Metric → Scorer → Metric preserves scores
# ---------------------------------------------------------------------------


class _Tiny(TemplateMetric):
    type: Literal["tiny"] = "tiny"

    def _score(self, input: MetricInput) -> float:
        return 1.0 if input.response == input.target else 0.0


async def test_round_trip_preserves_scores():
    """Wrap a Metric → Scorer → Metric again. Scores match the original."""
    original = _Tiny()
    bridged = scorer_as_metric(
        metric_as_scorer(original), name="tiny", score_names=["tiny"]
    )

    inp_pos = MetricInput(response="hi", target="hi")
    inp_neg = MetricInput(response="hi", target="bye")

    orig_pos = await original.compute_scores(inp_pos)
    bridged_pos = await bridged.compute_scores(inp_pos)
    assert orig_pos.scores[0].value == bridged_pos.scores[0].value == 1.0

    orig_neg = await original.compute_scores(inp_neg)
    bridged_neg = await bridged.compute_scores(inp_neg)
    assert orig_neg.scores[0].value == bridged_neg.scores[0].value == 0.0


# ---------------------------------------------------------------------------
# Real NEL function scorers wrap correctly
# ---------------------------------------------------------------------------


def test_nel_native_exact_match_wraps_as_metric():
    """NEL's `exact_match` function scorer satisfies Metric after wrapping."""
    from nemo_evaluator.scoring.text import exact_match

    m = scorer_as_metric(exact_match, name="exact_match", score_names=["correct"])
    assert isinstance(m, Metric)
    assert m.type == "exact_match"


async def test_nel_native_exact_match_compute_scores():
    from nemo_evaluator.scoring.text import exact_match

    m = scorer_as_metric(exact_match, name="exact_match", score_names=["correct"])
    result = await m.compute_scores(MetricInput(response="hello", target="hello"))
    assert result.scores[0].value == 1.0

    result_neg = await m.compute_scores(MetricInput(response="hello", target="goodbye"))
    assert result_neg.scores[0].value == 0.0


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_wrapped_class_preserves_fn_docstring():
    def _documented(input: MetricInput) -> dict:
        """A scorer with a docstring."""
        return {"correct": True}

    m = scorer_as_metric(_documented, name="doc", score_names=["correct"])
    assert "scorer with a docstring" in (m.__doc__ or "")


def test_wrapped_class_name_includes_fn_name():
    def my_special_scorer(input: MetricInput) -> dict:
        return {"correct": True}

    m = scorer_as_metric(my_special_scorer, name="x", score_names=["correct"])
    assert "my_special_scorer" in type(m).__name__


# ---------------------------------------------------------------------------
# Re-export from scoring package
# ---------------------------------------------------------------------------


def test_scorer_as_metric_exported_from_package():
    from nemo_evaluator.scoring import scorer_as_metric as imported
    assert imported is scorer_as_metric
