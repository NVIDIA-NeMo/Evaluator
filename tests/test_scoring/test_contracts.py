# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_evaluator.scoring.contracts.

The contracts module is the Approach-3 integration boundary between NEL
and NMP's nemo_evaluator_sdk. These tests verify the Protocols are
runtime-checkable and that SDK-style concrete classes satisfy them.
"""

from __future__ import annotations

import math

import pytest

from nemo_evaluator.scoring.contracts import (
    CorpusMetric,
    Metric,
    MetricResult,
    MetricScore,
    MetricWithPreflight,
    MetricWithSecrets,
    ScoreStats,
    metric_as_scorer,
)
from nemo_evaluator.scoring.types import ScorerInput


# -----------------------------------------------------------------------------
# Pydantic result types
# -----------------------------------------------------------------------------


def test_metric_score_basic():
    s = MetricScore(name="bleu", value=0.42)
    assert s.name == "bleu"
    assert s.value == pytest.approx(0.42)
    assert s.stats is None


def test_metric_score_nan_serialization():
    """NaN values round-trip as the string 'NaN' for JSON portability."""
    s = MetricScore(name="bleu", value=float("nan"))
    dumped = s.model_dump()
    assert dumped["value"] == "NaN"

    restored = MetricScore.model_validate(dumped)
    assert math.isnan(restored.value)


def test_metric_score_rejects_non_nan_strings():
    with pytest.raises(ValueError, match="NaN"):
        MetricScore(name="bleu", value="not-a-number")


def test_metric_result_holds_multiple_scores():
    r = MetricResult(scores=[
        MetricScore(name="bleu", value=0.3),
        MetricScore(name="bleu-1", value=0.5),
    ])
    assert len(r.scores) == 2


def test_score_stats_serializes_nan():
    stats = ScoreStats(mean=float("nan"), stddev=0.1, count=10)
    dumped = stats.model_dump()
    assert dumped["mean"] == "NaN"
    assert dumped["stddev"] == pytest.approx(0.1)
    assert dumped["count"] == 10


# -----------------------------------------------------------------------------
# Protocol satisfaction (SDK-style concrete class)
# -----------------------------------------------------------------------------


class _DuckMetric:
    """Fake metric class with SDK-style surface. No base class."""

    type = "duck"

    def metric(self, item: dict, sample: dict, trace=None) -> float:
        return 1.0 if item.get("reference") == sample.get("output_text") else 0.0

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        return MetricResult(scores=[MetricScore(name="duck", value=self.metric(item, sample))])

    def score_names(self) -> list[str]:
        return ["duck"]


class _DuckCorpus(_DuckMetric):
    async def compute_corpus_scores(self, items, samples):
        return MetricResult(scores=[MetricScore(name="duck_corpus", value=0.5)])


class _DuckSecrets(_DuckMetric):
    def secrets(self) -> dict:
        return {"API_KEY": _SecretRef(root="API_KEY")}

    async def resolve_secrets(self, resolver) -> None:
        pass


class _DuckPreflight(_DuckMetric):
    async def preflight(self) -> None:
        pass


class _SecretRef:
    def __init__(self, root: str):
        self._root = root

    @property
    def root(self) -> str:
        return self._root


def test_duck_class_satisfies_metric_protocol():
    """An SDK-style class satisfies the Metric Protocol structurally (no inheritance)."""
    assert isinstance(_DuckMetric(), Metric)


def test_corpus_metric_protocol():
    assert isinstance(_DuckCorpus(), CorpusMetric)


def test_metric_with_secrets_protocol():
    assert isinstance(_DuckSecrets(), MetricWithSecrets)


def test_metric_with_preflight_protocol():
    assert isinstance(_DuckPreflight(), MetricWithPreflight)


def test_plain_class_not_satisfying_protocol():
    class NotAMetric:
        pass

    assert not isinstance(NotAMetric(), Metric)


# -----------------------------------------------------------------------------
# Bridge: metric_as_scorer
# -----------------------------------------------------------------------------


def test_metric_as_scorer_positive_match():
    scorer = metric_as_scorer(_DuckMetric())
    result = scorer(ScorerInput(response="hello", target="hello"))

    assert isinstance(result, dict)
    assert result["score"] == 1.0
    assert result["metric_type"] == "duck"


def test_metric_as_scorer_negative_match():
    scorer = metric_as_scorer(_DuckMetric())
    result = scorer(ScorerInput(response="hello", target="goodbye"))

    assert result["score"] == 0.0


def test_metric_as_scorer_preserves_metadata():
    """Metadata from ScorerInput flows into item dict."""

    class _InspectMetric(_DuckMetric):
        captured_item: dict = {}

        def metric(self, item, sample, trace=None):
            _InspectMetric.captured_item = item
            return 0.0

    m = _InspectMetric()
    scorer = metric_as_scorer(m)
    scorer(ScorerInput(response="x", target="y", metadata={"problem_id": 7}))

    assert _InspectMetric.captured_item["reference"] == "y"
    assert _InspectMetric.captured_item["problem_id"] == 7


# -----------------------------------------------------------------------------
# Re-export from scoring package
# -----------------------------------------------------------------------------


def test_contracts_available_from_scoring_package():
    from nemo_evaluator.scoring import Metric as ImportedMetric
    from nemo_evaluator.scoring import MetricResult as ImportedResult
    from nemo_evaluator.scoring import metric_as_scorer as imported_bridge

    assert ImportedMetric is Metric
    assert ImportedResult is MetricResult
    assert imported_bridge is metric_as_scorer
