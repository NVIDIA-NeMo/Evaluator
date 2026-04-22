# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo_evaluator.scoring.contracts``.

Covers the metric-abstractions reshape:

- Unified :data:`MetricInput` (alias of :class:`ScorerInput`)
- :class:`Metric` Protocol (without the dropped ``metric()`` method)
- :class:`TemplateMetric` base class for ~20-30 LOC user metrics
- :func:`register_metric` / :func:`get_metric` / :func:`list_metrics`
- :func:`metric_as_scorer` bridge
"""

from __future__ import annotations

import math
from typing import Literal

import pytest

from nemo_evaluator.scoring.contracts import (
    CorpusMetric,
    Metric,
    MetricInput,
    MetricOutput,
    MetricResult,
    MetricScore,
    MetricWithSecrets,
    TemplateMetric,
    _METRIC_REGISTRY,
    get_metric,
    list_metrics,
    metric_as_scorer,
    register_metric,
)
from nemo_evaluator.scoring.types import ScorerInput


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test starts with an empty metric registry."""
    snapshot = dict(_METRIC_REGISTRY)
    _METRIC_REGISTRY.clear()
    yield
    _METRIC_REGISTRY.clear()
    _METRIC_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Input/output type aliases
# ---------------------------------------------------------------------------


def test_metric_input_is_scorer_input():
    """MetricInput is an alias for ScorerInput — same dataclass."""
    assert MetricInput is ScorerInput


def test_metric_output_is_metric_result():
    """MetricOutput is an alias for MetricResult."""
    assert MetricOutput is MetricResult


# ---------------------------------------------------------------------------
# Pydantic result types
# ---------------------------------------------------------------------------


def test_metric_score_basic():
    s = MetricScore(name="bleu", value=0.42)
    assert s.name == "bleu"
    assert s.value == pytest.approx(0.42)


def test_metric_score_nan_roundtrip():
    s = MetricScore(name="bleu", value=float("nan"))
    dumped = s.model_dump()
    assert dumped["value"] == "NaN"
    restored = MetricScore.model_validate(dumped)
    assert math.isnan(restored.value)


def test_metric_score_rejects_non_nan_strings():
    with pytest.raises(ValueError, match="NaN"):
        MetricScore(name="bleu", value="not-a-number")


# ---------------------------------------------------------------------------
# Protocol satisfaction — after dropping metric()
# ---------------------------------------------------------------------------


class _SdkShapedMetric:
    """Concrete class with the post-reshape Metric surface.

    Notably does NOT implement the old ``metric(item, sample, trace)`` method
    — it was dropped from the Protocol per Sandy Chapman's + Voytek Prazuch's
    feedback in steps.md.
    """

    type = "sdk-shaped"

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        value = 1.0 if input.response == input.target else 0.0
        return MetricResult(scores=[MetricScore(name="sdk-shaped", value=value)])

    def score_names(self) -> list[str]:
        return ["sdk-shaped"]


def test_sdk_shaped_satisfies_metric_protocol_without_legacy_metric_method():
    """A class without the dropped metric() method still satisfies the Protocol."""
    assert isinstance(_SdkShapedMetric(), Metric)


def test_plain_class_fails_metric_protocol():
    class NotAMetric:
        pass

    assert not isinstance(NotAMetric(), Metric)


class _CorpusShaped(_SdkShapedMetric):
    async def compute_corpus_scores(self, inputs):
        return MetricResult(scores=[MetricScore(name="corpus", value=0.5)])


class _SecretsShaped(_SdkShapedMetric):
    def secrets(self):
        return {"API_KEY": _FakeSecretRef("API_KEY")}

    async def resolve_secrets(self, resolver):
        pass


class _FakeSecretRef:
    def __init__(self, root: str):
        self._root = root

    @property
    def root(self) -> str:
        return self._root


def test_corpus_metric_protocol():
    assert isinstance(_CorpusShaped(), CorpusMetric)


def test_metric_with_secrets_protocol():
    assert isinstance(_SecretsShaped(), MetricWithSecrets)


# ---------------------------------------------------------------------------
# TemplateMetric — 20-30 LOC authoring pattern
# ---------------------------------------------------------------------------


class _TinyExactMatch(TemplateMetric):
    """Example of the target authoring shape — exact-match in <10 LOC."""

    type: Literal["exact-match"] = "exact-match"

    def _score(self, input: MetricInput) -> float:
        return 1.0 if str(input.response).strip() == str(input.target).strip() else 0.0


def test_template_metric_subclass_satisfies_metric_protocol():
    m = _TinyExactMatch()
    assert isinstance(m, Metric)
    assert m.type == "exact-match"
    assert m.score_names() == ["exact-match"]


async def test_template_metric_compute_scores_default_wrapper():
    m = _TinyExactMatch()
    out = await m.compute_scores(MetricInput(response="hello", target="hello"))
    assert len(out.scores) == 1
    assert out.scores[0].name == "exact-match"
    assert out.scores[0].value == 1.0


async def test_template_metric_compute_scores_negative_case():
    m = _TinyExactMatch()
    out = await m.compute_scores(MetricInput(response="hello", target="goodbye"))
    assert out.scores[0].value == 0.0


class _MultiScoreMetric(TemplateMetric):
    """Subclass overriding compute_scores for multi-score output."""

    type: Literal["multi"] = "multi"

    def _score(self, input: MetricInput) -> float:
        return 0.5

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        base = self._score(input)
        return MetricResult(scores=[
            MetricScore(name="a", value=base),
            MetricScore(name="b", value=base * 2),
        ])

    def score_names(self) -> list[str]:
        return ["a", "b"]


async def test_template_metric_compute_scores_override_for_multi():
    m = _MultiScoreMetric()
    out = await m.compute_scores(MetricInput(response="x", target="y"))
    assert [s.name for s in out.scores] == ["a", "b"]
    assert out.scores[0].value == 0.5
    assert out.scores[1].value == 1.0


# ---------------------------------------------------------------------------
# register_metric + get_metric
# ---------------------------------------------------------------------------


def test_register_metric_by_literal_default():
    @register_metric
    class FooMetric(TemplateMetric):
        type: Literal["foo"] = "foo"

        def _score(self, input: MetricInput) -> float:
            return 0.0

    assert "foo" in list_metrics()
    assert get_metric("foo") is FooMetric


def test_register_metric_by_class_var_attribute():
    @register_metric
    class BarMetric:
        type = "bar"  # plain class attribute

        async def compute_scores(self, input):
            return MetricResult(scores=[MetricScore(name="bar", value=0.0)])

        def score_names(self):
            return ["bar"]

    assert get_metric("bar") is BarMetric


def test_register_metric_without_type_raises():
    with pytest.raises(ValueError, match="type must be set"):
        @register_metric
        class Bad:
            pass


def test_get_metric_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown metric"):
        get_metric("does-not-exist")


def test_list_metrics_returns_sorted():
    @register_metric
    class Zeta(TemplateMetric):
        type: Literal["zeta"] = "zeta"

        def _score(self, input):
            return 0.0

    @register_metric
    class Alpha(TemplateMetric):
        type: Literal["alpha"] = "alpha"

        def _score(self, input):
            return 0.0

    assert list_metrics() == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# metric_as_scorer bridge
# ---------------------------------------------------------------------------


def test_metric_as_scorer_single_score():
    scorer = metric_as_scorer(_TinyExactMatch())
    result = scorer(MetricInput(response="x", target="x"))
    assert result["metric_type"] == "exact-match"
    assert result["exact-match"] == 1.0
    assert result["score"] == 1.0  # convenience for single-score case


def test_metric_as_scorer_negative():
    scorer = metric_as_scorer(_TinyExactMatch())
    result = scorer(MetricInput(response="a", target="b"))
    assert result["score"] == 0.0


def test_metric_as_scorer_multi_score_no_score_alias():
    """Multi-score metrics expose each score by name, no ambiguous 'score' key."""
    scorer = metric_as_scorer(_MultiScoreMetric())
    result = scorer(MetricInput(response="x", target="y"))
    assert result["a"] == 0.5
    assert result["b"] == 1.0
    assert "score" not in result  # ambiguous which score would win


# ---------------------------------------------------------------------------
# Re-exports from scoring package
# ---------------------------------------------------------------------------


def test_contracts_exported_from_scoring_package():
    from nemo_evaluator.scoring import (
        Metric as ImportedMetric,
        MetricInput as ImportedInput,
        MetricOutput as ImportedOutput,
        MetricResult as ImportedResult,
        TemplateMetric as ImportedBase,
        metric_as_scorer as imported_bridge,
        register_metric as imported_reg,
    )

    assert ImportedMetric is Metric
    assert ImportedInput is MetricInput
    assert ImportedOutput is MetricOutput
    assert ImportedResult is MetricResult
    assert ImportedBase is TemplateMetric
    assert imported_bridge is metric_as_scorer
    assert imported_reg is register_metric
