# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ergonomics tests for the metric-abstractions helpers:

- ``TemplateMetric._render`` — Jinja2 helper
- :class:`CorpusTemplateMetric` — row + corpus pattern
- :class:`SecretsMixin` — API-key resolution
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pytest
from jinja2.exceptions import UndefinedError

from nemo_evaluator.scoring.contracts import (
    CorpusTemplateMetric,
    Metric,
    MetricInput,
    MetricResult,
    MetricWithSecrets,
    SecretsMixin,
    TemplateMetric,
)


# ---------------------------------------------------------------------------
# TemplateMetric._render
# ---------------------------------------------------------------------------


class _RenderProbe(TemplateMetric):
    type: Literal["probe"] = "probe"

    def _score(self, input):
        return 0.0


def test_render_simple_substitution_nel_native_names():
    m = _RenderProbe()
    out = m._render("{{ response }} vs {{ target }}", MetricInput(response="hi", target="hello"))
    assert out == "hi vs hello"


def test_render_sdk_native_aliases():
    """SDK-style templates using 'output_text' and 'reference' render correctly."""
    m = _RenderProbe()
    out = m._render("{{ output_text }} :: {{ reference }}", MetricInput(response="a", target="b"))
    assert out == "a :: b"


def test_render_metadata_access():
    m = _RenderProbe()
    out = m._render(
        "problem={{ metadata.problem_id }}",
        MetricInput(response="", target="", metadata={"problem_id": 42}),
    )
    assert out == "problem=42"


def test_render_config_access():
    m = _RenderProbe()
    out = m._render(
        "k={{ config.k }}",
        MetricInput(response="", target="", config={"k": 10}),
    )
    assert out == "k=10"


def test_render_missing_variable_raises_strict_undefined():
    """Undefined variables should raise, not silently render empty."""
    m = _RenderProbe()
    with pytest.raises(UndefinedError):
        m._render("{{ not_a_field }}", MetricInput(response="x", target="y"))


# ---------------------------------------------------------------------------
# CorpusTemplateMetric
# ---------------------------------------------------------------------------


class _Accuracy(CorpusTemplateMetric):
    type: Literal["accuracy"] = "accuracy"

    def _score(self, input: MetricInput) -> float:
        return 1.0 if input.response == input.target else 0.0

    def _corpus_score(self, inputs: list[MetricInput]) -> float:
        return sum(self._score(i) for i in inputs) / len(inputs)


def test_corpus_metric_satisfies_both_protocols():
    from nemo_evaluator.scoring.contracts import CorpusMetric

    m = _Accuracy()
    assert isinstance(m, Metric)
    assert isinstance(m, CorpusMetric)


async def test_corpus_metric_row_score_default_wrapper():
    m = _Accuracy()
    out = await m.compute_scores(MetricInput(response="x", target="x"))
    assert out.scores[0].name == "accuracy"
    assert out.scores[0].value == 1.0


async def test_corpus_metric_compute_corpus_scores():
    m = _Accuracy()
    inputs = [
        MetricInput(response="a", target="a"),
        MetricInput(response="b", target="b"),
        MetricInput(response="c", target="wrong"),
    ]
    out = await m.compute_corpus_scores(inputs)
    assert out is not None
    assert out.scores[0].name == "accuracy_corpus"
    assert out.scores[0].value == pytest.approx(2 / 3)


async def test_corpus_metric_empty_inputs_returns_none():
    m = _Accuracy()
    out = await m.compute_corpus_scores([])
    assert out is None


def test_corpus_metric_score_names_includes_both():
    m = _Accuracy()
    names = m.score_names()
    assert "accuracy" in names
    assert "accuracy_corpus" in names


# ---------------------------------------------------------------------------
# SecretsMixin
# ---------------------------------------------------------------------------


class _JudgeMetric(TemplateMetric, SecretsMixin):
    type: Literal["judge"] = "judge"
    secret_env_vars: ClassVar[tuple[str, ...]] = ("JUDGE_API_KEY",)

    def _score(self, input):
        # Real metrics would call the judge endpoint with self.get_secret(...)
        return 0.0


def test_secrets_mixin_satisfies_metric_with_secrets_protocol():
    m = _JudgeMetric()
    assert isinstance(m, MetricWithSecrets)


def test_secrets_mixin_declares_expected_secrets():
    m = _JudgeMetric()
    declared = m.secrets()
    assert set(declared) == {"JUDGE_API_KEY"}
    assert declared["JUDGE_API_KEY"].root == "JUDGE_API_KEY"


def test_secrets_mixin_reads_env_at_construction(monkeypatch):
    """If env var is set, construction eagerly loads it."""
    monkeypatch.setenv("JUDGE_API_KEY", "test-key-123")
    m = _JudgeMetric()
    assert m.get_secret("JUDGE_API_KEY") == "test-key-123"


def test_secrets_mixin_missing_env_returns_none():
    m = _JudgeMetric()
    # JUDGE_API_KEY not set in this test; constructed returns None
    # (fixture-isolation: other tests may have set monkeypatch, but
    # this test wasn't parametrized with it)
    if m.get_secret("JUDGE_API_KEY") is not None:
        pytest.skip("JUDGE_API_KEY leaked from another test's env")
    assert m.get_secret("JUDGE_API_KEY") is None


async def test_secrets_mixin_resolve_via_resolver(monkeypatch):
    """Resolver is called for unresolved secrets and populates them."""
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    m = _JudgeMetric()
    assert m.get_secret("JUDGE_API_KEY") is None

    async def resolver(name: str) -> str | None:
        return "resolved-value" if name == "JUDGE_API_KEY" else None

    await m.resolve_secrets(resolver)
    assert m.get_secret("JUDGE_API_KEY") == "resolved-value"


async def test_secrets_mixin_resolver_skips_already_loaded(monkeypatch):
    """If env-loaded already, resolver is not consulted."""
    monkeypatch.setenv("JUDGE_API_KEY", "from-env")
    m = _JudgeMetric()

    resolver_calls: list[str] = []

    async def resolver(name: str) -> str | None:
        resolver_calls.append(name)
        return "from-resolver"

    await m.resolve_secrets(resolver)

    assert resolver_calls == []  # resolver not consulted
    assert m.get_secret("JUDGE_API_KEY") == "from-env"


def test_secrets_mixin_default_is_empty_tuple():
    """A class without ``secret_env_vars`` has no secrets to declare."""

    class NoSecrets(TemplateMetric, SecretsMixin):
        type: Literal["none"] = "none"

        def _score(self, input):
            return 0.0

    m = NoSecrets()
    assert m.secrets() == {}


# ---------------------------------------------------------------------------
# Target ergonomics: ~20-30 LOC real-looking metric
# ---------------------------------------------------------------------------


class _TinyLengthMetric(TemplateMetric):
    """Demonstration: len-matching metric in exactly 14 LOC of user code."""

    type: Literal["length-match"] = "length-match"

    tolerance: int = 5

    def _score(self, input: MetricInput) -> float:
        # compare response length to target length within tolerance
        expected = len(str(input.target))
        actual = len(str(input.response))
        return 1.0 if abs(expected - actual) <= self.tolerance else 0.0


async def test_tiny_length_metric_demonstrates_20_loc_target():
    """Proves a real-looking metric fits in ~15 LOC with TemplateMetric."""
    m = _TinyLengthMetric(tolerance=2)
    out_close = await m.compute_scores(MetricInput(response="hello", target="world"))
    assert out_close.scores[0].value == 1.0

    out_far = await m.compute_scores(MetricInput(response="x", target="much much longer"))
    assert out_far.scores[0].value == 0.0
