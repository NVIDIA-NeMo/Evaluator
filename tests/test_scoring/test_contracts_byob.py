# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: @benchmark accepts TemplateMetric classes + instances.

Proves the 20-30 LOC user promise end-to-end.
"""

from __future__ import annotations

from typing import Literal

import pytest

from nemo_evaluator.environments.custom import _BYOB_REGISTRY, benchmark, scorer
from nemo_evaluator.scoring import MetricInput, TemplateMetric


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate registry state across tests."""
    snapshot = dict(_BYOB_REGISTRY)
    yield
    for key in list(_BYOB_REGISTRY):
        if key not in snapshot:
            del _BYOB_REGISTRY[key]


# ---------------------------------------------------------------------------
# @benchmark accepts function scorer (existing behavior — regression guard)
# ---------------------------------------------------------------------------


def test_benchmark_accepts_function_scorer_classic_path():
    @benchmark(name="classic-fn-bench", dataset=lambda: [{"q": "1", "answer": "2"}], prompt="{q}", target_field="answer")
    @scorer
    def score(sample):
        return {"correct": sample.response == sample.target}

    defn = _BYOB_REGISTRY["classic-fn-bench"]
    assert defn.scorer_fn is score


# ---------------------------------------------------------------------------
# @benchmark accepts a TemplateMetric class (new behavior)
# ---------------------------------------------------------------------------


def test_benchmark_accepts_template_metric_class():
    @benchmark(
        name="class-metric-bench",
        dataset=lambda: [{"q": "1", "answer": "1"}],
        prompt="{q}",
        target_field="answer",
    )
    class MyMetric(TemplateMetric):
        type: Literal["my-metric"] = "my-metric"

        def _score(self, input: MetricInput) -> float:
            return 1.0 if input.response == input.target else 0.0

    defn = _BYOB_REGISTRY["class-metric-bench"]
    assert defn.scorer_fn is not None

    # Invoke the wrapped scorer to confirm wiring
    result = defn.scorer_fn(MetricInput(response="42", target="42"))
    assert result["score"] == 1.0
    assert result["metric_type"] == "my-metric"


def test_benchmark_rejects_metric_class_needing_args():
    """If the Metric class needs constructor args, the user must pass an instance."""
    with pytest.raises(TypeError, match="no-arg constructor"):
        @benchmark(name="needs-args", dataset=lambda: [], prompt="", target_field="a")
        class RequiresArg(TemplateMetric):
            type: Literal["requires-arg"] = "requires-arg"
            # Required field with no default -> no-arg construction fails
            required_field: str

            def _score(self, input):
                return 0.0


# ---------------------------------------------------------------------------
# @benchmark accepts a pre-configured Metric instance via metric= kwarg
# ---------------------------------------------------------------------------


def test_benchmark_accepts_metric_instance_via_kwarg():
    class ConfiguredMetric(TemplateMetric):
        type: Literal["configured"] = "configured"
        threshold: float = 0.5

        def _score(self, input: MetricInput) -> float:
            return 1.0 if len(input.response) > self.threshold else 0.0

    configured = ConfiguredMetric(threshold=3.0)

    @benchmark(
        name="configured-bench",
        dataset=lambda: [{"q": "1", "answer": "1"}],
        prompt="{q}",
        target_field="answer",
        metric=configured,
    )
    def _placeholder():  # body unused when metric= is provided
        pass

    defn = _BYOB_REGISTRY["configured-bench"]
    assert defn.scorer_fn is not None

    result_long = defn.scorer_fn(MetricInput(response="long-enough", target="_"))
    assert result_long["score"] == 1.0
    result_short = defn.scorer_fn(MetricInput(response="x", target="_"))
    assert result_short["score"] == 0.0


def test_benchmark_rejects_class_that_does_not_satisfy_metric():
    with pytest.raises(TypeError, match="does not satisfy the Metric contract"):
        @benchmark(name="bogus", dataset=lambda: [], prompt="", target_field="a")
        class NotAMetric:  # missing required Metric methods
            pass


# ---------------------------------------------------------------------------
# Real "exact-match" metric in <25 LOC — proof of the 20-30 LOC promise
# ---------------------------------------------------------------------------


# Below this line is user code (counted against the 20-30 LOC target).
# Everything needed for a working benchmark + metric.
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

@benchmark(
    name="tiny-exact-match",
    dataset=lambda: [{"q": "2+2=?", "answer": "4"}, {"q": "3+3=?", "answer": "6"}],
    prompt="{q}",
    target_field="answer",
)
class TinyExactMatch(TemplateMetric):
    """Exact-match after normalization (lowercase + strip)."""

    type: Literal["tiny-exact-match"] = "tiny-exact-match"

    def _score(self, input: MetricInput) -> float:
        return 1.0 if self._norm(input.response) == self._norm(input.target) else 0.0

    @staticmethod
    def _norm(s: object) -> str:
        return str(s).strip().lower()

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# End of user code — that's 16 LOC of logic + decorator config for a full
# working benchmark registered in the NEL BYOB system.


def test_real_exact_match_metric_registered_and_callable():
    defn = _BYOB_REGISTRY["tiny-exact-match"]
    scorer_fn = defn.scorer_fn

    assert scorer_fn(MetricInput(response="4", target="4"))["score"] == 1.0
    assert scorer_fn(MetricInput(response="  4 ", target="4"))["score"] == 1.0  # normalization
    assert scorer_fn(MetricInput(response="five", target="4"))["score"] == 0.0


def test_real_exact_match_metric_registered_in_environment_registry():
    """The @benchmark decorator also creates an EvalEnvironment in the registry."""
    from nemo_evaluator.environments.registry import get_environment

    env = get_environment("tiny-exact-match")
    assert env is not None


async def test_real_exact_match_metric_end_to_end_via_environment():
    """Full flow: instantiate the environment, call verify() -> the metric evaluates."""
    from nemo_evaluator.environments.registry import get_environment

    env = get_environment("tiny-exact-match")
    result = await env.verify(response="4", expected="4", sandbox=None)
    assert result.reward == 1.0

    result_wrong = await env.verify(response="five", expected="4", sandbox=None)
    assert result_wrong.reward == 0.0
