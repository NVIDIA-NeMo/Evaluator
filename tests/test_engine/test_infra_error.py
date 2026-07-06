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
"""Tests for InfraError classification, tagging, resume filtering, and ModelClient wrapping."""

from __future__ import annotations

import asyncio

import pytest

from nemo_evaluator.engine.eval_loop import _get_error_category, _solve_failed_error_category, run_evaluation
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.errors import GracefulError, InfraError
from nemo_evaluator.observability.types import ModelResponse, StepRecord
from nemo_evaluator.solvers.base import ErrorKind, SolveResult

# ── Exception hierarchy ──────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_infra_error_is_not_graceful(self):
        assert not issubclass(InfraError, GracefulError)

    def test_infra_error_is_exception(self):
        assert issubclass(InfraError, Exception)

    def test_graceful_does_not_catch_infra(self):
        with pytest.raises(InfraError):
            try:
                raise InfraError("model died")
            except GracefulError:
                pytest.fail("GracefulError should not catch InfraError")


# ── ErrorKind on SolveResult ─────────────────────────────────────────


class TestErrorKind:
    def test_default_is_none(self):
        sr = SolveResult(response="ok")
        assert sr.error_kind == ErrorKind.NONE

    def test_infra_kind(self):
        sr = SolveResult(response="", error="dead", error_kind=ErrorKind.INFRA)
        assert sr.error_kind == ErrorKind.INFRA
        assert sr.error_kind.value == "infra_error"

    def test_solve_timeout_kind(self):
        sr = SolveResult(response="", error_kind=ErrorKind.SOLVE_TIMEOUT)
        assert sr.error_kind == ErrorKind.SOLVE_TIMEOUT
        assert sr.error_kind.value == "solve_timeout"
        assert sr.error is None


# ── _get_error_category helper ───────────────────────────────────────


class TestGetErrorCategory:
    def test_valid_infra(self):
        assert _get_error_category({"scoring_details": {"error_category": "infra_error"}}) == "infra_error"

    def test_valid_graceful(self):
        assert _get_error_category({"scoring_details": {"error_category": "graceful"}}) == "graceful"

    def test_missing_scoring_details(self):
        assert _get_error_category({}) is None

    def test_none_scoring_details(self):
        assert _get_error_category({"scoring_details": None}) is None

    def test_non_dict_scoring_details(self):
        assert _get_error_category({"scoring_details": "some string"}) is None

    def test_missing_error_category_key(self):
        assert _get_error_category({"scoring_details": {"method": "solve_failed"}}) is None


class TestSolveFailedErrorCategory:
    @pytest.mark.parametrize(
        ("error", "is_solve_timeout", "expected"),
        [
            ("litellm.BadRequestError: Error code: 400 - Malformed native tool-call JSON", False, "model_error"),
            ("504 Gateway Timeout: upstream timed out", False, "model_timeout"),
            ("429 Too Many Requests: rate_limit", False, "rate_limit"),
            ("503 Service Unavailable", False, "server_error"),
            ("Turn budget exhausted: 20/20 turns used", True, "turn_budget_exhausted"),
            (
                "litellm.ContextWindowExceededError: maximum context length is 262144 tokens",
                False,
                "context_window_exceeded",
            ),
            (
                "litellm.BudgetExceededError: Budget has been exceeded! Current cost: 2, Max budget: 1",
                False,
                "budget_exceeded",
            ),
            ("litellm.RouterRateLimitError: router cooldown active", False, "rate_limit"),
            ("litellm.RouterRateLimitErrorBasic: router cooldown active", False, "rate_limit"),
            ("litellm.Timeout: request timed out", False, "model_timeout"),
            ("litellm.BadGatewayError: MidStreamFallbackError: APIConnectionError", False, "server_error"),
            ("litellm.APIResponseValidationError: response was missing choices", False, "model_error"),
            ("litellm.GuardrailRaisedException: guardrail raised", False, "model_error"),
            ("litellm.LiteLLMUnknownProvider: unknown model provider", False, "model_error"),
            ("litellm.MockException: synthetic model client failure", False, "model_error"),
            ("litellm.OpenAIError: untyped model client failure", False, "model_error"),
            ("plain timeout", True, "solve_timeout"),
            ("Agent crashed: ValueError", False, "graceful"),
        ],
    )
    def test_model_and_timeout_categories(self, error, is_solve_timeout, expected):
        assert _solve_failed_error_category(error, is_infra=False, is_solve_timeout=is_solve_timeout) == expected

    def test_infra_is_preserved(self):
        assert _solve_failed_error_category("Error code: 400", is_infra=True) == "infra_error"


class _TerminalBenchErrorEnv(EvalEnvironment):
    name = "harbor://terminal-bench@2.0"

    def __init__(self) -> None:
        super().__init__()
        self.dataset = [{"task_id": "regex-chess"}]
        self.verify_called = False

    async def seed(self, idx: int) -> SeedResult:
        task = self.dataset[idx]
        return SeedResult(
            prompt="Solve the terminal task",
            expected_answer="done",
            metadata={"task_id": task["task_id"]},
        )

    async def verify(self, response, expected, **metadata):
        self.verify_called = True
        return VerifyResult(
            reward=1.0 if response == expected else 0.0,
            scoring_details={"method": "terminal-bench"},
        )


class _TerminalBenchErrorSolver:
    def __init__(self, error: str, error_kind: ErrorKind = ErrorKind.NONE) -> None:
        self.error = error
        self.error_kind = error_kind

    async def solve(self, task: SeedResult) -> SolveResult:
        return SolveResult(response="", error=self.error, error_kind=self.error_kind)


class _TerminalBenchHealthySolver:
    async def solve(self, task: SeedResult) -> SolveResult:
        return SolveResult(
            response=task.expected_answer,
            model_response=ModelResponse(
                content=task.expected_answer,
                model="fake-terminal-model",
                finish_reason="stop",
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )


class _TerminalBenchVerifiedTurnBudgetSolver:
    async def solve(self, task: SeedResult) -> SolveResult:
        return SolveResult(
            response=task.expected_answer,
            model_response=ModelResponse(
                content="[workspace modified]",
                model="fake-terminal-model",
                finish_reason="",
                prompt_tokens=30,
                completion_tokens=8,
                total_tokens=38,
            ),
            error_kind=ErrorKind.SOLVE_TIMEOUT,
            scoring_details={
                "error": "Agent crashed: ConversationRunError: Turn budget exhausted: 215/200 turns used",
                "error_category": "turn_budget_exhausted",
            },
        )


class TestTerminalBenchErrorClassification:
    @pytest.mark.parametrize(
        ("error", "error_kind", "expected"),
        [
            ("Error code: 400 - Malformed native tool-call JSON", ErrorKind.NONE, "model_error"),
            ("504 Gateway Timeout: upstream timed out", ErrorKind.SOLVE_TIMEOUT, "model_timeout"),
            ("Turn budget exhausted: 20/20 turns used", ErrorKind.SOLVE_TIMEOUT, "turn_budget_exhausted"),
        ],
    )
    def test_terminal_bench_solve_errors_keep_specific_categories(self, error, error_kind, expected):
        env = _TerminalBenchErrorEnv()
        solver = _TerminalBenchErrorSolver(error, error_kind)

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, max_concurrent=1))

        result = bundle["_results"][0]
        assert not env.verify_called
        assert result["scoring_details"]["error_category"] == expected

        step = bundle["_artifacts"].steps[0]
        assert step.failure_category == expected

        failures = bundle["benchmark"]["scores"]["failures"]
        assert failures["categories"][expected]["count"] == 1

    def test_terminal_bench_successful_200_style_response_has_no_failure_category(self):
        env = _TerminalBenchErrorEnv()
        solver = _TerminalBenchHealthySolver()

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, max_concurrent=1))

        result = bundle["_results"][0]
        assert env.verify_called
        assert result["reward"] == 1.0
        assert "error_category" not in result["scoring_details"]

        step = bundle["_artifacts"].steps[0]
        assert step.model_response.total_tokens == 14
        assert step.failure_category is None
        assert bundle["benchmark"]["scores"]["failures"]["total_failures"] == 0

    def test_turn_budget_with_workspace_diff_is_verified_and_keeps_category(self):
        env = _TerminalBenchErrorEnv()
        solver = _TerminalBenchVerifiedTurnBudgetSolver()

        bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, max_concurrent=1))

        result = bundle["_results"][0]
        assert env.verify_called
        assert result["reward"] == 1.0
        assert result["scoring_details"]["error_category"] == "turn_budget_exhausted"
        assert result["scoring_details"]["error"].startswith("Agent crashed: ConversationRunError")

        step = bundle["_artifacts"].steps[0]
        assert step.model_error is None
        assert step.failure_category == "turn_budget_exhausted"


# ── Collector classification ─────────────────────────────────────────


class TestCollectorInfraClassification:
    def test_infra_error_classified(self):
        from nemo_evaluator.observability.collector import ArtifactCollector

        collector = ArtifactCollector()
        step = StepRecord(
            problem_idx=0,
            repeat=0,
            prompt="test",
            scoring_details={"error_category": "infra_error"},
        )
        collector._classify_failure(step)
        assert step.failure_category == "infra_error"

    def test_graceful_falls_through_to_string_matching(self):
        from nemo_evaluator.observability.collector import ArtifactCollector

        collector = ArtifactCollector()
        step = StepRecord(
            problem_idx=0,
            repeat=0,
            prompt="test",
            model_error="HTTP 503 Server Error",
            scoring_details={"error_category": "graceful"},
        )
        collector._classify_failure(step)
        assert step.failure_category == "server_error"


class TestCollectorSolveTimeoutClassification:
    def test_solve_timeout_classified(self):
        from nemo_evaluator.observability.collector import ArtifactCollector

        collector = ArtifactCollector()
        step = StepRecord(
            problem_idx=0,
            repeat=0,
            prompt="test",
            scoring_details={"error_category": "solve_timeout", "method": "harbor"},
        )
        collector._classify_failure(step)
        assert step.failure_category == "solve_timeout"

    def test_solve_timeout_takes_precedence_over_substring_scan(self):
        """`solve_timeout` must short-circuit before the substring scan, otherwise a
        model_error string containing 'timed out' would mis-bucket as 'timeout'
        (the HTTP-408/upstream-gateway-timeout label)."""
        from nemo_evaluator.observability.collector import ArtifactCollector

        collector = ArtifactCollector()
        step = StepRecord(
            problem_idx=0,
            repeat=0,
            prompt="test",
            model_error="agent timed out after 3600s",
            scoring_details={"error_category": "solve_timeout"},
        )
        collector._classify_failure(step)
        assert step.failure_category == "solve_timeout"

    def test_solve_timeout_does_not_collide_with_infra_error(self):
        from nemo_evaluator.observability.collector import ArtifactCollector

        collector = ArtifactCollector()
        infra_step = StepRecord(
            problem_idx=0,
            repeat=0,
            prompt="test",
            scoring_details={"error_category": "infra_error"},
        )
        timeout_step = StepRecord(
            problem_idx=1,
            repeat=0,
            prompt="test",
            scoring_details={"error_category": "solve_timeout"},
        )
        collector._classify_failure(infra_step)
        collector._classify_failure(timeout_step)
        assert infra_step.failure_category == "infra_error"
        assert timeout_step.failure_category == "solve_timeout"


# ── ModelClient wraps errors as InfraError ───────────────────────────


class TestModelClientInfraError:
    def test_parse_response_empty_choices(self):
        from nemo_evaluator.engine.model_client import ModelClient

        client = ModelClient(base_url="http://localhost:8000/v1", model="test")
        with pytest.raises(InfraError, match="empty choices"):
            client._parse_response({"choices": []}, 100.0, "prompt", None)

    def test_parse_response_no_choices_key(self):
        from nemo_evaluator.engine.model_client import ModelClient

        client = ModelClient(base_url="http://localhost:8000/v1", model="test")
        with pytest.raises(InfraError, match="empty choices"):
            client._parse_response({"model": "test"}, 100.0, "prompt", None)


# ── Resume filtering ─────────────────────────────────────────────────


class TestResumeFiltering:
    """Verify that _get_error_category is used correctly for resume filtering."""

    def test_infra_entries_identified(self):
        cache = {
            (0, 0): {"reward": 1.0, "scoring_details": {"method": "exact_match"}},
            (1, 0): {"reward": 0.0, "scoring_details": {"error_category": "infra_error", "method": "infra_error"}},
            (2, 0): {"reward": 0.0, "scoring_details": {"error_category": "graceful", "method": "solve_failed"}},
        }
        infra_keys = {k for k, v in cache.items() if _get_error_category(v) == "infra_error"}
        assert infra_keys == {(1, 0)}

        filtered = {k: v for k, v in cache.items() if k not in infra_keys}
        assert (0, 0) in filtered
        assert (1, 0) not in filtered
        assert (2, 0) in filtered

    def test_old_format_without_error_category_kept(self):
        cache = {
            (0, 0): {"reward": 0.0, "scoring_details": {"error": "some error", "method": "solve_failed"}},
            (1, 0): {"reward": 1.0, "scoring_details": {}},
        }
        infra_keys = {k for k, v in cache.items() if _get_error_category(v) == "infra_error"}
        assert infra_keys == set()

    def test_both_caches_filtered(self):
        verified = {
            (0, 0): {"reward": 0.0, "scoring_details": {"error_category": "infra_error"}},
            (1, 0): {"reward": 1.0, "scoring_details": {}},
        }
        inferred = {
            (0, 0): {"response": "...", "tokens": 0},
            (1, 0): {"response": "ok", "tokens": 100},
        }
        infra_keys = {k for k, v in verified.items() if _get_error_category(v) == "infra_error"}
        v_filtered = {k: v for k, v in verified.items() if k not in infra_keys}
        i_filtered = {k: v for k, v in inferred.items() if k not in infra_keys}

        assert (0, 0) not in v_filtered
        assert (0, 0) not in i_filtered
        assert (1, 0) in v_filtered
        assert (1, 0) in i_filtered
