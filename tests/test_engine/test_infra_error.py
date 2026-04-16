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


import pytest

from nemo_evaluator.errors import GracefulError, InfraError
from nemo_evaluator.solvers.base import ErrorKind, SolveResult
from nemo_evaluator.engine.eval_loop import _get_error_category
from nemo_evaluator.observability.types import StepRecord


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
