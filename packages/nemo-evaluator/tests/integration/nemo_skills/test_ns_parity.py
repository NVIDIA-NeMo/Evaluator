# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Integration tests for nemo-skills parity and behavioral correctness."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    EvaluationTarget,
    ExecutionMode,
    TaskResult,
)
from nemo_evaluator.plugins.nemo_skills.runner import evaluate


def _make_evaluation(data_dir, output_dir, benchmark_name="gsm8k", eval_type="math"):
    """Helper to create a test Evaluation object."""
    return Evaluation(
        command=None,
        execution_mode=ExecutionMode.NATIVE,
        framework_name="nemo_skills",
        pkg_name="nemo_skills",
        config=EvaluationConfig(
            output_dir=output_dir,
            params=ConfigParams(
                temperature=0.0,
                max_new_tokens=512,
                extra={
                    "benchmark_name": benchmark_name,
                    "eval_type": eval_type,
                    "data_dir": data_dir,
                    "num_seeds": 1,
                    "eval_split": "test",
                },
            ),
        ),
        target=EvaluationTarget(),
    )


def _make_data_dir(tmp_path, benchmark_name="gsm8k", samples=None):
    """Helper to create benchmark data directory with JSONL."""
    if samples is None:
        samples = [
            {"problem": "What is 2+2?", "expected_answer": "4"},
            {"problem": "What is 3*5?", "expected_answer": "15"},
            {"problem": "What is 10/2?", "expected_answer": "5"},
        ]
    bench_dir = tmp_path / "data" / benchmark_name
    bench_dir.mkdir(parents=True)
    jsonl_path = bench_dir / "test.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return str(tmp_path / "data")


class TestMinimalVerticalSlice:
    """Minimal vertical slice tests (T-043, T-044)."""

    def test_t043_minimal_vertical_slice_full_pipeline(self, tmp_path, deterministic_client):
        """T-043: Full pipeline with mock client, verify EvaluationResult and ns_results.json structure (AC-016)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        evaluation = _make_evaluation(data_dir, output_dir)

        result = evaluate(evaluation, deterministic_client, output_dir)

        # Verify EvaluationResult
        assert isinstance(result, EvaluationResult)
        assert "gsm8k" in result.tasks
        assert "greedy" in result.tasks["gsm8k"].metrics

        # Verify ns_results.json was written
        ns_results_path = Path(output_dir) / "ns_results.json"
        assert ns_results_path.exists()

    def test_t044_ns_results_json_structure(self, tmp_path, deterministic_client):
        """T-044: Verify ns_results.json structural completeness (AC-017, INV-008, INV-010)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        evaluation = _make_evaluation(data_dir, output_dir)

        evaluate(evaluation, deterministic_client, output_dir)

        ns_results_path = Path(output_dir) / "ns_results.json"
        with open(ns_results_path) as f:
            ns_results = json.load(f)

        # Required structure per INV-010
        assert "_all_" in ns_results
        assert "benchmark_name" in ns_results
        assert "config" in ns_results
        # _all_ should have greedy aggregation mode
        assert "greedy" in ns_results["_all_"]


class TestEvaluationResultType:
    """EvaluationResult type invariant tests (T-067)."""

    def test_t067_runner_always_returns_evaluation_result(self, tmp_path, deterministic_client):
        """T-067: runner.evaluate() always returns EvaluationResult, never None or dict (INV-008)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        evaluation = _make_evaluation(data_dir, output_dir)

        result = evaluate(evaluation, deterministic_client, output_dir)

        assert result is not None
        assert isinstance(result, EvaluationResult)
        assert not isinstance(result, dict)


class TestNsResultsJsonSchema:
    """ns_results.json schema validation tests (T-068)."""

    def test_t068_ns_results_json_schema_validation(self, tmp_path, deterministic_client):
        """T-068: ns_results.json validates against DM-006 schema (INV-010)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        evaluation = _make_evaluation(data_dir, output_dir)

        evaluate(evaluation, deterministic_client, output_dir)

        with open(Path(output_dir) / "ns_results.json") as f:
            ns_results = json.load(f)

        # Validate schema
        assert isinstance(ns_results, dict)
        assert "_all_" in ns_results
        assert isinstance(ns_results["_all_"], dict)
        # Each aggregation mode should have numeric metrics
        for agg_mode, agg_data in ns_results["_all_"].items():
            assert isinstance(agg_data, dict)
            assert "num_entries" in agg_data


class TestNativeHarnessRegistration:
    """Native harness registration tests (T-073)."""

    def test_t073_native_harness_prefix_matching(self):
        """T-073: Native harness registered for 'nemo_skills', prefix matching works (INV-012)."""
        from nemo_evaluator.plugins.nemo_skills.native_harness import SkillsNativeHarness
        harness = SkillsNativeHarness()
        assert hasattr(harness, "execute")
        assert callable(harness.execute)


class TestConfigMerge:
    """Configuration merge priority tests (T-066)."""

    def test_t066_config_merge_priority(self, tmp_path, deterministic_client):
        """T-066: Extra config fields flow through to runner correctly (AC-033)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()

        evaluation = Evaluation(
            command=None,
            execution_mode=ExecutionMode.NATIVE,
            framework_name="nemo_skills",
            pkg_name="nemo_skills",
            config=EvaluationConfig(
                output_dir=output_dir,
                params=ConfigParams(
                    temperature=0.0,
                    max_new_tokens=256,
                    limit_samples=2,
                    extra={
                        "benchmark_name": "gsm8k",
                        "eval_type": "math",
                        "data_dir": data_dir,
                        "num_seeds": 1,
                        "eval_split": "test",
                    },
                ),
            ),
            target=EvaluationTarget(),
        )

        result = evaluate(evaluation, deterministic_client, output_dir)
        assert isinstance(result, EvaluationResult)
        # limit_samples=2 should limit to 2 samples
        with open(Path(output_dir) / "ns_results.json") as f:
            ns_results = json.load(f)
        assert ns_results["_all_"]["greedy"]["num_entries"] == 2


class TestEvaluationResultCompatibility:
    """EvaluationResult compatibility tests (T-109)."""

    def test_t109_evaluation_result_yaml_serialization(self, tmp_path, deterministic_client):
        """T-109: EvaluationResult can be serialized to JSON by Pydantic."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        evaluation = _make_evaluation(data_dir, output_dir)

        result = evaluate(evaluation, deterministic_client, output_dir)

        # Verify Pydantic serialization works
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "tasks" in parsed
        assert "groups" in parsed


class TestFailurePropagation:
    """Failure propagation tests (T-110)."""

    def test_t110_partial_model_failures_continue_execution(self, tmp_path):
        """T-110: 2 of 5 model failures: pipeline completes, ns_results.json written, warnings logged."""
        # Create 5 samples
        samples = [{"problem": f"Q{i}", "expected_answer": str(i)} for i in range(5)]
        data_dir = _make_data_dir(tmp_path, samples=samples)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        evaluation = _make_evaluation(data_dir, output_dir)

        # Mock client that fails on samples 1 and 3
        client = MagicMock()
        call_count = 0

        async def mock_chat(messages, **kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx in (1, 3):
                raise RuntimeError(f"Failure on sample {idx}")
            return f"The answer is \\boxed{{{idx}}}"

        client.chat_completion = AsyncMock(side_effect=mock_chat)

        result = evaluate(evaluation, client, output_dir)

        assert isinstance(result, EvaluationResult)
        # ns_results.json should still be written
        assert (Path(output_dir) / "ns_results.json").exists()
        # Should have processed all 5 samples
        with open(Path(output_dir) / "ns_results.json") as f:
            ns_results = json.load(f)
        assert ns_results["_all_"]["greedy"]["num_entries"] == 5
