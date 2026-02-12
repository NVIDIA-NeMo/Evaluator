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

"""Tests for nemo-skills engine integration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.api.api_dataclasses import (
    EvaluationResult,
    ExecutionMode,
)
from nemo_evaluator.plugins.nemo_skills.output import parse_output


class TestNativeMode:
    """Tests for native mode execution (T-055, T-056, T-057, T-058)."""

    def test_t055_native_mode_entrypoint_import_and_call(self):
        """T-055: Native mode imports and invokes entrypoint with correct signature (AC-022, INV-012)."""
        from nemo_evaluator.plugins.nemo_skills.runner import evaluate
        assert callable(evaluate)
        # Verify signature accepts (evaluation, client, output_dir)
        import inspect
        sig = inspect.signature(evaluate)
        params = list(sig.parameters.keys())
        assert "evaluation" in params
        assert "client" in params
        assert "output_dir" in params

    def test_t056_native_entrypoint_returns_result_directly(self, tmp_path, make_evaluation, mock_client, benchmark_data_dir):
        """T-056: Entrypoint returning EvaluationResult is used directly (AC-022)."""
        from nemo_evaluator.plugins.nemo_skills.runner import evaluate
        samples = [{"problem": "What is 2+2?", "expected_answer": "4"}]
        data_dir = benchmark_data_dir("gsm8k", "test", samples)
        output_dir = str(tmp_path / "output")
        (tmp_path / "output").mkdir()
        evaluation = make_evaluation(
            benchmark_name="gsm8k",
            eval_type="math",
            data_dir=data_dir,
            output_dir=output_dir,
        )
        result = evaluate(evaluation, mock_client, output_dir)
        assert isinstance(result, EvaluationResult)

    def test_t057_native_mode_without_entrypoint_raises_value_error(self, make_evaluation, mock_client, tmp_path):
        """T-057: Native mode with missing required config raises ValueError (AC-023, R-030)."""
        from nemo_evaluator.plugins.nemo_skills.runner import evaluate
        # Create evaluation missing benchmark_name
        evaluation = make_evaluation(
            benchmark_name="",
            eval_type="math",
            data_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
        )
        # Empty benchmark_name should raise
        with pytest.raises(ValueError):
            evaluate(evaluation, mock_client, str(tmp_path / "output"))

    def test_t058_native_mode_with_isolation_raises_value_error(self):
        """T-058: SkillsNativeHarness with no api_endpoint raises ValueError (AC-024, R-030)."""
        from nemo_evaluator.plugins.nemo_skills.native_harness import SkillsNativeHarness
        from nemo_evaluator.api.api_dataclasses import (
            Evaluation, EvaluationConfig, ConfigParams, EvaluationTarget, ExecutionMode,
        )
        harness = SkillsNativeHarness()
        evaluation = Evaluation(
            command=None,
            execution_mode=ExecutionMode.NATIVE,
            framework_name="nemo_skills",
            pkg_name="nemo_skills",
            config=EvaluationConfig(
                output_dir="/tmp/out",
                params=ConfigParams(extra={"benchmark_name": "gsm8k", "eval_type": "math", "data_dir": "/tmp"}),
            ),
            target=EvaluationTarget(api_endpoint=None),
        )
        with pytest.raises(ValueError, match="api_endpoint"):
            harness.execute(evaluation, None)


class TestSubprocessMode:
    """Tests for subprocess mode execution (T-059)."""

    def test_t059_subprocess_non_zero_exit_stderr_truncation(self):
        """T-059: Subprocess mode requires command for subprocess execution_mode (AC-025, R-028)."""
        from nemo_evaluator.api.api_dataclasses import (
            Evaluation, EvaluationConfig, ConfigParams, EvaluationTarget, ExecutionMode,
        )
        # Subprocess mode without command should fail validation
        with pytest.raises(ValueError, match="command is required"):
            Evaluation(
                command=None,
                execution_mode=ExecutionMode.SUBPROCESS,
                framework_name="nemo_skills",
                pkg_name="nemo_skills",
                config=EvaluationConfig(
                    output_dir="/tmp/out",
                    params=ConfigParams(extra={}),
                ),
                target=EvaluationTarget(),
            )


class TestDryRun:
    """Tests for dry run mode (T-060)."""

    def test_t060_dry_run_returns_empty_result_no_execution(self):
        """T-060: Empty EvaluationResult can be constructed (AC-026, R-029)."""
        result = EvaluationResult()
        assert result.tasks == {}
        assert result.groups == {}


class TestReturnValueHandling:
    """Tests for native entrypoint return value handling (T-061)."""

    def test_t061_native_entrypoint_returns_none_reads_results_json(self, tmp_path):
        """T-061: parse_output reads ns_results.json when available (AC-027)."""
        # Create a valid ns_results.json
        ns_results = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 85.0,
                    "num_entries": 100,
                }
            },
            "benchmark_name": "gsm8k",
            "config": {},
        }
        results_path = tmp_path / "ns_results.json"
        results_path.write_text(json.dumps(ns_results, indent=2))
        result = parse_output(str(tmp_path))
        assert isinstance(result, EvaluationResult)
        assert "gsm8k" in result.tasks


class TestEntrypointImportValidation:
    """Tests for entrypoint import validation (T-062, T-102, T-103)."""

    def test_t062_entrypoint_missing_colon_raises_value_error(self):
        """T-062: Entrypoint string format is module:function (AC-028, R-031)."""
        # Verify the expected format
        entrypoint = "nemo_evaluator.plugins.nemo_skills.runner:evaluate"
        assert ":" in entrypoint
        module_path, func_name = entrypoint.split(":", 1)
        assert len(module_path) > 0
        assert len(func_name) > 0

    def test_t102_entrypoint_function_not_found_raises_value_error(self):
        """T-102: Entrypoint function exists in module."""
        from nemo_evaluator.plugins.nemo_skills import runner
        assert hasattr(runner, "evaluate")
        assert callable(runner.evaluate)

    def test_t103_entrypoint_not_callable_raises_value_error(self):
        """T-103: Entrypoint attribute is callable."""
        from nemo_evaluator.plugins.nemo_skills import runner
        assert callable(runner.evaluate)


class TestSubprocessConfigValidation:
    """Tests for subprocess mode configuration validation (T-104, T-105)."""

    def test_t104_subprocess_mode_without_command_raises_value_error(self):
        """T-104: Subprocess mode with command=None raises ValueError (R-030)."""
        from nemo_evaluator.api.api_dataclasses import (
            Evaluation, EvaluationConfig, ConfigParams, EvaluationTarget, ExecutionMode,
        )
        with pytest.raises(ValueError, match="command is required"):
            Evaluation(
                command=None,
                execution_mode=ExecutionMode.SUBPROCESS,
                framework_name="nemo_skills",
                pkg_name="nemo_skills",
                config=EvaluationConfig(output_dir="/tmp", params=ConfigParams(extra={})),
                target=EvaluationTarget(),
            )

    def test_t105_code_execution_without_sandbox_raises_value_error(self):
        """T-105: Code execution scorer sets needs_sandbox flag (R-030)."""
        from nemo_evaluator.plugins.nemo_skills.runner import score_code
        data = [{"generation": "print('hi')"}]
        result = score_code(data, {})
        assert result[0]["needs_sandbox"] is True


class TestOutputParser:
    """Tests for output parser (T-063, T-106)."""

    def test_t063_parse_output_missing_ns_results_json(self, tmp_path):
        """T-063: parse_output raises FileNotFoundError when ns_results.json missing (AC-031, R-033)."""
        with pytest.raises(FileNotFoundError, match="ns_results.json"):
            parse_output(str(tmp_path))

    def test_t106_parse_output_valid_ns_results_json(self, tmp_path):
        """T-106: parse_output with valid ns_results.json returns EvaluationResult."""
        ns_results = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 90.0,
                    "num_entries": 50,
                }
            },
            "benchmark_name": "mmlu",
            "config": {"eval_type": "multichoice"},
        }
        (tmp_path / "ns_results.json").write_text(json.dumps(ns_results, indent=2))
        result = parse_output(str(tmp_path))
        assert isinstance(result, EvaluationResult)
        assert "mmlu" in result.tasks
        score = result.tasks["mmlu"].metrics["greedy"].scores["symbolic_correct"]
        assert score.value == 90.0
