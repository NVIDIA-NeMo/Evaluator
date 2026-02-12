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

"""End-to-end integration tests for BYOB native mode execution.

This module tests the full native mode workflow:
- Import benchmark from file
- Load JSONL dataset
- Call mock model server via model_call_fn
- Score responses
- Aggregate statistics
- Validate output structure and parseability
"""

import json
from pathlib import Path

import pytest

from nemo_evaluator.api.api_dataclasses import Evaluation, EvaluationConfig, ConfigParams, EvaluationTarget, ApiEndpoint
from nemo_evaluator.byob.compiler import compile_benchmark, install_benchmark
from nemo_evaluator.byob.native_runner import ByobNativeHarness
from nemo_evaluator.core.native_harness import make_model_call_fn_direct


class TestNativeE2E:
    """E2E tests for native mode with real mock server."""

    def test_native_runner_e2e_with_mock_server(self, tmp_path, mock_model_server):
        """Full lifecycle: import benchmark, load dataset, call mock server, verify result.

        This test validates the complete native mode execution path:
        1. ByobNativeHarness.execute() imports user benchmark module
        2. Loads JSONL dataset
        3. Calls model via injected model_call_fn (routed to mock server)
        4. Scores responses using benchmark scorer
        5. Aggregates statistics using runner.aggregate_scores()
        6. Returns EvaluationResult directly (no JSON file intermediary)
        """
        # Create temporary benchmark file
        benchmark_file = tmp_path / "test_native_e2e.py"
        benchmark_file.write_text('''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="native-e2e-test",
    dataset="unused",  # Overridden via config
    prompt="Q: {question}\\nA:",
    target_field="answer",
    endpoint_type="chat"
)
@scorer
def simple_contains_scorer(response, target, metadata):
    """Returns True if target substring is in response (case-insensitive)."""
    return {"correct": target.lower() in response.lower()}
''')

        # Create temporary JSONL dataset with 3 samples
        dataset_file = tmp_path / "test_native_data.jsonl"
        dataset_file.write_text(
            '{"question": "Is the sky blue?", "answer": "yes"}\n'
            '{"question": "Is water dry?", "answer": "no"}\n'
            '{"question": "Do cats meow?", "answer": "yes"}\n'
        )

        # Build Evaluation config for native mode
        evaluation = Evaluation(
            name="native-e2e-test",
            description="Native mode E2E test",
            config=EvaluationConfig(
                type="byob_native_e2e_test.native-e2e-test",
                supported_endpoint_types=["chat"],
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(benchmark_file),
                        "benchmark_name": "native_e2e_test",
                        "dataset": str(dataset_file),
                    }
                )
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="chat")
            ),
        )

        # Create model_call_fn routing to mock server
        model_call_fn = make_model_call_fn_direct(
            model_url=mock_model_server.url,
            model_id="test-model",
            temperature=0.0,
            max_tokens=4096,
            api_key=None,
        )

        # Execute native evaluation
        harness = ByobNativeHarness()
        result = harness.execute(evaluation, model_call_fn)

        # Validate result structure
        assert result is not None, "Native runner should return EvaluationResult"
        assert hasattr(result, "tasks"), "EvaluationResult should have tasks attribute"
        assert "native_e2e_test" in result.tasks, \
            f"Expected 'native_e2e_test' task in result, got: {list(result.tasks.keys())}"

        task = result.tasks["native_e2e_test"]
        assert "pass@1" in task.metrics, \
            f"Expected 'pass@1' metric, got: {list(task.metrics.keys())}"

        metric = task.metrics["pass@1"]
        assert "correct" in metric.scores, \
            f"Expected 'correct' score, got: {list(metric.scores.keys())}"

        score = metric.scores["correct"]
        assert score.value is not None, "Score value should not be None"
        assert isinstance(score.value, (int, float)), \
            f"Score value should be numeric, got: {type(score.value)}"
        assert score.stats is not None, "Score stats should not be None"
        assert score.stats.count == 3, \
            f"Expected 3 samples processed, got count={score.stats.count}"
        assert score.stats.mean is not None, "Mean should be computed"
        assert score.stats.stderr is not None, "Stderr should be computed"
        assert score.stats.stddev is not None, "Stddev should be computed"

    def test_native_output_parseable_by_output_py(self, tmp_path, mock_model_server):
        """Validate that native mode result structure matches output.py parser expectations.

        This is a contract test: the EvaluationResult returned by native mode
        MUST have the same structure that output.py would produce from parsing
        byob_results.json in subprocess mode.

        Note: Native mode returns EvaluationResult directly, so we validate
        that the structure matches what the engine expects, not that JSON
        parsing works (since there is no JSON file in native mode).
        """
        # Create temporary benchmark file
        benchmark_file = tmp_path / "test_native_contract.py"
        benchmark_file.write_text('''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="contract-test",
    dataset="unused",
    prompt="Q: {question}\\nA:",
    target_field="answer",
    endpoint_type="chat"
)
@scorer
def contract_scorer(response, target, metadata):
    return {"correct": target.lower() in response.lower()}
''')

        # Create temporary dataset
        dataset_file = tmp_path / "contract_data.jsonl"
        dataset_file.write_text(
            '{"question": "Test question 1", "answer": "yes"}\n'
            '{"question": "Test question 2", "answer": "no"}\n'
        )

        # Build evaluation config
        evaluation = Evaluation(
            name="contract-test",
            description="Contract test",
            config=EvaluationConfig(
                type="byob_contract_test.contract-test",
                supported_endpoint_types=["chat"],
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(benchmark_file),
                        "benchmark_name": "contract_test",
                        "dataset": str(dataset_file),
                    }
                )
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="chat")
            ),
        )

        # Create model_call_fn
        model_call_fn = make_model_call_fn_direct(
            model_url=mock_model_server.url,
            model_id="test-model",
            temperature=0.0,
            max_tokens=4096,
            api_key=None,
        )

        # Execute native evaluation
        harness = ByobNativeHarness()
        result = harness.execute(evaluation, model_call_fn)

        # Exhaustive contract validation - every field must match output.py schema
        assert hasattr(result, "tasks"), "Missing 'tasks' attribute"
        assert isinstance(result.tasks, dict), "tasks should be dict"
        assert len(result.tasks) > 0, "tasks should not be empty"

        for task_name, task_result in result.tasks.items():
            assert hasattr(task_result, "metrics"), \
                f"Task '{task_name}' missing 'metrics' attribute"
            assert isinstance(task_result.metrics, dict), \
                f"Task '{task_name}' metrics should be dict"

            for metric_name, metric_result in task_result.metrics.items():
                assert hasattr(metric_result, "scores"), \
                    f"Metric '{metric_name}' missing 'scores' attribute"
                assert isinstance(metric_result.scores, dict), \
                    f"Metric '{metric_name}' scores should be dict"

                for score_name, score in metric_result.scores.items():
                    # Validate Score object structure
                    assert hasattr(score, "value"), \
                        f"Score '{score_name}' missing 'value' attribute"
                    assert isinstance(score.value, (int, float)), \
                        f"Score '{score_name}' value should be numeric, got {type(score.value)}"

                    assert hasattr(score, "stats"), \
                        f"Score '{score_name}' missing 'stats' attribute"
                    assert score.stats is not None, \
                        f"Score '{score_name}' stats should not be None"

                    # Validate ScoreStats object structure
                    stats = score.stats
                    assert hasattr(stats, "count"), "ScoreStats missing 'count'"
                    assert hasattr(stats, "mean"), "ScoreStats missing 'mean'"
                    assert hasattr(stats, "stderr"), "ScoreStats missing 'stderr'"
                    assert hasattr(stats, "stddev"), "ScoreStats missing 'stddev'"

                    assert isinstance(stats.count, int), \
                        f"count should be int, got {type(stats.count)}"
                    assert stats.count > 0, "count should be > 0"
                    assert isinstance(stats.mean, (int, float, type(None))), \
                        f"mean should be numeric or None, got {type(stats.mean)}"
                    assert isinstance(stats.stderr, (int, float, type(None))), \
                        f"stderr should be numeric or None, got {type(stats.stderr)}"
                    assert isinstance(stats.stddev, (int, float, type(None))), \
                        f"stddev should be numeric or None, got {type(stats.stddev)}"

        # Validate that at least one score was computed (not empty result)
        task = list(result.tasks.values())[0]
        metric = list(task.metrics.values())[0]
        score = list(metric.scores.values())[0]
        assert score.stats.count == 2, \
            f"Expected 2 samples processed, got {score.stats.count}"
