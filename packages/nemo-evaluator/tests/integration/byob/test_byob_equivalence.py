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

"""Equivalence tests for BYOB native mode vs subprocess mode.

CRITICAL CORRECTNESS TESTS: These tests validate that native mode produces
IDENTICAL results to subprocess mode for the same inputs. Any divergence is
a correctness bug.

All tests use the deterministic MockServer, which produces the same response
for the same prompt (via MD5 hash). This ensures reproducibility and makes
any divergence a real bug, not test flakiness.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_evaluator.api.api_dataclasses import Evaluation, EvaluationConfig, ConfigParams, EvaluationTarget, ApiEndpoint
from nemo_evaluator.byob.native_runner import ByobNativeHarness
from nemo_evaluator.core.native_harness import make_model_call_fn_direct


class TestEquivalence:
    """Equivalence tests between native and subprocess execution modes.

    These are the MOST IMPORTANT tests for native mode. Any failure indicates
    a divergence in evaluation logic between the two modes, which is a
    critical correctness bug.
    """

    def test_native_vs_subprocess_equivalence(self, tmp_path, mock_model_server):
        """CRITICAL: Native and subprocess modes MUST produce identical results.

        Runs the same benchmark with the same dataset in both modes and
        compares the resulting score dictionaries field-by-field. Uses
        deterministic MockServer so any divergence is a real bug.

        Validates:
        - Same task names
        - Same metric names
        - Same score names
        - Same numeric values (exact equality for deterministic inputs)
        - Same count, mean, stderr, stddev (exact equality)
        """
        # Create shared benchmark file
        benchmark_file = tmp_path / "equiv_benchmark.py"
        benchmark_file.write_text('''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="equiv-test",
    dataset="unused",
    prompt="Q: {question}\\nA:",
    target_field="answer",
    endpoint_type="chat"
)
@scorer
def equiv_scorer(response, target, metadata):
    """Simple contains scorer for equivalence testing."""
    return {"correct": target.lower() in response.lower()}
''')

        # Create shared dataset with 5 samples
        # MockServer is deterministic, so same prompts -> same responses
        dataset_file = tmp_path / "equiv_data.jsonl"
        dataset_file.write_text(
            '{"question": "Is the sky blue?", "answer": "yes"}\n'
            '{"question": "Is water dry?", "answer": "no"}\n'
            '{"question": "Do cats meow?", "answer": "yes"}\n'
            '{"question": "Can fish fly?", "answer": "no"}\n'
            '{"question": "Is grass green?", "answer": "yes"}\n'
        )

        # --- Run subprocess mode ---
        subprocess_output_dir = tmp_path / "subprocess_output"
        subprocess_output_dir.mkdir()

        proc = subprocess.run(
            [
                sys.executable, "-m", "nemo_evaluator.byob.runner",
                "--benchmark-module", str(benchmark_file),
                "--benchmark-name", "equiv_test",
                "--dataset", str(dataset_file),
                "--output-dir", str(subprocess_output_dir),
                "--model-url", mock_model_server.url,
                "--model-id", "test-model",
                "--model-type", "chat",
                "--temperature", "0",
                "--max-tokens", "4096",
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert proc.returncode == 0, \
            f"Subprocess mode failed with exit code {proc.returncode}. Stderr: {proc.stderr}"

        subprocess_results_path = subprocess_output_dir / "byob_results.json"
        assert subprocess_results_path.is_file(), \
            f"Subprocess mode did not write byob_results.json to {subprocess_results_path}"

        with open(subprocess_results_path) as f:
            subprocess_result = json.load(f)

        # --- Run native mode ---
        native_output_dir = tmp_path / "native_output"
        native_output_dir.mkdir()

        evaluation = Evaluation(
            name="equiv-test",
            description="Equivalence test",
            config=EvaluationConfig(
                type="byob_equiv_test.equiv-test",
                supported_endpoint_types=["chat"],
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(benchmark_file),
                        "benchmark_name": "equiv_test",
                        "dataset": str(dataset_file),
                    }
                )
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="chat")
            ),
        )

        model_call_fn = make_model_call_fn_direct(
            model_url=mock_model_server.url,
            model_id="test-model",
            temperature=0.0,
            max_tokens=4096,
            api_key=None,
        )

        harness = ByobNativeHarness()
        native_result_obj = harness.execute(evaluation, model_call_fn)

        # Convert native EvaluationResult to dict format matching subprocess JSON
        # This allows direct comparison of the data structures
        native_result = self._evaluation_result_to_dict(native_result_obj)

        # --- Compare results field-by-field ---
        # Using detailed assertions for clear failure messages
        assert "tasks" in subprocess_result, "Subprocess result missing 'tasks' key"
        assert "tasks" in native_result, "Native result missing 'tasks' key"

        subprocess_tasks = subprocess_result["tasks"]
        native_tasks = native_result["tasks"]

        assert subprocess_tasks.keys() == native_tasks.keys(), (
            f"Task name mismatch: subprocess={list(subprocess_tasks.keys())}, "
            f"native={list(native_tasks.keys())}"
        )

        for task_name in subprocess_tasks.keys():
            sub_task = subprocess_tasks[task_name]
            nat_task = native_tasks[task_name]

            assert sub_task.keys() == nat_task.keys(), (
                f"Task '{task_name}' structure mismatch: "
                f"subprocess keys={list(sub_task.keys())}, native keys={list(nat_task.keys())}"
            )

            sub_metrics = sub_task["metrics"]
            nat_metrics = nat_task["metrics"]

            assert sub_metrics.keys() == nat_metrics.keys(), (
                f"Metric names mismatch in task '{task_name}': "
                f"subprocess={list(sub_metrics.keys())}, native={list(nat_metrics.keys())}"
            )

            for metric_name in sub_metrics.keys():
                sub_metric = sub_metrics[metric_name]
                nat_metric = nat_metrics[metric_name]

                sub_scores = sub_metric["scores"]
                nat_scores = nat_metric["scores"]

                assert sub_scores.keys() == nat_scores.keys(), (
                    f"Score names mismatch in metric '{metric_name}': "
                    f"subprocess={list(sub_scores.keys())}, native={list(nat_scores.keys())}"
                )

                for score_name in sub_scores.keys():
                    sub_score = sub_scores[score_name]
                    nat_score = nat_scores[score_name]

                    # Compare all fields with exact equality (deterministic inputs)
                    for field in ["value", "count", "mean", "stderr", "stddev"]:
                        assert field in sub_score, \
                            f"Subprocess score '{score_name}' missing field '{field}'"
                        assert field in nat_score, \
                            f"Native score '{score_name}' missing field '{field}'"

                        sub_val = sub_score[field]
                        nat_val = nat_score[field]

                        assert sub_val == nat_val, (
                            f"EQUIVALENCE FAILURE in score '{score_name}' field '{field}':\n"
                            f"  Subprocess: {sub_val}\n"
                            f"  Native:     {nat_val}\n"
                            f"  Full subprocess score: {json.dumps(sub_score, indent=2)}\n"
                            f"  Full native score:     {json.dumps(nat_score, indent=2)}"
                        )

        # If we reach here, results are equivalent
        # Write a summary for debugging
        print(f"\n=== EQUIVALENCE TEST PASSED ===")
        print(f"Subprocess result: {json.dumps(subprocess_result, indent=2)}")
        print(f"Native result:     {json.dumps(native_result, indent=2)}")

    def test_equivalence_binary_scores(self, tmp_path, mock_model_server):
        """Binary scores (True/False) should be scaled identically (100x) in both modes.

        Binary detection: All values are 0.0 or 1.0 (exact float comparison).
        Binary scaling: multiply value, mean, stderr, stddev by 100.
        This test validates that both modes apply the same scaling logic.
        """
        # Create benchmark with boolean scorer
        benchmark_file = tmp_path / "binary_benchmark.py"
        benchmark_file.write_text('''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="binary-test",
    dataset="unused",
    prompt="Q: {question}\\nA:",
    target_field="answer",
    endpoint_type="chat"
)
@scorer
def binary_scorer(response, target, metadata):
    """Returns boolean value (will be converted to 0.0 or 1.0)."""
    # This scorer returns True/False which aggregate_scores converts to 0.0/1.0
    return {"correct": target.lower() in response.lower()}
''')

        dataset_file = tmp_path / "binary_data.jsonl"
        dataset_file.write_text(
            '{"question": "Q1", "answer": "yes"}\n'
            '{"question": "Q2", "answer": "yes"}\n'
            '{"question": "Q3", "answer": "yes"}\n'
        )

        # Run subprocess mode
        subprocess_output_dir = tmp_path / "subprocess_binary"
        subprocess_output_dir.mkdir()

        proc = subprocess.run(
            [
                sys.executable, "-m", "nemo_evaluator.byob.runner",
                "--benchmark-module", str(benchmark_file),
                "--benchmark-name", "binary_test",
                "--dataset", str(dataset_file),
                "--output-dir", str(subprocess_output_dir),
                "--model-url", mock_model_server.url,
                "--model-id", "test-model",
                "--model-type", "chat",
                "--temperature", "0",
                "--max-tokens", "4096",
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert proc.returncode == 0, f"Subprocess mode failed: {proc.stderr}"

        with open(subprocess_output_dir / "byob_results.json") as f:
            subprocess_result = json.load(f)

        # Run native mode
        evaluation = Evaluation(
            name="binary-test",
            description="Binary test",
            config=EvaluationConfig(
                type="byob_binary_test.binary-test",
                supported_endpoint_types=["chat"],
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(benchmark_file),
                        "benchmark_name": "binary_test",
                        "dataset": str(dataset_file),
                    }
                )
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="chat")
            ),
        )

        model_call_fn = make_model_call_fn_direct(
            model_url=mock_model_server.url,
            model_id="test-model",
            temperature=0.0,
            max_tokens=4096,
            api_key=None,
        )

        harness = ByobNativeHarness()
        native_result_obj = harness.execute(evaluation, model_call_fn)
        native_result = self._evaluation_result_to_dict(native_result_obj)

        # Compare binary scaling
        sub_scores = subprocess_result["tasks"]["binary_test"]["metrics"]["pass@1"]["scores"]["correct"]
        nat_scores = native_result["tasks"]["binary_test"]["metrics"]["pass@1"]["scores"]["correct"]

        # All scaled fields should match exactly
        for field in ["value", "mean", "stderr", "stddev"]:
            assert sub_scores[field] == nat_scores[field], (
                f"Binary score field '{field}' mismatch:\n"
                f"  Subprocess: {sub_scores[field]}\n"
                f"  Native:     {nat_scores[field]}"
            )

        # Validate count is the same
        assert sub_scores["count"] == nat_scores["count"], \
            f"Count mismatch: subprocess={sub_scores['count']}, native={nat_scores['count']}"

        # Validate that scaling was applied (value should be in range 0-100 for binary)
        assert 0 <= nat_scores["value"] <= 100, \
            f"Binary score should be in [0, 100], got {nat_scores['value']}"

    def test_equivalence_continuous_scores(self, tmp_path, mock_model_server):
        """Continuous (non-binary) scores should match exactly between modes.

        Continuous scores are NOT scaled by 100x. This test validates that
        both modes correctly detect continuous scores and apply no scaling.
        """
        # Create benchmark with continuous scorer
        benchmark_file = tmp_path / "continuous_benchmark.py"
        benchmark_file.write_text('''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="continuous-test",
    dataset="unused",
    prompt="Q: {question}\\nA:",
    target_field="answer",
    endpoint_type="chat"
)
@scorer
def continuous_scorer(response, target, metadata):
    """Returns continuous value between 0 and 1 (simulating F1 or similarity)."""
    # Return non-binary values to prevent 100x scaling
    if target.lower() in response.lower():
        return {"similarity": 0.95}  # High but not exactly 1.0
    else:
        return {"similarity": 0.15}  # Low but not exactly 0.0
''')

        dataset_file = tmp_path / "continuous_data.jsonl"
        dataset_file.write_text(
            '{"question": "Q1", "answer": "yes"}\n'
            '{"question": "Q2", "answer": "no"}\n'
            '{"question": "Q3", "answer": "yes"}\n'
        )

        # Run subprocess mode
        subprocess_output_dir = tmp_path / "subprocess_continuous"
        subprocess_output_dir.mkdir()

        proc = subprocess.run(
            [
                sys.executable, "-m", "nemo_evaluator.byob.runner",
                "--benchmark-module", str(benchmark_file),
                "--benchmark-name", "continuous_test",
                "--dataset", str(dataset_file),
                "--output-dir", str(subprocess_output_dir),
                "--model-url", mock_model_server.url,
                "--model-id", "test-model",
                "--model-type", "chat",
                "--temperature", "0",
                "--max-tokens", "4096",
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert proc.returncode == 0, f"Subprocess mode failed: {proc.stderr}"

        with open(subprocess_output_dir / "byob_results.json") as f:
            subprocess_result = json.load(f)

        # Run native mode
        evaluation = Evaluation(
            name="continuous-test",
            description="Continuous test",
            config=EvaluationConfig(
                type="byob_continuous_test.continuous-test",
                supported_endpoint_types=["chat"],
                params=ConfigParams(
                    limit_samples=None,
                    max_new_tokens=4096,
                    temperature=0.0,
                    extra={
                        "benchmark_module": str(benchmark_file),
                        "benchmark_name": "continuous_test",
                        "dataset": str(dataset_file),
                    }
                )
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(type="chat")
            ),
        )

        model_call_fn = make_model_call_fn_direct(
            model_url=mock_model_server.url,
            model_id="test-model",
            temperature=0.0,
            max_tokens=4096,
            api_key=None,
        )

        harness = ByobNativeHarness()
        native_result_obj = harness.execute(evaluation, model_call_fn)
        native_result = self._evaluation_result_to_dict(native_result_obj)

        # Compare continuous scores (no scaling)
        sub_scores = subprocess_result["tasks"]["continuous_test"]["metrics"]["pass@1"]["scores"]["similarity"]
        nat_scores = native_result["tasks"]["continuous_test"]["metrics"]["pass@1"]["scores"]["similarity"]

        # All fields should match exactly
        for field in ["value", "mean", "stderr", "stddev", "count"]:
            assert sub_scores[field] == nat_scores[field], (
                f"Continuous score field '{field}' mismatch:\n"
                f"  Subprocess: {sub_scores[field]}\n"
                f"  Native:     {nat_scores[field]}"
            )

        # Validate that NO scaling was applied (values should be in [0, 1])
        assert 0 <= nat_scores["value"] <= 1, \
            f"Continuous score should be in [0, 1], got {nat_scores['value']}"

    # --- Helper methods ---

    def _evaluation_result_to_dict(self, result):
        """Convert EvaluationResult to dict format matching byob_results.json.

        This enables direct comparison with subprocess mode JSON output.
        """
        tasks_dict = {}
        for task_name, task_result in result.tasks.items():
            metrics_dict = {}
            for metric_name, metric_result in task_result.metrics.items():
                scores_dict = {}
                for score_name, score in metric_result.scores.items():
                    scores_dict[score_name] = {
                        "value": score.value,
                        "count": score.stats.count,
                        "mean": score.stats.mean,
                        "stderr": score.stats.stderr,
                        "stddev": score.stats.stddev,
                    }
                metrics_dict[metric_name] = {"scores": scores_dict}
            tasks_dict[task_name] = {"metrics": metrics_dict}
        return {"tasks": tasks_dict}
