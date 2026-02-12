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

"""Determinism and reproducibility tests for nemo-skills integration."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
    ExecutionMode,
)
from nemo_evaluator.plugins.nemo_skills.runner import evaluate


def _make_evaluation(data_dir, output_dir):
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


def _make_data_dir(tmp_path, subdir="data"):
    samples = [
        {"problem": "What is 2+2?", "expected_answer": "4"},
        {"problem": "What is 3*5?", "expected_answer": "15"},
    ]
    bench_dir = tmp_path / subdir / "gsm8k"
    bench_dir.mkdir(parents=True)
    with open(bench_dir / "test.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return str(tmp_path / subdir)


def _make_deterministic_client():
    client = MagicMock()
    async def mock_chat(messages, **kwargs):
        return "The answer is \\boxed{42}"
    client.chat_completion = AsyncMock(side_effect=mock_chat)
    return client


class TestFixedSeedRepeatability:
    """Fixed seed repeatability tests (T-069, T-070)."""

    def test_t069_byte_identical_ns_results_two_runs(self, tmp_path):
        """T-069: Two identical runs produce byte-identical ns_results.json (AC-018, INV-004)."""
        data_dir = _make_data_dir(tmp_path)

        # Run 1
        output_dir_1 = str(tmp_path / "output1")
        Path(output_dir_1).mkdir()
        client1 = _make_deterministic_client()
        evaluation1 = _make_evaluation(data_dir, output_dir_1)
        evaluate(evaluation1, client1, output_dir_1)

        # Run 2
        output_dir_2 = str(tmp_path / "output2")
        Path(output_dir_2).mkdir()
        client2 = _make_deterministic_client()
        evaluation2 = _make_evaluation(data_dir, output_dir_2)
        evaluate(evaluation2, client2, output_dir_2)

        # Compare byte-for-byte
        with open(Path(output_dir_1) / "ns_results.json", "rb") as f1:
            bytes1 = f1.read()
        with open(Path(output_dir_2) / "ns_results.json", "rb") as f2:
            bytes2 = f2.read()

        assert bytes1 == bytes2, "Two identical runs did not produce byte-identical ns_results.json"

    def test_t070_golden_file_comparison(self, tmp_path):
        """T-070: Output matches expected structure (INV-004)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        client = _make_deterministic_client()
        evaluation = _make_evaluation(data_dir, output_dir)
        evaluate(evaluation, client, output_dir)

        with open(Path(output_dir) / "ns_results.json") as f:
            ns_results = json.load(f)

        # Validate golden structure
        assert "_all_" in ns_results
        assert "greedy" in ns_results["_all_"]
        greedy = ns_results["_all_"]["greedy"]
        assert "symbolic_correct" in greedy
        assert "num_entries" in greedy
        assert greedy["num_entries"] == 2
        # Both answers are \\boxed{42} but expected are "4" and "15"
        # Neither matches, so symbolic_correct should be 0.0
        assert greedy["symbolic_correct"] == 0.0


class TestOrderIndependence:
    """Order independence tests (T-071)."""

    def test_t071_order_independence_shuffled_data(self, tmp_path):
        """T-071: Shuffled sample order produces identical aggregate metrics (INV-004)."""
        # Original order
        samples_a = [
            {"problem": "What is 2+2?", "expected_answer": "4"},
            {"problem": "What is 3+3?", "expected_answer": "6"},
        ]
        # Reversed order
        samples_b = list(reversed(samples_a))

        for name, samples in [("a", samples_a), ("b", samples_b)]:
            bench_dir = tmp_path / f"data_{name}" / "gsm8k"
            bench_dir.mkdir(parents=True)
            with open(bench_dir / "test.jsonl", "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

        # Run both
        results = {}
        for name in ["a", "b"]:
            data_dir = str(tmp_path / f"data_{name}")
            output_dir = str(tmp_path / f"output_{name}")
            Path(output_dir).mkdir()
            client = _make_deterministic_client()
            evaluation = _make_evaluation(data_dir, output_dir)
            evaluate(evaluation, client, output_dir)
            with open(Path(output_dir) / "ns_results.json") as f:
                results[name] = json.load(f)

        # Aggregate metrics should be identical
        assert results["a"]["_all_"]["greedy"]["symbolic_correct"] == \
               results["b"]["_all_"]["greedy"]["symbolic_correct"]
        assert results["a"]["_all_"]["greedy"]["num_entries"] == \
               results["b"]["_all_"]["greedy"]["num_entries"]


class TestConcurrencyBehavior:
    """Concurrency behavior tests (T-072)."""

    @pytest.mark.slow
    def test_t072_concurrent_execution_determinism(self, tmp_path):
        """T-072: Serial runs produce identical outputs (INV-004).

        NOTE: This test verifies no shared mutable state by running
        two evaluations sequentially and comparing outputs.
        """
        data_dir = _make_data_dir(tmp_path)

        outputs = []
        for i in range(2):
            output_dir = str(tmp_path / f"output_{i}")
            Path(output_dir).mkdir()
            client = _make_deterministic_client()
            evaluation = _make_evaluation(data_dir, output_dir)
            evaluate(evaluation, client, output_dir)
            with open(Path(output_dir) / "ns_results.json", "rb") as f:
                outputs.append(f.read())

        assert outputs[0] == outputs[1], "Serial runs produced different outputs"


class TestArtifactConsistency:
    """Artifact consistency tests (T-111)."""

    def test_t111_ns_results_json_formatting(self, tmp_path):
        """T-111: ns_results.json formatted with indent=2, UTF-8, ensure_ascii=False (INV-004)."""
        data_dir = _make_data_dir(tmp_path)
        output_dir = str(tmp_path / "output")
        Path(output_dir).mkdir()
        client = _make_deterministic_client()
        evaluation = _make_evaluation(data_dir, output_dir)
        evaluate(evaluation, client, output_dir)

        ns_results_path = Path(output_dir) / "ns_results.json"
        raw = ns_results_path.read_text(encoding="utf-8")
        # Verify indent=2 formatting
        assert "  " in raw
        # Verify valid JSON
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        # Re-serialize with same settings and compare
        expected = json.dumps(parsed, indent=2, ensure_ascii=False)
        assert raw == expected
