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

"""End-to-end integration tests for BYOB feature.

This module tests the full BYOB workflow including:
- TruthfulQA scorer logic with mocked judge
- Compiler integration (compile + install)
- Runner E2E with mock server subprocess calls
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_evaluator.contrib.byob.compiler import compile_benchmark, install_benchmark
from nemo_evaluator.contrib.byob.decorators import ScorerInput


def _get_truthfulqa_benchmark_path():
    """Resolve path to TruthfulQA example benchmark.

    Searches multiple possible locations to handle different working directories.
    """
    test_file = Path(__file__).resolve()

    # Try relative to test file (go up to repo root)
    repo_root = test_file.parent.parent.parent.parent.parent.parent
    candidate = (
        repo_root
        / "packages"
        / "nemo-evaluator"
        / "examples"
        / "byob"
        / "truthfulqa"
        / "benchmark.py"
    )
    if candidate.exists():
        return candidate

    # Try relative to cwd
    candidate = Path("examples/byob/truthfulqa/benchmark.py")
    if candidate.exists():
        return candidate.resolve()

    # Try relative to package root
    try:
        import nemo_evaluator

        pkg_root = Path(nemo_evaluator.__file__).parent.parent
        candidate = pkg_root / "examples" / "byob" / "truthfulqa" / "benchmark.py"
        if candidate.exists():
            return candidate
    except (ImportError, AttributeError):
        pass

    pytest.skip("Could not locate TruthfulQA example benchmark.py")


def _load_truthfulqa_scorer():
    """Load the TruthfulQA scorer and its module from the example benchmark.

    Returns:
        (module, scorer_fn): the loaded module and its truthfulqa_scorer function.
        The module is returned so callers can patch module-level names (e.g. judge_score).
    """
    truthfulqa_benchmark_path = _get_truthfulqa_benchmark_path()

    spec = importlib.util.spec_from_file_location(
        "truthfulqa_benchmark", truthfulqa_benchmark_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod, mod.truthfulqa_scorer


class TestTruthfulQAScorer:
    """Integration tests for TruthfulQA scorer logic with mocked judge."""

    def test_truthful_response(self):
        """Scorer returns truthful=1.0 when judge grades the response as correct (C)."""
        mod, scorer_fn = _load_truthfulqa_scorer()
        with patch.object(
            mod, "judge_score", return_value={"judge_score": 1.0, "judge_grade": "C"}
        ):
            result = scorer_fn(
                ScorerInput(
                    response="The moon orbits Earth.",
                    target="The moon orbits Earth.",
                    metadata={
                        "correct_answers": ["The moon orbits Earth."],
                        "incorrect_answers": ["The sun orbits Earth."],
                    },
                )
            )
        assert result["truthful"] == 1.0
        assert result["judge_grade"] == "C"

    def test_untruthful_response(self):
        """Scorer returns truthful=0.0 when judge grades the response as incorrect (I)."""
        mod, scorer_fn = _load_truthfulqa_scorer()
        with patch.object(
            mod, "judge_score", return_value={"judge_score": 0.0, "judge_grade": "I"}
        ):
            result = scorer_fn(
                ScorerInput(
                    response="The Earth is flat.",
                    target="The Earth is roughly spherical.",
                    metadata={
                        "correct_answers": ["The Earth is roughly spherical."],
                        "incorrect_answers": ["The Earth is flat."],
                    },
                )
            )
        assert result["truthful"] == 0.0
        assert result["judge_grade"] == "I"


class TestCompilerIntegration:
    """Integration tests for BYOB compiler with real TruthfulQA example."""

    def test_compile_truthfulqa(self):
        """Compile the TruthfulQA example benchmark and validate FDF structure."""
        truthfulqa_benchmark_path = _get_truthfulqa_benchmark_path()

        result = compile_benchmark(str(truthfulqa_benchmark_path))

        assert "truthfulqa" in result, (
            f"Expected 'truthfulqa' key in compiled result, got: {list(result.keys())}"
        )
        fdf = result["truthfulqa"]

        # Framework section
        assert fdf["framework"]["name"] == "byob_truthfulqa"
        assert fdf["framework"]["pkg_name"] == "byob_truthfulqa"

        # Evaluations section
        assert len(fdf["evaluations"]) == 1
        assert fdf["evaluations"][0]["name"] == "truthfulqa"
        assert (
            "chat"
            in fdf["evaluations"][0]["defaults"]["config"]["supported_endpoint_types"]
        )

    def test_install_truthfulqa(self, tmp_path):
        """Install the TruthfulQA benchmark and validate package structure."""
        truthfulqa_benchmark_path = _get_truthfulqa_benchmark_path()

        compiled = compile_benchmark(str(truthfulqa_benchmark_path))

        for name, fdf in compiled.items():
            pkg_dir = install_benchmark(name, fdf, install_dir=str(tmp_path))
            pkg_path = Path(pkg_dir)

            assert (
                pkg_path / "nemo_evaluator" / "byob_truthfulqa" / "framework.yml"
            ).is_file(), "framework.yml should exist"
            assert (
                pkg_path / "nemo_evaluator" / "byob_truthfulqa" / "output.py"
            ).is_file(), "output.py should exist"
            assert (pkg_path / "pyproject.toml").is_file(), (
                "pyproject.toml should exist"
            )

            import yaml

            with open(
                pkg_path / "nemo_evaluator" / "byob_truthfulqa" / "framework.yml"
            ) as f:
                fw = yaml.safe_load(f)
            assert fw is not None, "framework.yml should be valid YAML"

            output_content = (
                pkg_path / "nemo_evaluator" / "byob_truthfulqa" / "output.py"
            ).read_text()
            assert "def parse_output" in output_content, (
                "output.py should contain parse_output function"
            )
            assert "byob_results.json" in output_content, (
                "output.py should reference byob_results.json"
            )


class TestRunnerE2E:
    """End-to-end tests for BYOB runner with mock server."""

    def test_runner_with_mock_server(self, tmp_path, mock_model_server):
        """Full lifecycle: create benchmark, dataset, call mock server, verify output."""
        # Create a temporary benchmark file
        benchmark_file = tmp_path / "test_benchmark.py"
        benchmark_file.write_text("""
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(
    name="test-e2e",
    dataset="unused",  # Will be overridden by CLI
    prompt="Q: {question}\\nA:",
    target_field="answer",
    endpoint_type="chat"
)
@scorer
def simple_scorer(sample):
    return {"correct": sample.target.lower() in sample.response.lower()}
""")

        # Create a temporary JSONL dataset
        dataset_file = tmp_path / "test_data.jsonl"
        dataset_file.write_text('{"question": "What is 2+2?", "answer": "yes"}\n')

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run the benchmark via subprocess
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nemo_evaluator.contrib.byob.runner",
                "--benchmark-module",
                str(benchmark_file),
                "--benchmark-name",
                "test_e2e",
                "--dataset",
                str(dataset_file),
                "--output-dir",
                str(output_dir),
                "--model-url",
                mock_model_server.url,
                "--model-id",
                "test-model",
                "--model-type",
                "chat",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Validate exit code
        assert result.returncode == 0, (
            f"Runner failed with exit code {result.returncode}. Stderr: {result.stderr}"
        )

        # Validate output file exists
        results_path = output_dir / "byob_results.json"
        assert results_path.is_file(), f"Expected byob_results.json at {results_path}"

        # Validate output structure
        with open(results_path) as f:
            output = json.load(f)

        assert "tasks" in output, "Output should contain 'tasks' key"
        assert "test_e2e" in output["tasks"], "Output should contain 'test_e2e' task"
        assert "metrics" in output["tasks"]["test_e2e"], "Task should contain 'metrics'"
        assert "pass@1" in output["tasks"]["test_e2e"]["metrics"], (
            "Metrics should contain 'pass@1'"
        )

        scores = output["tasks"]["test_e2e"]["metrics"]["pass@1"]["scores"]
        assert "correct" in scores, "Scores should contain 'correct' key"
        assert isinstance(scores["correct"]["value"], (int, float)), (
            f"Score value should be numeric, got: {type(scores['correct']['value'])}"
        )
        assert isinstance(scores["correct"]["stats"]["count"], int), (
            f"Score count should be int, got: {type(scores['correct']['stats']['count'])}"
        )
        assert scores["correct"]["stats"]["count"] > 0, "Score count should be > 0"

    def test_runner_cli_help(self):
        """Smoke test: --help should exit 0 and show expected flags."""
        result = subprocess.run(
            [sys.executable, "-m", "nemo_evaluator.contrib.byob.runner", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, f"--help should exit 0, got: {result.returncode}"
        assert "--benchmark-module" in result.stdout, (
            "Help output should mention --benchmark-module"
        )
        assert "--model-url" in result.stdout, "Help output should mention --model-url"
