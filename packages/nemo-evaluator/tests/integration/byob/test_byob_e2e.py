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
- BoolQ scorer logic with realistic inputs
- Compiler integration (compile + install)
- Runner E2E with mock server subprocess calls
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_evaluator.contrib.byob.compiler import compile_benchmark, install_benchmark
from nemo_evaluator.contrib.byob.decorators import ScorerInput


def _get_boolq_benchmark_path():
    """Resolve path to BoolQ example benchmark.

    Searches multiple possible locations to handle different working directories.
    """
    test_file = Path(__file__).resolve()

    # Try relative to test file (go up to repo root)
    repo_root = test_file.parent.parent.parent.parent.parent.parent
    candidate = repo_root / "packages" / "nemo-evaluator" / "examples" / "byob" / "boolq" / "benchmark.py"
    if candidate.exists():
        return candidate

    # Try relative to cwd
    candidate = Path("examples/byob/boolq/benchmark.py")
    if candidate.exists():
        return candidate.resolve()

    # Try relative to package root
    try:
        import nemo_evaluator
        pkg_root = Path(nemo_evaluator.__file__).parent.parent
        candidate = pkg_root / "examples" / "byob" / "boolq" / "benchmark.py"
        if candidate.exists():
            return candidate
    except (ImportError, AttributeError):
        pass

    pytest.skip("Could not locate BoolQ example benchmark.py")


def _load_boolq_scorer():
    """Load the BoolQ scorer function from the example benchmark."""
    boolq_benchmark_path = _get_boolq_benchmark_path()

    import importlib.util
    spec = importlib.util.spec_from_file_location("boolq_benchmark", boolq_benchmark_path)
    boolq_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boolq_module)

    return boolq_module.boolq_scorer


class TestBoolQScorer:
    """Integration tests for BoolQ scorer logic."""

    def test_yes_true(self):
        """BoolQ scorer: 'Yes' response with target 'true' should be correct."""
        scorer = _load_boolq_scorer()
        result = scorer(ScorerInput(response="Yes, they do speak the same language.", target="true", metadata={}))
        assert result["correct"] is True, "Expected 'Yes' + target='true' to be correct"

    def test_no_false(self):
        """BoolQ scorer: 'No' response with target 'false' should be correct."""
        scorer = _load_boolq_scorer()
        result = scorer(ScorerInput(response="No, it is not based on a true story.", target="false", metadata={}))
        assert result["correct"] is True, "Expected 'No' + target='false' to be correct"

    def test_yes_false_is_wrong(self):
        """BoolQ scorer: 'Yes' response with target 'false' should be incorrect."""
        scorer = _load_boolq_scorer()
        result = scorer(ScorerInput(response="Yes, it is.", target="false", metadata={}))
        assert result["correct"] is False, "Expected 'Yes' + target='false' to be incorrect"


class TestCompilerIntegration:
    """Integration tests for BYOB compiler with real BoolQ example."""

    def test_compile_boolq(self):
        """Compile the BoolQ example benchmark and validate FDF structure."""
        boolq_benchmark_path = _get_boolq_benchmark_path()

        # Compile the benchmark
        result = compile_benchmark(str(boolq_benchmark_path))

        # Validate FDF structure
        assert "boolq" in result, f"Expected 'boolq' key in compiled result, got: {list(result.keys())}"
        fdf = result["boolq"]

        # Framework section
        assert fdf["framework"]["name"] == "byob_boolq"
        assert fdf["framework"]["pkg_name"] == "byob_boolq"

        # Evaluations section
        assert len(fdf["evaluations"]) == 1
        assert fdf["evaluations"][0]["name"] == "boolq"
        assert "chat" in fdf["evaluations"][0]["defaults"]["config"]["supported_endpoint_types"]

    def test_install_boolq(self, tmp_path):
        """Install the BoolQ benchmark and validate package structure."""
        boolq_benchmark_path = _get_boolq_benchmark_path()

        # Compile first
        compiled = compile_benchmark(str(boolq_benchmark_path))

        # Install to temp directory
        for name, fdf in compiled.items():
            pkg_dir = install_benchmark(name, fdf, install_dir=str(tmp_path))
            pkg_path = Path(pkg_dir)

            # Validate directory structure
            assert (pkg_path / "core_evals" / "byob_boolq" / "framework.yml").is_file(), \
                "framework.yml should exist"
            assert (pkg_path / "core_evals" / "byob_boolq" / "output.py").is_file(), \
                "output.py should exist"
            assert (pkg_path / "pyproject.toml").is_file(), \
                "pyproject.toml should exist"

            # Validate framework.yml is valid YAML
            import yaml
            with open(pkg_path / "core_evals" / "byob_boolq" / "framework.yml") as f:
                fw = yaml.safe_load(f)
            assert fw is not None, "framework.yml should be valid YAML"

            # Validate output.py contains parse_output function
            output_content = (pkg_path / "core_evals" / "byob_boolq" / "output.py").read_text()
            assert "def parse_output" in output_content, \
                "output.py should contain parse_output function"
            assert "byob_results.json" in output_content, \
                "output.py should reference byob_results.json"


class TestRunnerE2E:
    """End-to-end tests for BYOB runner with mock server."""

    def test_runner_with_mock_server(self, tmp_path, mock_model_server):
        """Full lifecycle: create benchmark, dataset, call mock server, verify output."""
        # Create a temporary benchmark file
        benchmark_file = tmp_path / "test_benchmark.py"
        benchmark_file.write_text('''
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
''')

        # Create a temporary JSONL dataset
        dataset_file = tmp_path / "test_data.jsonl"
        dataset_file.write_text('{"question": "What is 2+2?", "answer": "yes"}\n')

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run the benchmark via subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "nemo_evaluator.contrib.byob.runner",
                "--benchmark-module", str(benchmark_file),
                "--benchmark-name", "test_e2e",
                "--dataset", str(dataset_file),
                "--output-dir", str(output_dir),
                "--model-url", mock_model_server.url,
                "--model-id", "test-model",
                "--model-type", "chat",
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Validate exit code
        assert result.returncode == 0, \
            f"Runner failed with exit code {result.returncode}. Stderr: {result.stderr}"

        # Validate output file exists
        results_path = output_dir / "byob_results.json"
        assert results_path.is_file(), f"Expected byob_results.json at {results_path}"

        # Validate output structure
        with open(results_path) as f:
            output = json.load(f)

        assert "tasks" in output, "Output should contain 'tasks' key"
        assert "test_e2e" in output["tasks"], "Output should contain 'test_e2e' task"
        assert "metrics" in output["tasks"]["test_e2e"], "Task should contain 'metrics'"
        assert "pass@1" in output["tasks"]["test_e2e"]["metrics"], "Metrics should contain 'pass@1'"

        scores = output["tasks"]["test_e2e"]["metrics"]["pass@1"]["scores"]
        assert "correct" in scores, "Scores should contain 'correct' key"
        assert isinstance(scores["correct"]["value"], (int, float)), \
            f"Score value should be numeric, got: {type(scores['correct']['value'])}"
        assert isinstance(scores["correct"]["count"], int), \
            f"Score count should be int, got: {type(scores['correct']['count'])}"
        assert scores["correct"]["count"] > 0, "Score count should be > 0"

    def test_runner_cli_help(self):
        """Smoke test: --help should exit 0 and show expected flags."""
        result = subprocess.run(
            [sys.executable, "-m", "nemo_evaluator.contrib.byob.runner", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, f"--help should exit 0, got: {result.returncode}"
        assert "--benchmark-module" in result.stdout, \
            "Help output should mention --benchmark-module"
        assert "--model-url" in result.stdout, \
            "Help output should mention --model-url"
