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

"""Smoketest: BoolQ BYOB benchmark against a live NVIDIA API endpoint.

This test calls a real model endpoint, so it requires:
  - NVIDIA_API_KEY environment variable set
  - Network access to integrate.api.nvidia.com

Run with:
  NVIDIA_API_KEY="nvapi-..." pytest tests/integration/byob/test_byob_boolq_smoketest.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_evaluator.byob.eval_logic import import_benchmark, run_eval_loop
from nemo_evaluator.byob.runner import aggregate_scores, call_model_chat, load_dataset

_NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1"
_NVIDIA_MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
# Only evaluate a small subset to keep the smoketest fast and cheap.
_SAMPLE_LIMIT = 3
# The model is a reasoning model that needs enough tokens for chain-of-thought
# plus the final answer.  64 is too low -- the model exhausts its budget on
# reasoning and returns content=null.  512 is sufficient for BoolQ yes/no.
_MAX_TOKENS = 512


def _get_api_key():
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        pytest.skip("NVIDIA_API_KEY not set – skipping live endpoint smoketest")
    return key


def _get_boolq_dir():
    """Resolve the BoolQ example directory."""
    test_file = Path(__file__).resolve()
    repo_root = test_file.parent.parent.parent.parent.parent.parent
    boolq_dir = repo_root / "packages" / "nemo-evaluator" / "examples" / "byob" / "boolq"
    if not boolq_dir.exists():
        pytest.skip(f"BoolQ example directory not found at {boolq_dir}")
    return boolq_dir


@pytest.mark.smoketest
class TestBoolQSmoketest:
    """Smoketest: run BoolQ BYOB benchmark against a live NVIDIA endpoint."""

    def test_live_endpoint_reachable(self):
        """Verify the NVIDIA API endpoint responds to a simple request."""
        api_key = _get_api_key()

        response = call_model_chat(
            url=_NVIDIA_API_URL,
            model_id=_NVIDIA_MODEL_ID,
            prompt="Say hello in one sentence.",
            temperature=0.2,
            max_tokens=_MAX_TOKENS,
            api_key=api_key,
            timeout=60,
        )
        assert isinstance(response, str), "Expected string response from model"
        assert len(response) > 0, "Response should not be empty"

    def test_boolq_eval_loop(self):
        """Run the BoolQ eval loop against the live endpoint and verify results."""
        api_key = _get_api_key()
        boolq_dir = _get_boolq_dir()

        # Import the BoolQ benchmark
        benchmark_path = str(boolq_dir / "benchmark.py")
        bench = import_benchmark(benchmark_path, "boolq")

        # Load a small slice of the dataset
        dataset_path = str(boolq_dir / "data.jsonl")
        dataset = load_dataset(dataset_path, limit=_SAMPLE_LIMIT)
        assert len(dataset) == _SAMPLE_LIMIT

        # Build model call function hitting the live endpoint
        def model_call_fn(prompt, endpoint_type: str) -> str:
            return call_model_chat(
                url=_NVIDIA_API_URL,
                model_id=_NVIDIA_MODEL_ID,
                prompt=prompt,
                temperature=0.2,
                max_tokens=_MAX_TOKENS,
                api_key=api_key,
                timeout=60,
            )

        # Run the evaluation loop
        scores, _ = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=model_call_fn,
            endpoint_type="chat",
        )

        # All samples should have been scored (no skips)
        assert len(scores) == _SAMPLE_LIMIT, (
            f"Expected {_SAMPLE_LIMIT} scored samples, got {len(scores)}"
        )

        # Each score dict must contain the 'correct' key with a boolean value
        for i, score in enumerate(scores):
            assert "correct" in score, f"Sample {i} missing 'correct' key: {score}"
            assert isinstance(score["correct"], bool), (
                f"Sample {i} 'correct' should be bool, got {type(score['correct'])}"
            )

        # Aggregate and validate output structure
        results = aggregate_scores(scores, "boolq")
        assert "tasks" in results
        assert "boolq" in results["tasks"]
        metrics = results["tasks"]["boolq"]["metrics"]["pass@1"]["scores"]
        assert "correct" in metrics
        assert metrics["correct"]["count"] == _SAMPLE_LIMIT
        # Value is scaled to 0-100 for binary metrics
        assert 0 <= metrics["correct"]["value"] <= 100

    def test_boolq_subprocess_runner(self, tmp_path):
        """Run the BoolQ benchmark via the subprocess runner CLI."""
        api_key = _get_api_key()
        boolq_dir = _get_boolq_dir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable, "-m", "nemo_evaluator.byob.runner",
                "--benchmark-module", str(boolq_dir / "benchmark.py"),
                "--benchmark-name", "boolq",
                "--dataset", str(boolq_dir / "data.jsonl"),
                "--output-dir", str(output_dir),
                "--model-url", _NVIDIA_API_URL,
                "--model-id", _NVIDIA_MODEL_ID,
                "--model-type", "chat",
                "--temperature", "0.2",
                "--max-tokens", str(_MAX_TOKENS),
                "--limit-samples", str(_SAMPLE_LIMIT),
                "--api-key-name", "NVIDIA_API_KEY",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"Runner exited with {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Validate output file
        results_path = output_dir / "byob_results.json"
        assert results_path.is_file(), f"Expected byob_results.json at {results_path}"

        with open(results_path) as f:
            output = json.load(f)

        assert "tasks" in output
        assert "boolq" in output["tasks"]
        scores = output["tasks"]["boolq"]["metrics"]["pass@1"]["scores"]
        assert "correct" in scores
        assert scores["correct"]["count"] == _SAMPLE_LIMIT
        assert 0 <= scores["correct"]["value"] <= 100
        assert isinstance(scores["correct"]["mean"], (int, float))
        assert isinstance(scores["correct"]["stderr"], (int, float))
        assert isinstance(scores["correct"]["stddev"], (int, float))
