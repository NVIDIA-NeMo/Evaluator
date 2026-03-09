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

"""Smoketest: TruthfulQA BYOB benchmark against a live NVIDIA API endpoint.

This test calls a real model endpoint and a real judge endpoint, so it requires:
  - NVIDIA_API_KEY environment variable set (subject model)
  - JUDGE_API_KEY environment variable set (LLM-as-Judge)
  - Network access to integrate.api.nvidia.com / inference-api.nvidia.com

Run with:
  NVIDIA_API_KEY="nvapi-..." JUDGE_API_KEY="nvapi-..." \\
    pytest tests/integration/byob/test_byob_boolq_smoketest.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_evaluator.contrib.byob.eval_logic import import_benchmark, run_eval_loop
from nemo_evaluator.contrib.byob.runner import (
    aggregate_scores,
    call_model_chat,
    load_dataset,
)

_NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1"
_NVIDIA_MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
# Only evaluate a small subset to keep the smoketest fast and cheap.
_SAMPLE_LIMIT = 3
# TruthfulQA answers can be verbose; give the model enough room.
_MAX_TOKENS = 512

# A small inline TruthfulQA-shaped dataset so the smoketest does not depend
# on network access to HuggingFace.
_SAMPLE_DATA = [
    {
        "question": "What is the boiling point of water at sea level?",
        "best_answer": "100 degrees Celsius (212 degrees Fahrenheit)",
        "correct_answers": [
            "100 degrees Celsius",
            "212 degrees Fahrenheit",
            "373 Kelvin",
        ],
        "incorrect_answers": ["90 degrees Celsius", "50 degrees Celsius"],
    },
    {
        "question": "How many continents are there on Earth?",
        "best_answer": "Seven continents",
        "correct_answers": ["Seven", "7"],
        "incorrect_answers": ["Six", "Eight", "Five"],
    },
    {
        "question": "What is the chemical symbol for gold?",
        "best_answer": "Au",
        "correct_answers": ["Au"],
        "incorrect_answers": ["Go", "Gd", "Ag"],
    },
]


def _get_api_key():
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        pytest.skip("NVIDIA_API_KEY not set – skipping live endpoint smoketest")
    return key


def _get_judge_api_key():
    key = os.environ.get("JUDGE_API_KEY")
    if not key:
        pytest.skip("JUDGE_API_KEY not set – skipping judge-dependent smoketest")
    return key


def _get_truthfulqa_dir():
    """Resolve the TruthfulQA example directory."""
    test_file = Path(__file__).resolve()
    repo_root = test_file.parent.parent.parent.parent.parent.parent
    truthfulqa_dir = (
        repo_root / "packages" / "nemo-evaluator" / "examples" / "byob" / "truthfulqa"
    )
    if not truthfulqa_dir.exists():
        pytest.skip(f"TruthfulQA example directory not found at {truthfulqa_dir}")
    return truthfulqa_dir


@pytest.mark.smoketest
class TestTruthfulQASmoketest:
    """Smoketest: run TruthfulQA BYOB benchmark against a live NVIDIA endpoint."""

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

    def test_truthfulqa_eval_loop(self, tmp_path):
        """Run the TruthfulQA eval loop against the live endpoint and verify results."""
        api_key = _get_api_key()
        _get_judge_api_key()  # skip if JUDGE_API_KEY not set
        truthfulqa_dir = _get_truthfulqa_dir()

        # Write inline dataset to a temp JSONL file
        dataset_file = tmp_path / "data.jsonl"
        with open(dataset_file, "w") as f:
            for sample in _SAMPLE_DATA[:_SAMPLE_LIMIT]:
                f.write(json.dumps(sample) + "\n")

        # Import the TruthfulQA benchmark
        benchmark_path = str(truthfulqa_dir / "benchmark.py")
        bench = import_benchmark(benchmark_path, "truthfulqa")

        dataset = load_dataset(str(dataset_file), limit=_SAMPLE_LIMIT)
        assert len(dataset) == _SAMPLE_LIMIT

        def model_call_fn(
            prompt: str, endpoint_type: str, system_prompt=None, timeout=60
        ) -> str:
            return call_model_chat(
                url=_NVIDIA_API_URL,
                model_id=_NVIDIA_MODEL_ID,
                prompt=prompt,
                temperature=0.2,
                max_tokens=_MAX_TOKENS,
                api_key=api_key,
                timeout=timeout,
            )

        scores, _ = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=model_call_fn,
            endpoint_type="chat",
        )

        assert len(scores) == _SAMPLE_LIMIT, (
            f"Expected {_SAMPLE_LIMIT} scored samples, got {len(scores)}"
        )

        for i, score in enumerate(scores):
            assert "truthful" in score, f"Sample {i} missing 'truthful' key: {score}"
            assert "judge_grade" in score, (
                f"Sample {i} missing 'judge_grade' key: {score}"
            )
            assert isinstance(score["truthful"], (int, float)), (
                f"Sample {i} 'truthful' should be numeric, got {type(score['truthful'])}"
            )

        results = aggregate_scores(scores, "truthfulqa")
        assert "tasks" in results
        assert "truthfulqa" in results["tasks"]
        metrics = results["tasks"]["truthfulqa"]["metrics"]["pass@1"]["scores"]
        assert "truthful" in metrics
        assert metrics["truthful"]["stats"]["count"] == _SAMPLE_LIMIT
        assert 0 <= metrics["truthful"]["value"] <= 1.0

    def test_truthfulqa_subprocess_runner(self, tmp_path):
        """Run the TruthfulQA benchmark via the subprocess runner CLI."""
        _get_api_key()
        _get_judge_api_key()  # skip if JUDGE_API_KEY not set
        truthfulqa_dir = _get_truthfulqa_dir()

        # Write inline dataset to a temp JSONL file
        dataset_file = tmp_path / "data.jsonl"
        with open(dataset_file, "w") as f:
            for sample in _SAMPLE_DATA[:_SAMPLE_LIMIT]:
                f.write(json.dumps(sample) + "\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nemo_evaluator.contrib.byob.runner",
                "--benchmark-module",
                str(truthfulqa_dir / "benchmark.py"),
                "--benchmark-name",
                "truthfulqa",
                "--dataset",
                str(dataset_file),
                "--output-dir",
                str(output_dir),
                "--model-url",
                _NVIDIA_API_URL,
                "--model-id",
                _NVIDIA_MODEL_ID,
                "--model-type",
                "chat",
                "--temperature",
                "0.2",
                "--max-tokens",
                str(_MAX_TOKENS),
                "--limit-samples",
                str(_SAMPLE_LIMIT),
                "--api-key-name",
                "NVIDIA_API_KEY",
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

        results_path = output_dir / "byob_results.json"
        assert results_path.is_file(), f"Expected byob_results.json at {results_path}"

        with open(results_path) as f:
            output = json.load(f)

        assert "tasks" in output
        assert "truthfulqa" in output["tasks"]
        scores = output["tasks"]["truthfulqa"]["metrics"]["pass@1"]["scores"]
        assert "truthful" in scores
        assert scores["truthful"]["stats"]["count"] == _SAMPLE_LIMIT
        assert 0 <= scores["truthful"]["value"] <= 1.0
        assert isinstance(scores["truthful"]["stats"]["mean"], (int, float))
        assert isinstance(scores["truthful"]["stats"]["stderr"], (int, float))
        assert isinstance(scores["truthful"]["stats"]["stddev"], (int, float))
