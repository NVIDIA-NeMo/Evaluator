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

"""Shared fixtures for nemo-skills unit tests."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
    ExecutionMode,
)


@pytest.fixture
def mock_client():
    """Mock NeMoEvaluatorClient that returns deterministic responses.

    Returns a simple mock that returns "The answer is \\boxed{42}" for all requests.
    Use this for basic unit tests that don't require complex response logic.
    """
    client = MagicMock()

    async def mock_chat_completion(messages, **kwargs):
        # Extract the user prompt content
        prompt = messages[-1]["content"] if messages else ""
        # Deterministic: always return boxed answer
        return "The answer is \\boxed{42}"

    client.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    return client


@pytest.fixture
def benchmark_data_dir(tmp_path: Path) -> Callable:
    """Create a temporary benchmark data directory with JSONL files.

    Returns a factory function that creates benchmark data on demand.

    Usage:
        data_dir = benchmark_data_dir("gsm8k", "test", [
            {"problem": "What is 2+2?", "expected_answer": "4"},
            {"problem": "What is 3*5?", "expected_answer": "15"},
        ])
        # data_dir now points to tmp_path/data
        # File exists at tmp_path/data/gsm8k/test.jsonl
    """

    def create(benchmark_name: str, split: str, samples: List[Dict[str, Any]]) -> str:
        bench_dir = tmp_path / "data" / benchmark_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = bench_dir / f"{split}.jsonl"
        with open(jsonl_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        return str(tmp_path / "data")

    return create


@pytest.fixture
def make_evaluation():
    """Factory for creating Evaluation objects with nemo-skills config.

    Returns a factory function that creates Evaluation instances with sensible
    defaults for nemo-skills testing.

    Usage:
        evaluation = make_evaluation(
            benchmark_name="gsm8k",
            eval_type="math",
            data_dir="/tmp/data",
        )
    """

    def create(
        benchmark_name: str = "gsm8k",
        eval_type: str = "math",
        data_dir: str = "/tmp/data",
        num_seeds: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        eval_split: str = "test",
        execution_mode: ExecutionMode = ExecutionMode.NATIVE,
        native_entrypoint: str = "nemo_evaluator.plugins.nemo_skills.runner:evaluate",
        output_dir: str = "/tmp/output",
    ) -> Evaluation:
        return Evaluation(
            command=None if execution_mode == ExecutionMode.NATIVE else "run_benchmark {{benchmark_name}}",
            execution_mode=execution_mode,
            framework_name="nemo_skills",
            pkg_name="nemo_skills",
            config=EvaluationConfig(
                output_dir=output_dir,
                params=ConfigParams(
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    extra={
                        "benchmark_name": benchmark_name,
                        "eval_type": eval_type,
                        "data_dir": data_dir,
                        "num_seeds": num_seeds,
                        "max_tokens": max_tokens,
                        "eval_split": eval_split,
                        "native_entrypoint": native_entrypoint,
                    },
                ),
            ),
            target=EvaluationTarget(),
        )

    return create


# Standard fixture samples for common test cases
MATH_SAMPLES = [
    {"problem": "What is 2+2?", "expected_answer": "4"},
    {"problem": "What is 3*5?", "expected_answer": "15"},
    {"problem": "What is 10/2?", "expected_answer": "5"},
]

MULTICHOICE_SAMPLES = [
    {
        "question": "Capital of France?",
        "expected_answer": "A",
        "choices": "A) Paris B) London C) Berlin",
    },
    {"question": "2+2?", "expected_answer": "B", "choices": "A) 3 B) 4 C) 5"},
]
