# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""NeMo Skills benchmarks as EvalEnvironments."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


def _require_skills():
    try:
        from nemo_skills.dataset import utils as ds_utils

        return ds_utils
    except ImportError:
        raise ImportError(
            "SkillsEnvironment requires 'nemo-skills'. "
            "Install: pip install nemo-skills  or  pip install -e /path/to/Skills"
        )


def _unpack_module(result):
    if isinstance(result, tuple):
        return result
    return result, None


def _prepare_dataset(benchmark: str) -> None:
    logger.info("Dataset for %s not found, preparing automatically...", benchmark)
    result = subprocess.run(
        [sys.executable, "-m", "nemo_skills.dataset.prepare", benchmark],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare dataset {benchmark!r}: {result.stderr[-500:]}")
    logger.info("Dataset %s prepared successfully", benchmark)


def _resolve_data_dir(ds_utils, default_dir: str | None) -> str:
    if default_dir and Path(default_dir).exists():
        return default_dir
    pkg_dir = Path(ds_utils.__file__).parent
    if pkg_dir.exists():
        return str(pkg_dir)
    candidate = ds_utils.get_default_data_dir()
    if Path(candidate).exists():
        return candidate
    return str(pkg_dir)


def _load_dataset(benchmark: str, split: str | None, data_dir: str | None) -> list[dict[str, Any]]:
    ds_utils = _require_skills()
    module, default_dir = _unpack_module(ds_utils.get_dataset_module(benchmark))
    resolved_split = split or getattr(module, "EVAL_SPLIT", "test")
    resolved_dir = data_dir or _resolve_data_dir(ds_utils, default_dir)

    bench_path = Path(resolved_dir) / benchmark.replace(".", "/")
    data_file = bench_path / f"{resolved_split}.jsonl"

    if not data_file.exists():
        _prepare_dataset(benchmark)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found after preparation: {data_file}")

    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d samples from %s (split=%s)", len(samples), benchmark, resolved_split)
    return samples


def _get_evaluator(eval_type: str):
    try:
        from nemo_skills.evaluation.evaluator import get_evaluator_class

        return get_evaluator_class(eval_type, config={})
    except (ImportError, KeyError, ValueError, TypeError):
        return None


def _get_prompt_field(sample: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt", "input", "text"):
        if key in sample:
            return str(sample[key])
    return str(sample)


def _get_answer_field(sample: dict[str, Any]) -> str:
    for key in ("expected_answer", "answer", "solution", "target", "label"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return ""


class SkillsEnvironment(EvalEnvironment):
    """Wraps a NeMo Skills benchmark."""

    def __init__(
        self,
        benchmark: str,
        split: str | None = None,
        data_dir: str | None = None,
        prompt_template: str | None = None,
        eval_type: str | None = None,
    ) -> None:
        super().__init__()
        self._benchmark = benchmark
        self.name = f"skills:{benchmark}"

        ds_utils = _require_skills()
        self._module, _ = _unpack_module(ds_utils.get_dataset_module(benchmark))
        self._samples = _load_dataset(benchmark, split, data_dir)
        self._dataset = self._samples

        self._eval_type = eval_type or getattr(self._module, "METRICS_TYPE", "math")
        self._evaluator = _get_evaluator(self._eval_type)
        self._prompt_template = prompt_template

        logger.info("Skills: %s (%d samples, eval_type=%s)", benchmark, len(self._samples), self._eval_type)

    def __len__(self) -> int:
        return len(self._samples)

    async def seed(self, idx: int) -> SeedResult:
        sample = self._samples[idx]
        prompt = _get_prompt_field(sample)
        expected = _get_answer_field(sample)

        if self._prompt_template:
            prompt = self._prompt_template.format(problem=prompt, question=prompt)

        meta: dict[str, Any] = {
            "source": "nemo_skills",
            "benchmark": self._benchmark,
            "eval_type": self._eval_type,
        }
        for k in ("category", "subject", "difficulty", "level", "type", "domain", "subset"):
            if k in sample:
                meta[k] = sample[k]
        if "choices" in sample:
            meta["choices"] = sample["choices"]
        return SeedResult(prompt=prompt, expected_answer=expected, metadata=meta)

    async def verify(self, response: str, expected: str, sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        reward, details = self._score(response, expected, meta)
        return VerifyResult(
            reward=reward,
            extracted_answer=details.get("extracted", response.strip()[:200]),
            scoring_details=details,
        )

    async def dataset_size(self) -> int:
        return len(self._samples)

    def _score(self, response: str, expected: str, meta: dict[str, Any]) -> tuple[float, dict]:
        handler = self._score_handlers.get(self._eval_type)
        if handler:
            return handler(self, response, expected, meta)
        if self._evaluator is not None:
            return self._score_via_evaluator(response, expected, meta)
        return self._score_exact(response, expected)

    def _score_math(self, response: str, expected: str, meta: dict[str, Any] = None) -> tuple[float, dict]:
        from nemo_skills.evaluation.evaluator.math_grader import extract_answer, math_equal

        extracted = extract_answer(response)
        correct = math_equal(extracted, expected)
        return (1.0 if correct else 0.0, {"method": "skills_math_equal", "extracted": extracted, "match": correct})

    def _score_multichoice(self, response: str, expected: str, meta: dict[str, Any]) -> tuple[float, dict]:
        try:
            from nemo_skills.evaluation.evaluator.math_grader import extract_answer

            extracted = extract_answer(response)
        except ImportError:
            extracted = response.strip().upper()[:1]
        correct = extracted.strip().upper() == expected.strip().upper()
        return (1.0 if correct else 0.0, {"method": "multichoice", "extracted": extracted, "match": correct})

    def _score_via_evaluator(self, response: str, expected: str, meta: dict[str, Any]) -> tuple[float, dict]:
        try:
            sample = {"predicted_answer": response, "expected_answer": expected}
            result = self._evaluator.eval_single(sample)
            return float(result.get("is_correct", 0)), {"method": f"skills_{self._eval_type}", **result}
        except Exception as e:
            logger.warning("Skills evaluator failed: %s", e)
            return self._score_exact(response, expected)

    def _score_exact(self, response: str, expected: str) -> tuple[float, dict]:
        correct = response.strip().lower() == expected.strip().lower()
        return (1.0 if correct else 0.0, {"method": "exact_match_fallback", "match": correct})

    _score_handlers: dict[str, Any] = {
        "math": _score_math,
        "multichoice": _score_multichoice,
    }

    @property
    def eval_type(self) -> str:
        return self._eval_type

    @property
    def benchmark(self) -> str:
        return self._benchmark


def list_skills_benchmarks(data_dir: str | None = None) -> list[dict[str, str]]:
    ds_utils = _require_skills()
    resolved_dir = data_dir or _resolve_data_dir(ds_utils, None)
    base = Path(resolved_dir)
    benchmarks = []
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_dir():
                jsonls = list(p.glob("*.jsonl"))
                if jsonls:
                    module = None
                    try:
                        module, _ = _unpack_module(ds_utils.get_dataset_module(p.name))
                    except (ImportError, KeyError, ValueError, AttributeError):
                        pass
                    metrics_type = getattr(module, "METRICS_TYPE", "unknown") if module else "unknown"
                    benchmarks.append(
                        {"benchmark": p.name, "splits": [f.stem for f in jsonls], "metrics_type": metrics_type}
                    )
    return benchmarks
