"""Adapter for NeMo Skills benchmarks.

Loads datasets and scoring logic from nemo_skills directly, giving full
per-request observability (trajectories, latency, token counts, etc.) for
all 85+ Skills benchmarks.

Requires: pip install nemo-skills
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nemo_evaluator.adapters.base import EnvironmentAdapter
from nemo_evaluator.environments.base import SeedResult, VerifyResult

logger = logging.getLogger(__name__)


def _require_skills():
    try:
        from nemo_skills.dataset import utils as ds_utils
        from nemo_skills.evaluation.evaluator import EVALUATOR_CLASS_MAP_PATHS
        return ds_utils, EVALUATOR_CLASS_MAP_PATHS
    except ImportError:
        raise ImportError(
            "SkillsAdapter requires 'nemo-skills'. "
            "Install: pip install nemo-skills  or  pip install -e /path/to/Skills"
        )


def _load_dataset(benchmark: str, split: str | None, data_dir: str | None) -> list[dict[str, Any]]:
    ds_utils, _ = _require_skills()
    module = ds_utils.get_dataset_module(benchmark)

    resolved_split = split or getattr(module, "EVAL_SPLIT", "test")
    resolved_dir = data_dir or ds_utils.get_default_data_dir()

    bench_path = Path(resolved_dir) / benchmark.replace(".", "/")
    data_file = bench_path / f"{resolved_split}.jsonl"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}. "
            f"Run: ns prepare_data {benchmark}  to download it."
        )

    samples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info("Loaded %d samples from %s (split=%s)", len(samples), benchmark, resolved_split)
    return samples


def _get_evaluator(eval_type: str):
    """Get an evaluator class instance for the given eval_type."""
    try:
        from nemo_skills.evaluation.evaluator import get_evaluator_class
        return get_evaluator_class(eval_type)
    except Exception:
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


class SkillsAdapter(EnvironmentAdapter):
    """Wraps a NeMo Skills benchmark as an EvalSource.

    Evaluator owns the model call, so you get full trajectories,
    per-request latency, token counts, failure analysis, etc.
    """

    def __init__(
        self,
        benchmark: str,
        split: str | None = None,
        data_dir: str | None = None,
        prompt_template: str | None = None,
        eval_type: str | None = None,
    ) -> None:
        self._benchmark = benchmark
        self.name = f"skills:{benchmark}"

        ds_utils, _ = _require_skills()
        self._module = ds_utils.get_dataset_module(benchmark)
        self._samples = _load_dataset(benchmark, split, data_dir)

        self._eval_type = eval_type or getattr(self._module, "METRICS_TYPE", "math")
        self._evaluator = _get_evaluator(self._eval_type)
        self._prompt_template = prompt_template

        gen_args = getattr(self._module, "GENERATION_ARGS", "")
        self._prompt_config = None
        if "prompt_config=" in gen_args:
            self._prompt_config = gen_args.split("prompt_config=")[1].split()[0]

        logger.info(
            "Skills adapter: %s (%d samples, eval_type=%s)",
            benchmark, len(self._samples), self._eval_type,
        )

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

    async def verify(self, response: str, expected: str, **meta: Any) -> VerifyResult:
        reward, details = self._score(response, expected, meta)
        return VerifyResult(
            reward=reward,
            extracted_answer=details.get("extracted", response.strip()[:200]),
            scoring_details=details,
        )

    def _score(self, response: str, expected: str, meta: dict[str, Any]) -> tuple[float, dict]:
        if self._eval_type == "math":
            return self._score_math(response, expected)
        elif self._eval_type == "multichoice":
            return self._score_multichoice(response, expected, meta)
        elif self._evaluator is not None:
            return self._score_via_evaluator(response, expected, meta)
        else:
            return self._score_exact(response, expected)

    def _score_math(self, response: str, expected: str) -> tuple[float, dict]:
        try:
            from nemo_skills.evaluation.evaluator.math_grader import extract_answer, math_equal
            extracted = extract_answer(response)
            correct = math_equal(extracted, expected)
            return (
                1.0 if correct else 0.0,
                {"method": "skills_math_equal", "extracted": extracted, "match": correct},
            )
        except ImportError:
            from nemo_evaluator.scoring import extract_answer, math_equal
            extracted = extract_answer(response)
            correct = math_equal(extracted, expected)
            return (
                1.0 if correct else 0.0,
                {"method": "nel_math_equal_fallback", "extracted": extracted, "match": correct},
            )

    def _score_multichoice(self, response: str, expected: str,
                           meta: dict[str, Any]) -> tuple[float, dict]:
        try:
            from nemo_skills.evaluation.evaluator.math_grader import extract_answer
            extracted = extract_answer(response)
        except ImportError:
            extracted = response.strip().upper()[:1]

        norm_extracted = extracted.strip().upper()
        norm_expected = expected.strip().upper()
        correct = norm_extracted == norm_expected
        return (
            1.0 if correct else 0.0,
            {"method": "multichoice", "extracted": extracted, "match": correct},
        )

    def _score_via_evaluator(self, response: str, expected: str,
                             meta: dict[str, Any]) -> tuple[float, dict]:
        try:
            sample = {"predicted_answer": response, "expected_answer": expected}
            result = self._evaluator.eval_single(sample)
            reward = float(result.get("is_correct", 0))
            return reward, {"method": f"skills_{self._eval_type}", **result}
        except Exception as e:
            logger.warning("Skills evaluator failed, falling back: %s", e)
            return self._score_exact(response, expected)

    def _score_exact(self, response: str, expected: str) -> tuple[float, dict]:
        correct = response.strip().lower() == expected.strip().lower()
        return (
            1.0 if correct else 0.0,
            {"method": "exact_match_fallback", "match": correct},
        )

    async def dataset_size(self) -> int:
        return len(self._samples)

    @property
    def eval_type(self) -> str:
        return self._eval_type

    @property
    def benchmark(self) -> str:
        return self._benchmark


def list_skills_benchmarks(data_dir: str | None = None) -> list[dict[str, str]]:
    """List available Skills benchmarks that have data prepared."""
    ds_utils, _ = _require_skills()
    resolved_dir = data_dir or ds_utils.get_default_data_dir()
    base = Path(resolved_dir)

    benchmarks = []
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.is_dir():
                jsonls = list(p.glob("*.jsonl"))
                if jsonls:
                    module = None
                    try:
                        module = ds_utils.get_dataset_module(p.name)
                    except Exception:
                        pass
                    metrics_type = getattr(module, "METRICS_TYPE", "unknown") if module else "unknown"
                    benchmarks.append({
                        "benchmark": p.name,
                        "splits": [f.stem for f in jsonls],
                        "metrics_type": metrics_type,
                    })
    return benchmarks
