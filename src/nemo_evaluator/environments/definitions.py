"""Benchmark definition API: @benchmark + @scorer.

All benchmarks -- built-in and user-provided (BYOB) -- use this API.

    from nemo_evaluator import benchmark, scorer, ScorerInput

    @benchmark(name="my-bench", dataset="hf://my/data?split=test",
               prompt="Question: {q}\\nAnswer:", target_field="answer")
    @scorer
    def score(sample: ScorerInput) -> dict:
        return {"correct": sample.response.strip() == str(sample.target)}

Extension hooks:
  - prepare_row(row, idx, rng) -> row:  transform each dataset row after loading
  - seed_fn(row, idx) -> SeedResult:    fully custom seed (overrides prompt template)

Dataset specs:
  - "hf://dataset?split=test&config=cfg"  (HuggingFace)
  - "path/to/data.jsonl"                  (local JSONL)
  - callable returning list[dict]         (programmatic)
"""
from __future__ import annotations

import json
import logging
import random
import re
import string
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class ScorerInput:
    """Passed to scorer functions."""
    response: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkDefinition:
    name: str
    dataset: str | Callable[[], list[dict]]
    prompt: str
    target_field: str = "target"
    endpoint_type: str = "chat"
    system_prompt: str | None = None
    field_mapping: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    requirements: list[str] | None = None
    scorer_fn: Callable[[ScorerInput], dict] | None = None
    prepare_row: Callable[[dict, int, random.Random], dict] | None = None
    seed_fn: Callable[[dict, int], SeedResult] | None = None


_BYOB_REGISTRY: dict[str, BenchmarkDefinition] = {}


# ── Dataset loading ───────────────────────────────────────────────────────

def _load_dataset_from_spec(spec: str | Callable) -> list[dict[str, Any]]:
    if callable(spec):
        return spec()

    if spec.startswith("hf://"):
        return _load_hf(spec[5:])

    path = Path(spec)
    if path.exists():
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    # Try as HF dataset name directly
    return _load_hf(spec)


def _load_hf(spec: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    parts = spec.split("?")
    dataset_name = parts[0]
    params: dict[str, str] = {}
    if len(parts) > 1:
        for kv in parts[1].split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k] = v

    split = params.get("split", "test")
    config = params.get("config")

    args = [dataset_name]
    if config:
        args.append(config)
    ds = load_dataset(*args, split=split)
    return [dict(row) for row in ds]


def _format_prompt(template: str, row: dict, field_mapping: dict | None = None) -> str:
    data = dict(row)
    if field_mapping:
        for src, dst in field_mapping.items():
            if src in data:
                data[dst] = data[src]
    if template.endswith((".txt", ".md", ".jinja")) and Path(template).exists():
        template = Path(template).read_text()
    try:
        return template.format(**data)
    except KeyError as e:
        logger.warning("Prompt template has unknown field %s (available: %s)", e, list(data.keys()))
        return template


# ── Environment class ─────────────────────────────────────────────────────

class ByobEnvironment(EvalEnvironment):
    """Auto-generated from @benchmark + @scorer."""

    def __init__(self, definition: BenchmarkDefinition, num_examples: int | None = None) -> None:
        super().__init__()
        self._defn = definition
        self.name = definition.name

        raw = _load_dataset_from_spec(definition.dataset)
        if num_examples:
            raw = raw[:num_examples]

        rng = random.Random(42)
        if definition.prepare_row:
            raw = [definition.prepare_row(row, i, rng) for i, row in enumerate(raw)]

        self._dataset = raw
        logger.info("BYOB %s: %d samples", definition.name, len(raw))

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]

        if self._defn.seed_fn:
            return self._defn.seed_fn(row, idx)

        prompt = _format_prompt(self._defn.prompt, row, self._defn.field_mapping)
        target = str(row.get(self._defn.target_field, ""))

        meta: dict[str, Any] = {"source": "byob", "benchmark": self._defn.name}
        for k, v in row.items():
            if k != self._defn.target_field:
                meta[k] = v

        messages = [{"role": "user", "content": prompt}]
        if self._defn.system_prompt:
            messages.insert(0, {"role": "system", "content": self._defn.system_prompt})

        return SeedResult(prompt=prompt, expected_answer=target, messages=messages,
                          system=self._defn.system_prompt, metadata=meta)

    async def verify(self, response: str, expected: str, **meta: Any) -> VerifyResult:
        if self._defn.scorer_fn is None:
            correct = response.strip().lower() == expected.strip().lower()
            return VerifyResult(reward=1.0 if correct else 0.0,
                                extracted_answer=response.strip()[:200],
                                scoring_details={"method": "default_exact_match"})

        sample = ScorerInput(response=response, target=expected, metadata=meta,
                             config=self._defn.extra)
        scores = self._defn.scorer_fn(sample)

        reward = float(scores.get("correct", scores.get("reward", next(iter(scores.values()), 0))))
        return VerifyResult(
            reward=reward,
            extracted_answer=scores.get("extracted", response.strip()[:200]),
            scoring_details={"method": f"byob_{self._defn.name}", **scores},
        )


# ── Decorators ────────────────────────────────────────────────────────────

def benchmark(
    name: str,
    dataset: str | Callable,
    prompt: str = "",
    target_field: str = "target",
    endpoint_type: str = "chat",
    system_prompt: str | None = None,
    field_mapping: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    requirements: list[str] | None = None,
    prepare_row: Callable | None = None,
    seed_fn: Callable | None = None,
    **kwargs,
):
    """Register a benchmark. Decorate a scorer function."""
    defn = BenchmarkDefinition(
        name=name, dataset=dataset, prompt=prompt,
        target_field=target_field, endpoint_type=endpoint_type,
        system_prompt=system_prompt, field_mapping=field_mapping,
        extra={**(extra or {}), **kwargs}, requirements=requirements,
        prepare_row=prepare_row, seed_fn=seed_fn,
    )

    def decorator(fn):
        defn.scorer_fn = fn
        _BYOB_REGISTRY[name] = defn

        @register(name)
        class _Env(ByobEnvironment):
            def __init__(self, num_examples: int | None = None):
                super().__init__(defn, num_examples)

        _Env.__name__ = f"Bench_{name}"
        _Env.__qualname__ = f"Bench_{name}"
        return fn

    return decorator


def scorer(fn: Callable[[ScorerInput], dict]) -> Callable[[ScorerInput], dict]:
    """Marks a function as a scorer."""
    fn._is_scorer = True
    return fn


# ── Built-in scoring primitives ──────────────────────────────────────────

def multichoice_regex(sample: ScorerInput, pattern: str = r"(?i)Answer\s*:\s*([A-D])") -> dict:
    m = re.search(pattern, sample.response)
    extracted = m.group(1).upper() if m else None
    return {"correct": extracted == str(sample.target).upper(), "extracted": extracted}


def answer_line(sample: ScorerInput, pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> dict:
    m = re.search(pattern, sample.response)
    extracted = m.group(1).strip() if m else sample.response.strip().split("\n")[-1]
    return {"correct": _normalize_math(extracted) == _normalize_math(str(sample.target)),
            "extracted": extracted}


def fuzzy_match(sample: ScorerInput) -> dict:
    pred = _normalize_text(sample.response)
    targets = sample.metadata.get("correct_answers", [str(sample.target)])
    correct = any(_normalize_text(t) in pred or pred in _normalize_text(t) for t in targets)
    return {"correct": correct, "extracted": sample.response.strip()[:200]}


def exact_match(sample: ScorerInput) -> dict:
    return {"correct": _normalize_text(sample.response) == _normalize_text(str(sample.target))}


def numeric_match(sample: ScorerInput) -> dict:
    nums = re.findall(r"-?\d+\.?\d*", sample.response.replace(",", ""))
    extracted = nums[-1].rstrip("0").rstrip(".") if nums else ""
    target = str(sample.target).strip().rstrip("0").rstrip(".")
    return {"correct": extracted == target, "extracted": extracted}


def code_sandbox(sample: ScorerInput) -> dict:
    """Run code in Docker sandbox. Pipes code via stdin to avoid shell injection."""
    import subprocess

    prompt_code = sample.metadata.get("_prompt", "")
    test_code = sample.metadata.get("_test", "")
    entry_point = sample.metadata.get("entry_point", "solution")

    m = re.search(r"```(?:python)?\s*\n((?:\n|.)+?)```", sample.response, re.DOTALL)
    completion = m.group(1) if m else sample.response

    code = f"{prompt_code}{completion}\n{test_code}\ncheck({entry_point})"
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "-i",
             "--network", "none",
             "--memory", "256m",
             "--cpus", "1",
             "--pids-limit", "64",
             "--read-only",
             "--tmpfs", "/tmp:size=64m",
             "--security-opt", "no-new-privileges",
             "python:3.12-slim", "python", "-"],
            input=code, capture_output=True, text=True, timeout=30,
        )
        passed = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        passed = False

    return {"correct": passed, "extracted": completion[:500]}


def needs_judge(sample: ScorerInput) -> dict:
    """Signals that this sample needs LLM-as-judge scoring (eval loop post-processor)."""
    return {"correct": False, "needs_judge": True, "extracted": sample.response[:500]}


# ── Helpers ───────────────────────────────────────────────────────────────

def _normalize_math(s: str) -> str:
    s = s.strip().lower()
    for ch in (",", "$", "%", " "):
        s = s.replace(ch, "")
    return s.rstrip(".")


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())
