"""Adapter for OpenAI's simple-evals (https://github.com/openai/simple-evals).

Wraps simple-evals by implementing their SamplerBase interface and routing
all model calls through our ModelClient. This gives us full observability
(trajectories, token counts, latency) without modifying the simple-evals code.

Requires: pip install nemo-evaluator[simple-evals]  (or clone the repo)
"""
from __future__ import annotations

import importlib
import inspect
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_evaluator.observability.types import StepRecord
from nemo_evaluator.runner.model_client import ModelClient

logger = logging.getLogger(__name__)

EVAL_REGISTRY: dict[str, dict[str, Any]] = {
    "mmlu": {"module": "mmlu_eval", "class": "MMLUEval", "kwargs_key": "num_examples"},
    "math": {"module": "math_eval", "class": "MathEval", "kwargs_key": "num_examples"},
    "gpqa": {"module": "gpqa_eval", "class": "GPQAEval", "kwargs_key": "num_examples"},
    "mgsm": {"module": "mgsm_eval", "class": "MGSMEval", "kwargs_key": "num_examples_per_lang"},
    "drop": {"module": "drop_eval", "class": "DropEval", "kwargs_key": "num_examples"},
    "humaneval": {"module": "humaneval_eval", "class": "HumanEval", "kwargs_key": "num_examples"},
    "simpleqa": {"module": "simpleqa_eval", "class": "SimpleQAEval", "kwargs_key": "num_examples"},
    "browsecomp": {"module": "browsecomp_eval", "class": "BrowseCompEval", "kwargs_key": "num_examples"},
}


def _ensure_importable(repo_path: str | None = None) -> None:
    """Make sure simple-evals is importable. Tries package import first, then sys.path."""
    try:
        importlib.import_module("simple_evals.types")
        return
    except ImportError:
        pass

    if repo_path:
        p = str(Path(repo_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
        return

    for candidate in [
        Path.cwd() / "simple-evals",
        Path.cwd() / "simple-evals-original",
        Path.home() / "simple-evals",
    ]:
        if (candidate / "types.py").exists():
            sys.path.insert(0, str(candidate))
            return

    raise ImportError(
        "simple-evals not found. Install via:\n"
        "  pip install simple-evals  (if packaged)\n"
        "  git clone https://github.com/openai/simple-evals.git\n"
        "  Then pass --repo-path <path> when running nel harness"
    )


class NelSampler:
    """SamplerBase implementation that routes through our ModelClient.

    simple-evals calls __call__(messages) for every model interaction.
    We forward to ModelClient, capture full ModelResponse, and return
    what simple-evals expects (SamplerResponse).
    """

    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.call_log: list[StepRecord] = []
        self._call_idx = 0

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            return {"role": role, "content": content}
        return {"role": role, "content": content}

    def __call__(self, message_list: list[dict[str, Any]]) -> Any:
        from simple_evals.types import SamplerResponse

        from nemo_evaluator.runner.async_bridge import run_async

        msgs = [{"role": m["role"], "content": m["content"]} for m in message_list]
        resp = run_async(self.client.chat(messages=msgs))

        step = StepRecord(
            problem_idx=self._call_idx,
            prompt=msgs[-1].get("content", "") if msgs else "",
            model_response=resp,
            model_ms=resp.latency_ms,
        )
        self.call_log.append(step)
        self._call_idx += 1

        return SamplerResponse(
            response_text=resp.content,
            actual_queried_message_list=message_list,
            response_metadata={
                "model": resp.model,
                "tokens": resp.total_tokens,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "latency_ms": resp.latency_ms,
                "finish_reason": resp.finish_reason,
            },
        )


def list_evals(repo_path: str | None = None) -> list[dict[str, str]]:
    """List available simple-evals benchmarks."""
    return [
        {"name": name, "module": info["module"], "class": info["class"]}
        for name, info in EVAL_REGISTRY.items()
    ]


def _load_eval(
    eval_name: str,
    repo_path: str | None = None,
    num_examples: int | None = None,
    n_repeats: int | None = None,
    grader_sampler: Any | None = None,
    equality_checker: Any | None = None,
) -> Any:
    """Instantiate a simple-evals Eval object."""
    _ensure_importable(repo_path)

    if eval_name not in EVAL_REGISTRY:
        available = ", ".join(sorted(EVAL_REGISTRY))
        raise ValueError(f"Unknown simple-eval: {eval_name!r}. Available: {available}")

    info = EVAL_REGISTRY[eval_name]
    mod = importlib.import_module(f"simple_evals.{info['module']}")
    cls = getattr(mod, info["class"])

    kwargs: dict[str, Any] = {}
    if num_examples is not None:
        kwargs[info["kwargs_key"]] = num_examples

    sig = inspect.signature(cls.__init__)
    if "n_repeats" in sig.parameters and n_repeats is not None:
        kwargs["n_repeats"] = n_repeats
    if "equality_checker" in sig.parameters and equality_checker is not None:
        kwargs["equality_checker"] = equality_checker
    if "grader_model" in sig.parameters and grader_sampler is not None:
        kwargs["grader_model"] = grader_sampler

    return cls(**kwargs)


@dataclass
class SimpleEvalsResult:
    eval_name: str
    score: float | None
    metrics: dict[str, float]
    n_samples: int
    step_records: list[StepRecord]
    convos: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] | None = None


def run_simple_eval(
    eval_name: str,
    client: ModelClient,
    repo_path: str | None = None,
    num_examples: int | None = None,
    n_repeats: int = 1,
    grader_client: ModelClient | None = None,
) -> SimpleEvalsResult:
    """Run a simple-evals benchmark through our ModelClient.

    The eval harness owns the loop, scoring, and prompt formatting.
    We only provide the model (NelSampler) and capture everything.
    """
    _ensure_importable(repo_path)

    grader_sampler = NelSampler(grader_client) if grader_client else None
    equality_checker = grader_sampler

    eval_obj = _load_eval(
        eval_name,
        repo_path=repo_path,
        num_examples=num_examples,
        n_repeats=n_repeats,
        grader_sampler=grader_sampler,
        equality_checker=equality_checker,
    )

    sampler = NelSampler(client)

    logger.info("Running simple-evals/%s (examples=%s)", eval_name, num_examples)
    t0 = time.monotonic()
    result = eval_obj(sampler)
    elapsed = time.monotonic() - t0
    logger.info("Completed simple-evals/%s in %.1fs, score=%.4f",
                eval_name, elapsed, result.score or 0)

    return SimpleEvalsResult(
        eval_name=eval_name,
        score=result.score,
        metrics=result.metrics or {},
        n_samples=len(result.htmls) if result.htmls else 0,
        step_records=sampler.call_log,
        convos=result.convos or [],
        metadata=result.metadata,
    )
