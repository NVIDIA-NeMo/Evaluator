"""Adapter for EleutherAI's lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness).

Wraps lm-eval by implementing their LM abstract base class and routing
generate_until calls through our ModelClient. This gives us access to 600+
benchmarks with full observability, without modifying the lm-eval code.

Requires: pip install nemo-evaluator[lm-eval]  (i.e. lm_eval>=0.4)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nemo_evaluator.observability.types import StepRecord
from nemo_evaluator.runner.model_client import ModelClient

logger = logging.getLogger(__name__)


def _ensure_importable() -> None:
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness not found. Install via:\n"
            "  pip install lm_eval\n"
            "  or: pip install nemo-evaluator[lm-eval]"
        )


class UnsupportedTaskTypeError(RuntimeError):
    """Raised when a task requires loglikelihood which chat APIs cannot provide."""


class NelLM:
    """LM implementation that routes generate_until through our ModelClient.

    Chat-completion APIs can only generate text, not score arbitrary
    continuations. Tasks requiring loglikelihood (HellaSwag, ARC, PIQA,
    WinoGrande, BoolQ, OpenBookQA, etc.) are NOT supported and will
    raise UnsupportedTaskTypeError. Use lm-eval directly with a local
    model for those tasks.
    """

    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.call_log: list[StepRecord] = []
        self._call_idx = 0
        self._rank = 0
        self._world_size = 1
        self._loglikelihood_warned = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.client.model

    @property
    def max_length(self) -> int:
        return 4096

    def generate_until(self, requests: list) -> list[str]:
        from nemo_evaluator.runner.async_bridge import run_async

        results = []
        for req in requests:
            context = req.args[0] if hasattr(req, "args") else str(req)
            gen_kwargs = req.args[1] if hasattr(req, "args") and len(req.args) > 1 else {}
            until = gen_kwargs.get("until", [])

            resp = run_async(self.client.chat(prompt=context))
            text = resp.content

            if until:
                for stop in until:
                    if stop in text:
                        text = text[:text.index(stop)]

            step = StepRecord(
                problem_idx=self._call_idx,
                prompt=context[:500],
                model_response=resp,
                model_ms=resp.latency_ms,
            )
            self.call_log.append(step)
            self._call_idx += 1
            results.append(text)

        return results

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        if not self._loglikelihood_warned:
            self._loglikelihood_warned = True
            logger.error(
                "loglikelihood called with %d requests but chat APIs cannot "
                "compute token-level log-probabilities. Results will be INVALID. "
                "Run these tasks with a local model via lm-eval directly.",
                len(requests),
            )
        raise UnsupportedTaskTypeError(
            f"loglikelihood is not supported via chat API ({len(requests)} requests). "
            "Tasks like HellaSwag, ARC, PIQA, WinoGrande, BoolQ require logprobs. "
            "Use lm-eval directly with a local model, or filter to generate-only tasks: "
            "nel harness list --harness lm-eval --generate-only"
        )

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        raise UnsupportedTaskTypeError(
            f"loglikelihood_rolling is not supported via chat API ({len(requests)} requests). "
            "Use lm-eval directly with a local model for perplexity tasks."
        )

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return list(range(len(string.split())))

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return " ".join(str(t) for t in tokens)

    @property
    def eot_token_id(self) -> int:
        return 0

    @property
    def max_gen_toks(self) -> int:
        return self.client.max_tokens


def list_tasks() -> list[str]:
    """List all available lm-eval tasks."""
    _ensure_importable()
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    return sorted(tm.all_tasks)


def list_generate_tasks() -> list[str]:
    """List tasks that only use generate_until (compatible with chat APIs).

    Tasks requiring loglikelihood (multiple-choice ranking, perplexity)
    are excluded -- those need a local model with logprob access.
    """
    _ensure_importable()
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    generate_only = []
    for name in tm.all_tasks:
        try:
            task_dict = get_task_dict([name], tm)
            task = list(task_dict.values())[0]
            if hasattr(task, "OUTPUT_TYPE") and task.OUTPUT_TYPE == "generate_until":
                generate_only.append(name)
            elif hasattr(task, "output_type") and task.output_type == "generate_until":
                generate_only.append(name)
        except Exception:
            continue
    return sorted(generate_only)


def list_groups() -> list[str]:
    """List available task groups (e.g. 'leaderboard', 'mmlu')."""
    _ensure_importable()
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    return sorted(tm.all_groups)


@dataclass
class LMEvalResult:
    tasks: list[str]
    results: dict[str, dict[str, Any]]
    step_records: list[StepRecord]
    n_samples: dict[str, int] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)
    per_sample: dict[str, list] = field(default_factory=dict)


def run_lm_eval(
    tasks: list[str],
    client: ModelClient,
    num_fewshot: int | None = None,
    limit: int | None = None,
    log_samples: bool = True,
) -> LMEvalResult:
    """Run lm-eval benchmarks through our ModelClient.

    lm-eval owns the task definition, prompt formatting, few-shot assembly,
    and scoring. We only provide the model (NelLM) and capture everything.
    """
    _ensure_importable()
    import lm_eval

    nel_lm = NelLM(client)

    logger.info("Running lm-eval tasks: %s (fewshot=%s, limit=%s)", tasks, num_fewshot, limit)
    t0 = time.monotonic()

    try:
        eval_results = lm_eval.simple_evaluate(
            model=nel_lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=log_samples,
        )
    except UnsupportedTaskTypeError as e:
        logger.error(
            "One or more tasks require loglikelihood, which chat APIs cannot provide. "
            "Filter to generate-only tasks with: nel harness list --harness lm-eval --generate-only\n"
            "Error: %s", e,
        )
        raise

    elapsed = time.monotonic() - t0
    logger.info("Completed lm-eval in %.1fs", elapsed)

    if eval_results is None:
        return LMEvalResult(tasks=tasks, results={}, step_records=nel_lm.call_log)

    results_dict = eval_results.get("results", {})
    for task_name, metrics in results_dict.items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.info("  %s/%s = %.4f", task_name, k, v)

    n_samples = {}
    ns = eval_results.get("n-samples", {})
    for task_name, info in ns.items():
        if isinstance(info, dict):
            n_samples[task_name] = info.get("effective", info.get("original", 0))
        elif isinstance(info, int):
            n_samples[task_name] = info

    per_sample = {}
    if log_samples and "samples" in eval_results:
        per_sample = eval_results["samples"]

    return LMEvalResult(
        tasks=tasks,
        results=results_dict,
        step_records=nel_lm.call_log,
        n_samples=n_samples,
        configs=eval_results.get("configs", {}),
        per_sample=per_sample,
    )
