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
"""Inspect AI exporter: produce inspect_ai-compatible evaluation logs.

Generates ``.json`` (or ``.eval``) log files that can be opened with
``inspect view`` for interactive browsing of samples, messages, scores,
and model usage.

Requires ``inspect_ai`` as an optional dependency::

    pip install nemo-evaluator[inspect]
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _try_import_inspect() -> None:
    try:
        import inspect_ai.log  # noqa: F401
    except ImportError:
        raise ImportError(
            "inspect_ai is required for the 'inspect' exporter. Install with: pip install nemo-evaluator[inspect]"
        ) from None


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _atif_steps_to_messages(trajectory: list[dict[str, Any]]) -> list[Any]:
    """Convert ATIF trajectory steps into inspect_ai ChatMessage objects."""
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageUser,
    )

    messages: list[Any] = []
    for doc in trajectory:
        for step in doc.get("steps", []):
            source = step.get("source", "")
            text = step.get("message", "")
            if source == "system":
                messages.append(ChatMessageSystem(content=text))
            elif source == "user":
                messages.append(ChatMessageUser(content=text))
            elif source == "agent":
                messages.append(ChatMessageAssistant(content=text))
    return messages


def _build_model_output(result: dict[str, Any], model_name: str) -> Any:
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ModelOutput,
        ModelUsage,
    )

    response_text = result.get("model_response", "") or ""
    tokens = result.get("tokens", 0) or 0
    latency = result.get("latency_ms", 0) or 0

    usage = ModelUsage(
        input_tokens=0,
        output_tokens=tokens,
        total_tokens=tokens,
    )

    choice = ChatCompletionChoice(
        message=ChatMessageAssistant(content=response_text),
        stop_reason="stop",
    )

    return ModelOutput(
        model=model_name,
        choices=[choice],
        completion=response_text,
        usage=usage,
        time=latency / 1000.0 if latency else None,
    )


def _build_sample(
    result: dict[str, Any],
    model_name: str,
) -> Any:
    """Build an inspect_ai EvalSample from one NEL result dict."""
    from inspect_ai.log import EvalSample
    from inspect_ai.model import ModelUsage
    from inspect_ai.scorer import Score

    sample_id = result.get("problem_idx", 0)
    epoch = result.get("repeat", 0) + 1
    prompt = result.get("metadata", {}).get("prompt", "")
    if not prompt:
        prompt = result.get("expected_answer", "")
    target = result.get("expected_answer", "")

    trajectory = result.get("trajectory", [])
    messages = _atif_steps_to_messages(trajectory) if trajectory else []

    output = _build_model_output(result, model_name)

    reward = result.get("reward", 0.0)
    extracted = result.get("extracted_answer")
    scores: dict[str, Score] = {}

    scores["default"] = Score(
        value=reward,
        answer=str(extracted) if extracted is not None else None,
        explanation=None,
    )

    scorer_rewards = result.get("scorer_rewards", {})
    for scorer_name, scorer_val in scorer_rewards.items():
        scores[scorer_name] = Score(value=scorer_val)

    tokens = result.get("tokens", 0) or 0
    model_usage: dict[str, ModelUsage] = {}
    if tokens:
        model_usage[model_name] = ModelUsage(
            input_tokens=0,
            output_tokens=tokens,
            total_tokens=tokens,
        )

    metadata = dict(result.get("metadata", {}))
    scoring_details = result.get("scoring_details", {})
    if scoring_details:
        metadata["scoring_details"] = scoring_details

    return EvalSample(
        id=sample_id,
        epoch=epoch,
        input=prompt or "(no prompt)",
        target=target,
        messages=messages,
        output=output,
        scores=scores,
        metadata=metadata,
        model_usage=model_usage,
    )


def _build_eval_log(bundle: dict[str, Any]) -> Any:
    """Build an inspect_ai EvalLog from a NEL bundle (with _results)."""
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalMetric,
        EvalPlan,
        EvalPlanStep,
        EvalResults,
        EvalScore,
        EvalSpec,
        EvalStats,
    )

    bm = bundle.get("benchmark", {})
    cfg = bundle.get("config", {})
    bench_name = bm.get("name", "unknown")
    model_name = cfg.get("model", "unknown")
    run_id = bundle.get("run_id", f"nel-{bench_name}")
    timestamp = bundle.get("timestamp", _iso_now())

    results_list: list[dict[str, Any]] = bundle.get("_results", [])

    n_samples = bm.get("samples", len(results_list))

    eval_spec = EvalSpec(
        eval_id=run_id,
        run_id=run_id,
        created=timestamp,
        task=bench_name,
        task_id=bench_name,
        task_version=0,
        task_file=None,
        model=model_name,
        model_base_url=cfg.get("base_url"),
        dataset=EvalDataset(
            name=bench_name,
            samples=n_samples,
        ),
        config=EvalConfig(),
        packages={"nemo_evaluator": bundle.get("sdk_version", "0.0.0")},
    )

    solver_name = cfg.get("harbor_agent") or cfg.get("endpoint_type", "nel")
    plan = EvalPlan(
        name="nel",
        steps=[EvalPlanStep(solver=solver_name)],
    )

    nel_scores = bm.get("scores", {})
    eval_scores: list[EvalScore] = []
    for metric_name, metric_val in nel_scores.items():
        if isinstance(metric_val, dict) and "value" in metric_val:
            eval_scores.append(
                EvalScore(
                    name=metric_name,
                    scorer=metric_name,
                    metrics={
                        metric_name: EvalMetric(
                            name=metric_name,
                            value=metric_val["value"],
                        )
                    },
                )
            )
        elif isinstance(metric_val, (int, float)):
            eval_scores.append(
                EvalScore(
                    name=metric_name,
                    scorer=metric_name,
                    metrics={
                        metric_name: EvalMetric(
                            name=metric_name,
                            value=metric_val,
                        )
                    },
                )
            )

    eval_results = EvalResults(
        total_samples=n_samples,
        completed_samples=len(results_list),
        scores=eval_scores,
    )

    stats = EvalStats(
        started_at=timestamp,
        completed_at=_iso_now(),
    )

    is_failed = bundle.get("_failed", False)
    status = "error" if is_failed else "success"

    samples = [_build_sample(r, model_name) for r in results_list]

    return EvalLog(
        version=2,
        status=status,
        eval=eval_spec,
        plan=plan,
        results=eval_results,
        stats=stats,
        samples=samples if samples else None,
    )


class InspectExporter:
    """Export NEL evaluation results as inspect_ai-compatible log files."""

    def __init__(self, format: str = "json", **kwargs: Any) -> None:
        _try_import_inspect()
        self._format = format

    def export(self, bundles: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        output_dir = Path((config or {}).get("output_dir", "./eval_results"))

        for bundle in bundles:
            if bundle.get("_failed") and not bundle.get("_results"):
                continue

            bm = bundle.get("benchmark", {})
            bench_name = bm.get("name", "unknown")
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", bench_name)
            task_dir = output_dir / safe_name
            task_dir.mkdir(parents=True, exist_ok=True)

            log = _build_eval_log(bundle)

            if self._format == "eval":
                self._write_eval_format(log, task_dir, safe_name)
            else:
                self._write_json_format(log, task_dir, safe_name)

    def _write_json_format(self, log: Any, task_dir: Path, name: str) -> None:
        path = task_dir / f"{name}_inspect.json"
        data = json.loads(log.model_dump_json(exclude_none=True))
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Inspect AI log written: %s", path)

    def _write_eval_format(self, log: Any, task_dir: Path, name: str) -> None:
        try:
            from inspect_ai.log import write_eval_log

            path = task_dir / f"{name}_inspect.eval"
            write_eval_log(log, location=str(path), format="eval")
            logger.info("Inspect AI log written: %s", path)
        except Exception:
            logger.warning(
                "Failed to write .eval format, falling back to .json",
                exc_info=True,
            )
            self._write_json_format(log, task_dir, name)
