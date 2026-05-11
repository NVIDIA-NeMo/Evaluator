# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF-to-MLflow-Trace adapter.

Converts NEL result dicts into MLflow Traces (``mlflow.start_span`` calls).
Gracefully degrades across benchmark kinds:

Tier A — agentic with tool calls (PinchBench, Harbor):
    AGENT root → [LLM span + one TOOL span per call] per assistant turn.

Tier B — trajectory with messages only (chat benchmarks):
    AGENT root → one LLM span per assistant turn.

Tier C — no trajectory but ``model_response`` present (GSM8K, MMLU):
    AGENT root → a single LLM span covering prompt → response.

Tier D — neither (scored-only / failed sample):
    AGENT root with metadata only.

The per-step walker is MLflow-agnostic; only :func:`emit_sample_trace` imports
``mlflow``. Callers must invoke it inside an active ``mlflow.start_run`` context.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_CONTENT_MAX = 4000

_SYNTHETIC_TOOL_LABEL_RE = re.compile(r"^Executed \S+ \S+$")


def _trim(value: Any, limit: int) -> Any:
    """Recursively truncate strings within ``value`` to ``limit`` chars."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…[truncated]"
    if isinstance(value, list):
        return [_trim(v, limit) for v in value]
    if isinstance(value, tuple):
        return tuple(_trim(v, limit) for v in value)
    if isinstance(value, dict):
        return {k: _trim(v, limit) for k, v in value.items()}
    return value


def iter_atif_steps(trajectory: Any) -> Iterable[dict[str, Any]]:
    """Yield ATIF step dicts from a trajectory list-of-documents structure.

    Defensive against malformed inputs (non-list trajectory, non-dict docs,
    missing ``steps`` key, non-dict step entries).
    """
    if not isinstance(trajectory, list):
        return
    for doc in trajectory:
        if not isinstance(doc, dict):
            continue
        steps = doc.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict):
                yield step


def _classify_trajectory(trajectory: Any) -> str:
    """Return ``"rich"`` if any step has tool_calls, ``"messages"`` if it has
    at least one step otherwise, ``"empty"`` if the trajectory is empty/invalid.
    """
    has_any = False
    has_tool = False
    for step in iter_atif_steps(trajectory):
        has_any = True
        if step.get("tool_calls"):
            has_tool = True
            break
    if has_tool:
        return "rich"
    if has_any:
        return "messages"
    return "empty"


def _observation_map(step: dict[str, Any]) -> dict[str, str]:
    """Map ``source_call_id`` → observation string from an ATIF step."""
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return {}
    results = observation.get("results")
    if not isinstance(results, list):
        return {}
    out: dict[str, str] = {}
    for res in results:
        if not isinstance(res, dict):
            continue
        rid = str(res.get("source_call_id") or res.get("tool_call_id") or "")
        content = res.get("content")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, default=str)
            except Exception:
                content = str(content)
        out[rid] = content
    return out


def _resolve_prompt(sample: dict[str, Any]) -> str:
    """Resolve the user-facing prompt, falling back through metadata,
    trajectory, and expected_answer. Returns ``""`` when none available.
    """
    prompt = sample.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    metadata = sample.get("metadata") or {}
    if isinstance(metadata, dict):
        mp = metadata.get("prompt")
        if isinstance(mp, str) and mp.strip():
            return mp
    for step in iter_atif_steps(sample.get("trajectory") or []):
        if step.get("source") in ("user", "system"):
            msg = step.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg
    ea = sample.get("expected_answer")
    if isinstance(ea, str) and ea.strip():
        return ea
    return ""


def emit_sample_trace(
    sample: dict[str, Any],
    *,
    model_name: str,
    bench_name: str,
    content_max: int = DEFAULT_CONTENT_MAX,
) -> str:
    """Emit one MLflow trace for ``sample`` and return the tier that was used.

    Tiers returned: ``"rich"`` / ``"messages"`` / ``"response"`` / ``"meta"``.
    Must be called inside an active ``mlflow.start_run``.
    """
    import mlflow
    from mlflow.entities import SpanType

    trajectory = sample.get("trajectory") or []
    problem_idx = sample.get("problem_idx")
    repeat = sample.get("repeat", 0) or 0
    reward = sample.get("reward")
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    prompt = _resolve_prompt(sample)
    model_response = sample.get("model_response") or ""

    root_name = f"sample_{problem_idx}" if problem_idx is not None else "sample"
    if repeat:
        root_name = f"{root_name}.r{repeat}"

    tier = _classify_trajectory(trajectory)

    with mlflow.start_span(name=root_name, span_type=SpanType.AGENT) as root:
        root.set_inputs({"prompt": _trim(prompt, content_max)})
        attrs: dict[str, Any] = {
            "nel.benchmark": bench_name,
            "nel.model": model_name,
        }
        if problem_idx is not None:
            attrs["nel.problem_idx"] = problem_idx
        if repeat:
            attrs["nel.repeat"] = repeat
        if isinstance(reward, (int, float)):
            attrs["nel.reward"] = float(reward)

        for meta_key in ("task_id", "task_name", "category", "grading_type", "source"):
            mv = metadata.get(meta_key)
            if isinstance(mv, (str, int, float, bool)) and mv != "":
                attrs[f"nel.{meta_key}"] = mv

        scoring_details = sample.get("scoring_details")
        if isinstance(scoring_details, dict) and scoring_details:
            attrs["nel.scoring_details"] = _trim(scoring_details, content_max)

        scorer_rewards = sample.get("scorer_rewards")
        if isinstance(scorer_rewards, dict):
            for sk, sv in scorer_rewards.items():
                if isinstance(sv, (int, float)) and isinstance(sk, str):
                    attrs[f"nel.scorer.{sk}"] = float(sv)

        root.set_attributes(attrs)

        synthesized_user = False
        if tier in ("rich", "messages"):
            steps = list(iter_atif_steps(trajectory))
            has_user = any(step.get("source") in ("user", "system") for step in steps)
            if not has_user and prompt:
                _emit_user_prompt_span(prompt, content_max=content_max)
                synthesized_user = True
            _emit_trajectory_spans(steps, content_max=content_max)
            used_tier = tier
        elif model_response:
            _emit_singleshot_llm_span(
                prompt=prompt,
                response=model_response,
                model_name=model_name,
                tokens=sample.get("tokens"),
                latency_ms=sample.get("latency_ms"),
                content_max=content_max,
            )
            used_tier = "response"
        else:
            used_tier = "meta"

        if synthesized_user:
            root.set_attributes({"nel.user_prompt_synthesized": True})

        root.set_outputs(
            {
                "reward": reward,
                "extracted_answer": _trim(sample.get("extracted_answer"), content_max),
                "response": _trim(model_response, content_max) if model_response else None,
            }
        )

    return used_tier


def _emit_user_prompt_span(prompt: str, *, content_max: int) -> None:
    import mlflow
    from mlflow.entities import SpanType

    with mlflow.start_span(name="user", span_type=SpanType.CHAIN) as s:
        s.set_inputs({"content": _trim(prompt, content_max)})


def _emit_trajectory_spans(steps: list[dict[str, Any]], *, content_max: int) -> None:
    import mlflow
    from mlflow.entities import SpanType

    for step in steps:
        source = step.get("source", "")
        text = step.get("message", "") or ""
        tool_calls = step.get("tool_calls") or []

        if source in ("system", "user") and text and not tool_calls:
            with mlflow.start_span(name=source, span_type=SpanType.CHAIN) as s:
                s.set_inputs({"content": _trim(text, content_max)})
            continue

        if source != "agent":
            continue

        cleaned_text = text
        if tool_calls and _SYNTHETIC_TOOL_LABEL_RE.match((text or "").strip()):
            cleaned_text = ""

        if cleaned_text or not tool_calls:
            with mlflow.start_span(name="assistant", span_type=SpanType.LLM) as llm:
                llm.set_outputs({"message": _trim(cleaned_text, content_max)})

        if not tool_calls:
            continue

        obs_by_id = _observation_map(step)
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = str(call.get("function_name") or call.get("function") or "tool")
            cid = str(call.get("tool_call_id") or call.get("id") or "")
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"value": args}
            with mlflow.start_span(name=fn, span_type=SpanType.TOOL) as tool:
                tool.set_inputs(_trim(args, content_max))
                observation = obs_by_id.get(cid)
                if observation is not None:
                    tool.set_outputs({"observation": _trim(observation, content_max)})


def _emit_singleshot_llm_span(
    *,
    prompt: str,
    response: str,
    model_name: str,
    tokens: int | None,
    latency_ms: int | None,
    content_max: int,
) -> None:
    import mlflow
    from mlflow.entities import SpanType

    with mlflow.start_span(name="llm", span_type=SpanType.LLM) as s:
        s.set_inputs({"prompt": _trim(prompt, content_max)})
        s.set_outputs({"response": _trim(response, content_max)})
        attrs: dict[str, Any] = {"nel.model": model_name}
        if isinstance(tokens, int) and tokens > 0:
            attrs["nel.output_tokens"] = tokens
        if isinstance(latency_ms, (int, float)) and latency_ms > 0:
            attrs["nel.latency_ms"] = float(latency_ms)
        s.set_attributes(attrs)
