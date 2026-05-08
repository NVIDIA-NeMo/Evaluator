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
"""ATIF v1.6 trajectory validator (test-only utility).

``strict=True`` additionally requires at least one user/system step
whenever the trajectory contains meaningful agent activity.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

__all__ = [
    "validate_atif",
    "is_valid_atif",
    "ATIFValidationError",
    "assert_atif_compliant",
]

_VALID_SOURCES = frozenset({"system", "user", "agent"})
_VALID_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

_AGENT_ONLY_FIELDS = frozenset({"model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics"})

_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$")


class ATIFValidationError(AssertionError):
    pass


def is_valid_atif(doc: Any, *, strict: bool = False) -> bool:
    return not validate_atif(doc, strict=strict)


def assert_atif_compliant(doc: Any, *, strict: bool = False) -> None:
    errors = validate_atif(doc, strict=strict)
    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ATIFValidationError(f"ATIF validation failed:\n  - {bullet_list}")


def validate_atif(doc: Any, *, strict: bool = False) -> list[str]:
    """Return a list of ATIF validation errors, empty if compliant.

    Accepts either the root document or a single-element list wrapper.
    """
    if isinstance(doc, list):
        if len(doc) != 1:
            return [f"root: expected a single-document list wrapper, got list of length {len(doc)}"]
        doc = doc[0]

    if not isinstance(doc, dict):
        return [f"root: expected dict, got {type(doc).__name__}"]

    errors: list[str] = []
    errors.extend(_check_root(doc))
    steps = doc.get("steps")
    if isinstance(steps, list):
        errors.extend(_check_steps(steps))
        if strict:
            errors.extend(_check_completeness(steps))
    errors.extend(_check_final_metrics(doc.get("final_metrics")))
    errors.extend(_check_agent(doc.get("agent")))
    return errors


def _check_root(doc: dict[str, Any]) -> Iterable[str]:
    required = {"schema_version": str, "session_id": str, "agent": dict, "steps": list}
    for field, expected_type in required.items():
        if field not in doc:
            yield f"root: missing required field '{field}'"
            continue
        if not isinstance(doc[field], expected_type):
            yield f"root.{field}: expected {expected_type.__name__}, got {type(doc[field]).__name__}"

    schema_version = doc.get("schema_version")
    if isinstance(schema_version, str) and not schema_version.startswith("ATIF-"):
        yield f"root.schema_version: must start with 'ATIF-', got {schema_version!r}"

    session_id = doc.get("session_id")
    if isinstance(session_id, str) and not session_id.strip():
        yield "root.session_id: must be non-empty"

    steps = doc.get("steps")
    if isinstance(steps, list) and len(steps) == 0:
        yield "root.steps: must be non-empty"

    for optional_obj_field in ("extra", "final_metrics"):
        val = doc.get(optional_obj_field)
        if val is not None and not isinstance(val, dict):
            yield f"root.{optional_obj_field}: expected dict, got {type(val).__name__}"


def _check_agent(agent: Any) -> Iterable[str]:
    if agent is None:
        return
    if not isinstance(agent, dict):
        return
    for required in ("name", "version"):
        val = agent.get(required)
        if not isinstance(val, str) or not val.strip():
            yield f"root.agent.{required}: missing or empty (must be non-empty string)"
    for optional_str in ("model_name",):
        val = agent.get(optional_str)
        if val is not None and not isinstance(val, str):
            yield f"root.agent.{optional_str}: expected string when present"
    tool_defs = agent.get("tool_definitions")
    if tool_defs is not None:
        if not isinstance(tool_defs, list):
            yield "root.agent.tool_definitions: expected list when present"
        else:
            for i, tdef in enumerate(tool_defs):
                if not isinstance(tdef, dict):
                    yield f"root.agent.tool_definitions[{i}]: expected dict"
                    continue
                if tdef.get("type") != "function":
                    yield f"root.agent.tool_definitions[{i}].type: expected 'function'"
                if not isinstance(tdef.get("function"), dict):
                    yield f"root.agent.tool_definitions[{i}].function: expected dict"


def _check_final_metrics(fm: Any) -> Iterable[str]:
    if fm is None:
        return
    if not isinstance(fm, dict):
        return
    int_fields = ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens", "total_steps")
    for f in int_fields:
        v = fm.get(f)
        if v is not None and not isinstance(v, int):
            yield f"root.final_metrics.{f}: expected int when present, got {type(v).__name__}"
    cost = fm.get("total_cost_usd")
    if cost is not None and not isinstance(cost, (int, float)):
        yield f"root.final_metrics.total_cost_usd: expected number, got {type(cost).__name__}"
    extra = fm.get("extra")
    if extra is not None and not isinstance(extra, dict):
        yield "root.final_metrics.extra: expected dict when present"


def _check_steps(steps: list[Any]) -> Iterable[str]:
    seen_tool_call_ids: set[str] = set()
    tool_calls_by_step: dict[int, set[str]] = {}

    for idx, step in enumerate(steps):
        prefix = f"steps[{idx}]"
        if not isinstance(step, dict):
            yield f"{prefix}: expected dict, got {type(step).__name__}"
            continue

        step_id = step.get("step_id")
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            yield f"{prefix}.step_id: required int, got {type(step_id).__name__}"
        else:
            expected = idx + 1
            if step_id != expected:
                yield (f"{prefix}.step_id: expected sequential ordinal {expected} (1-based), got {step_id}")

        source = step.get("source")
        if source not in _VALID_SOURCES:
            yield f"{prefix}.source: must be one of {sorted(_VALID_SOURCES)}, got {source!r}"

        if "message" not in step:
            yield f"{prefix}.message: required field missing"
        else:
            yield from _check_message_field(step["message"], f"{prefix}.message")

        timestamp = step.get("timestamp")
        if timestamp is not None:
            if not isinstance(timestamp, str) or not _ISO8601_RE.match(timestamp):
                yield f"{prefix}.timestamp: not a valid ISO 8601 string: {timestamp!r}"

        for field_name in _AGENT_ONLY_FIELDS:
            if field_name in step and source != "agent":
                yield (f"{prefix}.{field_name}: only allowed on agent steps, found on {source!r} step")

        tool_calls = step.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                yield f"{prefix}.tool_calls: expected list when present"
            else:
                step_ids: set[str] = set()
                for ti, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        yield f"{prefix}.tool_calls[{ti}]: expected dict"
                        continue
                    yield from _check_tool_call(call, f"{prefix}.tool_calls[{ti}]")
                    tcid = call.get("tool_call_id")
                    if isinstance(tcid, str):
                        if tcid in seen_tool_call_ids:
                            yield f"{prefix}.tool_calls[{ti}].tool_call_id: duplicate id {tcid!r}"
                        else:
                            seen_tool_call_ids.add(tcid)
                            step_ids.add(tcid)
                if step_ids:
                    tool_calls_by_step[idx] = step_ids

        observation = step.get("observation")
        if observation is not None:
            yield from _check_observation(
                observation,
                f"{prefix}.observation",
                tool_call_ids_this_step=tool_calls_by_step.get(idx, set()),
                all_tool_call_ids=seen_tool_call_ids,
            )

        metrics = step.get("metrics")
        if metrics is not None:
            if source != "agent":
                # Already emitted by _AGENT_ONLY_FIELDS block above.
                continue
            yield from _check_metrics(metrics, f"{prefix}.metrics")


def _check_message_field(msg: Any, path: str) -> Iterable[str]:
    if isinstance(msg, str):
        return
    if isinstance(msg, list):
        for i, part in enumerate(msg):
            yield from _check_content_part(part, f"{path}[{i}]")
        return
    yield f"{path}: expected string or list of ContentPart objects, got {type(msg).__name__}"


def _check_content_part(part: Any, path: str) -> Iterable[str]:
    if not isinstance(part, dict):
        yield f"{path}: expected dict ContentPart, got {type(part).__name__}"
        return
    ptype = part.get("type")
    if ptype not in ("text", "image"):
        yield f"{path}.type: must be 'text' or 'image', got {ptype!r}"
        return
    if ptype == "text":
        if "text" not in part or not isinstance(part["text"], str):
            yield f"{path}.text: required string for text parts"
        if "source" in part:
            yield f"{path}.source: must be omitted when type='text'"
    else:
        if "source" not in part or not isinstance(part["source"], dict):
            yield f"{path}.source: required dict for image parts"
        else:
            src = part["source"]
            if src.get("media_type") not in _VALID_MEDIA_TYPES:
                yield f"{path}.source.media_type: must be one of {sorted(_VALID_MEDIA_TYPES)}"
            if not isinstance(src.get("path"), str) or not src["path"]:
                yield f"{path}.source.path: required non-empty string"
        if "text" in part:
            yield f"{path}.text: must be omitted when type='image'"


def _check_tool_call(call: dict[str, Any], path: str) -> Iterable[str]:
    tcid = call.get("tool_call_id")
    if not isinstance(tcid, str) or not tcid:
        yield f"{path}.tool_call_id: required non-empty string"
    fname = call.get("function_name")
    if not isinstance(fname, str) or not fname:
        yield f"{path}.function_name: required non-empty string"
    args = call.get("arguments")
    if not isinstance(args, dict):
        yield f"{path}.arguments: required dict (may be empty), got {type(args).__name__}"


def _check_observation(
    obs: Any,
    path: str,
    *,
    tool_call_ids_this_step: set[str],
    all_tool_call_ids: set[str],
) -> Iterable[str]:
    if not isinstance(obs, dict):
        yield f"{path}: expected dict, got {type(obs).__name__}"
        return
    results = obs.get("results")
    if not isinstance(results, list):
        yield f"{path}.results: required list, got {type(results).__name__}"
        return
    for ri, res in enumerate(results):
        if not isinstance(res, dict):
            yield f"{path}.results[{ri}]: expected dict"
            continue
        scid = res.get("source_call_id")
        if scid is not None:
            if not isinstance(scid, str):
                yield f"{path}.results[{ri}].source_call_id: expected string when present"
            elif scid not in all_tool_call_ids:
                yield (
                    f"{path}.results[{ri}].source_call_id: {scid!r} does not reference any "
                    f"known tool_call_id in the trajectory"
                )
        content = res.get("content")
        if content is not None:
            if isinstance(content, str):
                pass
            elif isinstance(content, list):
                for ci, part in enumerate(content):
                    yield from _check_content_part(part, f"{path}.results[{ri}].content[{ci}]")
            else:
                yield (
                    f"{path}.results[{ri}].content: expected string or list of ContentPart, "
                    f"got {type(content).__name__}"
                )
        sub_ref = res.get("subagent_trajectory_ref")
        if sub_ref is not None:
            if not isinstance(sub_ref, list):
                yield f"{path}.results[{ri}].subagent_trajectory_ref: expected list when present"
            else:
                for si, sub in enumerate(sub_ref):
                    if not isinstance(sub, dict):
                        yield f"{path}.results[{ri}].subagent_trajectory_ref[{si}]: expected dict"
                        continue
                    if not isinstance(sub.get("session_id"), str) or not sub["session_id"]:
                        yield (
                            f"{path}.results[{ri}].subagent_trajectory_ref[{si}].session_id: required non-empty string"
                        )


def _check_metrics(metrics: Any, path: str) -> Iterable[str]:
    if not isinstance(metrics, dict):
        yield f"{path}: expected dict, got {type(metrics).__name__}"
        return
    int_fields = ("prompt_tokens", "completion_tokens", "cached_tokens")
    for f in int_fields:
        v = metrics.get(f)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
            yield f"{path}.{f}: expected int when present, got {type(v).__name__}"
    cost = metrics.get("cost_usd")
    if cost is not None and not isinstance(cost, (int, float)):
        yield f"{path}.cost_usd: expected number when present, got {type(cost).__name__}"
    for arr_field in ("prompt_token_ids", "completion_token_ids", "logprobs"):
        v = metrics.get(arr_field)
        if v is not None and not isinstance(v, list):
            yield f"{path}.{arr_field}: expected list when present"

    prompt_tokens = metrics.get("prompt_tokens")
    cached_tokens = metrics.get("cached_tokens")
    if isinstance(prompt_tokens, int) and isinstance(cached_tokens, int) and cached_tokens > prompt_tokens:
        yield (
            f"{path}.cached_tokens ({cached_tokens}) exceeds prompt_tokens ({prompt_tokens}); "
            f"per ATIF v1.2, cached_tokens is a subset of prompt_tokens"
        )


def _check_completeness(steps: list[Any]) -> Iterable[str]:
    """Require a user/system step when the trajectory has agent activity."""
    has_user_or_system = False
    has_meaningful_agent = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        src = step.get("source")
        if src in ("user", "system"):
            has_user_or_system = True
        elif src == "agent":
            msg = step.get("message")
            msg_nonempty = (isinstance(msg, str) and msg.strip()) or (isinstance(msg, list) and any(msg))
            if msg_nonempty or step.get("tool_calls"):
                has_meaningful_agent = True
    if has_meaningful_agent and not has_user_or_system:
        yield (
            "root.steps: trajectory has agent activity but no user or system step; "
            "ATIF defines a trajectory as 'the complete interaction history, including "
            "all user messages (initial and subsequent)' -- the initiating user turn "
            "must be recorded"
        )
