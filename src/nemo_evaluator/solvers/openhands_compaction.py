# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime
from typing import Any

from openhands.sdk.context.condenser.utils import get_total_token_count
from openhands.sdk.context.view import View
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.llm import LLM

from nemo_evaluator.solvers.compaction_logging import (
    CompactionEvent,
    CompactionObservationExtra,
    CompactionTokens,
    CompactionTrigger,
    OpenHandsCompactionExtra,
    StepsCompacted,
    build_compaction_step,
    llm_call_count_for_strategy,
)

DEFAULT_OPENHANDS_MAX_SIZE = 80
DEFAULT_OPENHANDS_KEEP_FIRST = 4


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _timestamp_sort_key(value: str | None) -> tuple[int, str]:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return (1, str(value or ""))
    return (0, parsed.isoformat())


def is_hard_reset(condensation: Condensation, view_before: View) -> bool:
    if condensation.summary_offset != 0:
        return False
    if not view_before.events:
        return False
    forgotten = condensation.forgotten_event_ids
    if not forgotten:
        return False
    view_ids = {event.id for event in view_before.events}
    return view_ids.issubset(forgotten)


def _segment_since_previous_condensation(events: list[Event], condensation_idx: int) -> list[Event]:
    start = 0
    idx = 0
    while idx < condensation_idx:
        if isinstance(events[idx], Condensation):
            start = idx + 1
        idx += 1
    return events[start:condensation_idx]


def _has_condensation_request(segment: list[Event]) -> bool:
    for event in segment:
        if isinstance(event, CondensationRequest):
            return True
    return False


def _replay_openhands_reasons(
    view_before: View,
    llm: LLM,
    *,
    max_size: int,
    max_tokens: int | None,
) -> set[str]:
    reasons: set[str] = set()
    if len(view_before) > max_size:
        reasons.add("events")
    if max_tokens is not None:
        total_tokens = get_total_token_count(view_before.events, llm)
        if total_tokens > max_tokens:
            reasons.add("tokens")
    return reasons


def infer_openhands_trigger(
    events: list[Event],
    condensation_idx: int,
    condensation: Condensation,
    view_before: View,
    llm: LLM,
    *,
    max_size: int,
    max_tokens: int | None,
) -> CompactionTrigger:
    if is_hard_reset(condensation, view_before):
        return "openhands_condensation_hard_reset"
    segment = _segment_since_previous_condensation(events, condensation_idx)
    if _has_condensation_request(segment):
        return "openhands_condensation_request"
    reasons = _replay_openhands_reasons(
        view_before,
        llm,
        max_size=max_size,
        max_tokens=max_tokens,
    )
    if "tokens" in reasons:
        return "openhands_condensation_tokens"
    if "events" in reasons:
        return "openhands_condensation_events"
    return "unknown"


def token_snapshot(
    view_events: list[Any],
    llm: LLM,
    context_limit: int,
) -> CompactionTokens:
    prompt_tokens = int(get_total_token_count(view_events, llm))
    free_tokens = max(0, context_limit - prompt_tokens)
    return CompactionTokens(
        prompt_tokens_approx=prompt_tokens,
        context_limit=context_limit,
        free_tokens=free_tokens,
    )


def _observation_extra_for_response(llm: LLM, response_id: str | None) -> CompactionObservationExtra | None:
    if not response_id:
        return None
    usages = getattr(getattr(llm, "metrics", None), "token_usages", None) or []
    for usage in usages:
        if getattr(usage, "response_id", None) != response_id:
            continue
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        cost = getattr(usage, "cost", None)
        return CompactionObservationExtra(
            compaction_prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            compaction_completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
            compaction_cost_usd=float(cost) if cost is not None else None,
        )
    return None


def build_event_to_step_map(events: list[Event]) -> dict[str, int]:
    event_to_step: dict[str, int] = {}
    step_id = 1
    for event in events:
        if isinstance(event, MessageEvent):
            if event.source == "user":
                event_to_step[event.id] = step_id
                step_id += 1
            elif event.source == "agent":
                event_to_step[event.id] = step_id
                step_id += 1
        elif isinstance(event, ActionEvent):
            event_to_step[event.id] = step_id
            step_id += 1
        elif isinstance(event, ObservationEvent):
            pass
        elif isinstance(event, Condensation):
            pass
    return event_to_step


def map_forgotten_to_steps(
    forgotten_event_ids: set[str],
    event_to_step: dict[str, int],
) -> StepsCompacted | None:
    step_ids: list[int] = []
    for event_id in forgotten_event_ids:
        mapped = event_to_step.get(event_id)
        if mapped is not None:
            step_ids.append(mapped)
    if not step_ids:
        return None
    step_ids = sorted(set(step_ids))
    return StepsCompacted(
        first_step_id=step_ids[0],
        last_step_id=step_ids[-1],
        step_count=len(step_ids),
    )


def _resolve_condenser_config(condenser_config: dict[str, Any] | None) -> dict[str, int | None]:
    cfg = condenser_config or {}
    max_size = cfg.get("max_size")
    max_tokens = cfg.get("max_tokens")
    keep_first = cfg.get("keep_first")
    return {
        "max_size": int(max_size) if max_size is not None else DEFAULT_OPENHANDS_MAX_SIZE,
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "keep_first": int(keep_first) if keep_first is not None else DEFAULT_OPENHANDS_KEEP_FIRST,
    }


def build_compaction_events_from_run(
    events: list[Event],
    llm: LLM,
    context_limit: int,
    *,
    condenser_config: dict[str, Any] | None = None,
) -> list[tuple[str, CompactionEvent]]:
    cfg = _resolve_condenser_config(condenser_config)
    event_to_step = build_event_to_step_map(events)
    out: list[tuple[str, CompactionEvent]] = []
    compaction_index = 0
    idx = 0
    while idx < len(events):
        event = events[idx]
        if not isinstance(event, Condensation):
            idx += 1
            continue
        compaction_index += 1
        view_before = View.from_events(events[:idx])
        view_after_events = event.apply(view_before.events)
        tokens_before = token_snapshot(view_before.events, llm, context_limit)
        tokens_after = token_snapshot(view_after_events, llm, context_limit)
        trigger = infer_openhands_trigger(
            events,
            idx,
            event,
            view_before,
            llm,
            max_size=int(cfg["max_size"]),
            max_tokens=cfg["max_tokens"],
        )
        hard_reset = trigger == "openhands_condensation_hard_reset"
        reason = None
        if trigger == "openhands_condensation_events":
            reason = "events"
        elif trigger == "openhands_condensation_tokens":
            reason = "tokens"
        elif trigger == "openhands_condensation_request":
            reason = "request"
        elif trigger == "openhands_condensation_hard_reset":
            reason = "hard_reset"
        openhands_extra = OpenHandsCompactionExtra(
            reason=reason,
            requirement="hard" if hard_reset or trigger == "openhands_condensation_request" else "soft",
            hard_reset=hard_reset,
            forgotten_event_count=len(event.forgotten_event_ids),
            max_size=int(cfg["max_size"]),
            max_tokens=cfg["max_tokens"],
            keep_first=int(cfg["keep_first"]),
        )
        compaction_event = CompactionEvent(
            compaction_index=compaction_index,
            trigger=trigger,
            strategy="openhands_condensation",
            replacement_content=event.summary or "",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            llm_call_count=llm_call_count_for_strategy("openhands_condensation"),
            steps_compacted=map_forgotten_to_steps(event.forgotten_event_ids, event_to_step),
            observation_extra=_observation_extra_for_response(llm, event.llm_response_id),
            openhands_extra=openhands_extra,
            timestamp=event.timestamp,
        )
        out.append((event.timestamp, compaction_event))
        idx += 1
    return out


def splice_compaction_steps_into_trajectory(
    trajectory: dict[str, Any],
    compaction_entries: list[tuple[str, CompactionEvent]],
) -> dict[str, Any]:
    if not compaction_entries:
        return trajectory
    steps = list(trajectory.get("steps") or [])
    pending = list(compaction_entries)
    pending.sort(key=lambda item: _timestamp_sort_key(item[0]))
    merged: list[dict[str, Any]] = []
    pending_idx = 0
    for step in steps:
        step_ts = step.get("timestamp")
        while pending_idx < len(pending):
            compaction_ts, compaction_event = pending[pending_idx]
            if _timestamp_sort_key(compaction_ts) > _timestamp_sort_key(step_ts):
                break
            merged.append(build_compaction_step(compaction_event, step_id=0))
            pending_idx += 1
        merged.append(step)
    while pending_idx < len(pending):
        _, compaction_event = pending[pending_idx]
        merged.append(build_compaction_step(compaction_event, step_id=0))
        pending_idx += 1
    for index, step in enumerate(merged):
        step["step_id"] = index + 1
    trajectory["steps"] = merged
    return trajectory


def enrich_trajectory_with_compaction(
    trajectory: dict[str, Any],
    events: list[Event],
    llm: LLM,
    context_limit: int,
    *,
    condenser_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compaction_entries = build_compaction_events_from_run(
        events,
        llm,
        context_limit,
        condenser_config=condenser_config,
    )
    if not compaction_entries:
        return trajectory
    return splice_compaction_steps_into_trajectory(trajectory, compaction_entries)
