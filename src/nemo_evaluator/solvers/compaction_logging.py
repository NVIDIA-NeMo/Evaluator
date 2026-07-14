# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

CompactionTrigger = Literal[
    "terminus_proactive_threshold",
    "terminus_context_length_exceeded",
    "terminus_agent_initiated",
    "openhands_condensation_events",
    "openhands_condensation_tokens",
    "openhands_condensation_request",
    "openhands_condensation_hard_reset",
    "unknown",
]

CompactionStrategy = Literal[
    "terminus_three_phase_subagent",
    "terminus_short_fallback",
    "terminus_ultimate_fallback",
    "openhands_condensation",
    "single_pass_summary",
    "unknown",
]

CompactionBoundary = Literal["replace", "truncate", "append"]

CompactionOutcome = Literal["failed"]

CompactionPhase = Literal["summary", "questions", "answers"]

COMPACTION_MESSAGE = "Context compaction performed"


@dataclass
class CompactionTokens:
    prompt_tokens_approx: int
    context_limit: int
    free_tokens: int


@dataclass
class CompactionTokensIntermediate:
    after_unwind: CompactionTokens | None = None
    after_chat_reset: CompactionTokens | None = None


@dataclass
class CompactionMechanism:
    unwind_applied: bool = False
    chat_reset_applied: bool = False
    messages_unwound_pairs: int = 0
    unwind_target_free_tokens: int | None = None


@dataclass
class CompactionAttempt:
    strategy: CompactionStrategy
    outcome: CompactionOutcome = "failed"
    error: str | None = None
    llm_calls: int | None = None
    phases_completed: list[CompactionPhase] | None = None
    subagent_trajectory_ref: list[dict[str, Any]] | None = None


@dataclass
class OpenHandsCompactionExtra:
    reason: str | None = None
    requirement: str | None = None
    hard_reset: bool = False
    forgotten_event_count: int = 0
    max_size: int | None = None
    max_tokens: int | None = None
    keep_first: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.reason is not None:
            out["reason"] = self.reason
        if self.requirement is not None:
            out["requirement"] = self.requirement
        if self.hard_reset:
            out["hard_reset"] = True
        if self.forgotten_event_count:
            out["forgotten_event_count"] = self.forgotten_event_count
        if self.max_size is not None:
            out["max_size"] = self.max_size
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.keep_first is not None:
            out["keep_first"] = self.keep_first
        return out


@dataclass
class CompactionObservationExtra:
    compaction_prompt_tokens: int | None = None
    compaction_completion_tokens: int | None = None
    compaction_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.compaction_prompt_tokens is not None:
            out["compaction_prompt_tokens"] = self.compaction_prompt_tokens
        if self.compaction_completion_tokens is not None:
            out["compaction_completion_tokens"] = self.compaction_completion_tokens
        if self.compaction_cost_usd is not None:
            out["compaction_cost_usd"] = self.compaction_cost_usd
        return out


@dataclass
class CompactionEvent:
    compaction_index: int
    trigger: CompactionTrigger
    strategy: CompactionStrategy
    replacement_content: str
    tokens_before: CompactionTokens
    tokens_after: CompactionTokens
    llm_call_count: int
    tokens_intermediate: CompactionTokensIntermediate | None = None
    boundary: CompactionBoundary = "replace"
    mechanism: CompactionMechanism | None = None
    subagent_trajectory_ref: list[dict[str, Any]] | None = None
    attempts: list[CompactionAttempt] | None = None
    observation_extra: CompactionObservationExtra | None = None
    openhands_extra: OpenHandsCompactionExtra | None = None
    timestamp: str | None = None
    message: str = COMPACTION_MESSAGE


def _subagent_ref_to_dict(ref: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    session_id = ref.get("session_id")
    if session_id is None and ref.get("trajectory_id") is not None:
        session_id = ref.get("trajectory_id")
    if session_id is not None:
        out["session_id"] = session_id
    extra = ref.get("extra")
    if extra:
        out["extra"] = extra
    return out


def _attempt_to_dict(attempt: CompactionAttempt) -> dict[str, Any]:
    out: dict[str, Any] = {
        "strategy": attempt.strategy,
        "outcome": attempt.outcome,
    }
    if attempt.error is not None:
        out["error"] = attempt.error
    if attempt.llm_calls is not None:
        out["llm_calls"] = attempt.llm_calls
    if attempt.phases_completed is not None:
        out["phases_completed"] = list(attempt.phases_completed)
    if attempt.subagent_trajectory_ref is not None:
        out["subagent_trajectory_ref"] = [_subagent_ref_to_dict(r) for r in attempt.subagent_trajectory_ref]
    return out


def _tokens_to_dict(tokens: CompactionTokens) -> dict[str, Any]:
    return {
        "prompt_tokens_approx": tokens.prompt_tokens_approx,
        "context_limit": tokens.context_limit,
        "free_tokens": tokens.free_tokens,
    }


def _intermediate_tokens_to_dict(
    intermediate: CompactionTokensIntermediate,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if intermediate.after_unwind is not None:
        out["after_unwind"] = _tokens_to_dict(intermediate.after_unwind)
    if intermediate.after_chat_reset is not None:
        out["after_chat_reset"] = _tokens_to_dict(intermediate.after_chat_reset)
    return out


def _mechanism_to_dict(mechanism: CompactionMechanism) -> dict[str, Any]:
    out: dict[str, Any] = {
        "unwind_applied": mechanism.unwind_applied,
        "chat_reset_applied": mechanism.chat_reset_applied,
        "messages_unwound_pairs": mechanism.messages_unwound_pairs,
    }
    if mechanism.unwind_target_free_tokens is not None:
        out["unwind_target_free_tokens"] = mechanism.unwind_target_free_tokens
    return out


def subagent_trajectory_refs_to_dicts(refs: Any) -> list[dict[str, Any]] | None:
    if not refs:
        return None
    out: list[dict[str, Any]] = []
    for ref in refs:
        if hasattr(ref, "model_dump"):
            dumped = ref.model_dump(exclude_none=True, mode="json")
            out.append(dumped)
        elif isinstance(ref, dict):
            out.append(dict(ref))
        else:
            raise TypeError(f"Unsupported subagent trajectory ref type: {type(ref)!r}")
    return out


def compaction_event_to_harbor_step(event: CompactionEvent, step_id: int) -> Any:
    from harbor.models.trajectories.observation import Observation
    from harbor.models.trajectories.observation_result import ObservationResult
    from harbor.models.trajectories.step import Step
    from harbor.models.trajectories.subagent_trajectory_ref import SubagentTrajectoryRef

    raw = build_compaction_step(event, step_id)
    observation_raw = raw["observation"]["results"][0]
    refs = None
    subagent_refs = observation_raw.get("subagent_trajectory_ref")
    if subagent_refs:
        refs = [SubagentTrajectoryRef.model_validate(item) for item in subagent_refs]
    observation = Observation(
        results=[
            ObservationResult(
                content=observation_raw.get("content"),
                subagent_trajectory_ref=refs,
            )
        ]
    )
    step = Step(
        step_id=raw["step_id"],
        timestamp=raw["timestamp"],
        source=raw["source"],
        message=raw["message"],
        observation=observation,
        extra=raw["extra"],
    )
    return step


def build_compaction_step(event: CompactionEvent, step_id: int) -> dict[str, Any]:
    timestamp = event.timestamp or datetime.now(timezone.utc).isoformat()

    observation_result: dict[str, Any] = {
        "content": event.replacement_content,
    }
    if event.subagent_trajectory_ref:
        observation_result["subagent_trajectory_ref"] = [
            _subagent_ref_to_dict(r) for r in event.subagent_trajectory_ref
        ]
    if event.observation_extra is not None:
        obs_extra = event.observation_extra.to_dict()
        if obs_extra:
            observation_result["extra"] = obs_extra

    compaction_extra: dict[str, Any] = {
        "compaction_index": event.compaction_index,
        "trigger": event.trigger,
        "strategy": event.strategy,
        "llm_call_count": event.llm_call_count,
        "tokens_before": _tokens_to_dict(event.tokens_before),
        "tokens_after": _tokens_to_dict(event.tokens_after),
    }
    if event.tokens_intermediate is not None:
        intermediate = _intermediate_tokens_to_dict(event.tokens_intermediate)
        if intermediate:
            compaction_extra["tokens_intermediate"] = intermediate
    strategy_details: dict[str, Any] | None = None
    if event.mechanism is not None:
        strategy_details = _mechanism_to_dict(event.mechanism)
    elif event.openhands_extra is not None:
        oh_extra = event.openhands_extra.to_dict()
        if oh_extra:
            strategy_details = oh_extra
    if strategy_details is not None:
        compaction_extra["strategy_details"] = strategy_details
    if event.attempts:
        compaction_extra["attempts"] = [_attempt_to_dict(a) for a in event.attempts]

    return {
        "step_id": step_id,
        "timestamp": timestamp,
        "source": "system",
        "message": event.message,
        "observation": {
            "results": [observation_result],
        },
        "extra": {
            "context_management": {
                "type": "compaction",
                "boundary": event.boundary,
            },
            "compaction": compaction_extra,
        },
    }


def llm_call_count_for_strategy(strategy: CompactionStrategy) -> int:
    if strategy == "terminus_three_phase_subagent":
        return 3
    if strategy == "terminus_short_fallback":
        return 1
    if strategy == "terminus_ultimate_fallback":
        return 0
    if strategy in ("openhands_condensation", "single_pass_summary"):
        return 1
    return 0
