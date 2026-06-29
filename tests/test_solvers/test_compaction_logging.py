# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from nemo_evaluator.solvers.compaction_logging import (
    CompactionAttempt,
    CompactionEvent,
    CompactionMechanism,
    CompactionObservationExtra,
    CompactionTokens,
    CompactionTokensIntermediate,
    StepsCompacted,
    build_compaction_step,
    compaction_event_to_harbor_step,
    llm_call_count_for_strategy,
)


def _minimal_event(**overrides) -> CompactionEvent:
    base = CompactionEvent(
        compaction_index=1,
        trigger="proactive_threshold",
        strategy="terminus_three_phase_subagent",
        replacement_content="handoff text",
        tokens_before=CompactionTokens(
            prompt_tokens_approx=190_000,
            context_limit=200_000,
            free_tokens=10_000,
        ),
        tokens_after=CompactionTokens(
            prompt_tokens_approx=8_000,
            context_limit=200_000,
            free_tokens=192_000,
        ),
        llm_call_count=3,
        steps_compacted=StepsCompacted(first_step_id=2, last_step_id=10, step_count=9),
        mechanism=CompactionMechanism(
            unwind_applied=False,
            chat_reset_applied=False,
            messages_unwound_pairs=0,
        ),
        subagent_trajectory_ref=[
            {
                "trajectory_path": "trajectory.summarization-1-summary.json",
                "session_id": "sess-summarization-1-summary",
                "extra": {"phase": "summary"},
            }
        ],
        timestamp="2026-06-23T02:17:14.197933+00:00",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_build_compaction_step_full_three_phase() -> None:
    step = build_compaction_step(_minimal_event(), step_id=11)
    assert step["source"] == "system"
    assert step["llm_call_count"] == 3
    assert step["extra"]["context_management"] == {"type": "compaction", "boundary": "replace"}
    assert step["extra"]["compaction"]["strategy"] == "terminus_three_phase_subagent"
    assert step["extra"]["compaction"]["tokens_before"]["prompt_tokens_approx"] == 190_000
    assert step["extra"]["compaction"]["tokens_after"]["prompt_tokens_approx"] == 8_000
    assert step["observation"]["results"][0]["content"] == "handoff text"
    refs = step["observation"]["results"][0]["subagent_trajectory_ref"]
    assert len(refs) == 1
    assert refs[0]["session_id"] == "sess-summarization-1-summary"


def test_build_compaction_step_with_intermediate_tokens() -> None:
    event = _minimal_event(
        trigger="context_length_exceeded",
        strategy="terminus_short_fallback",
        tokens_intermediate=CompactionTokensIntermediate(
            after_unwind=CompactionTokens(
                prompt_tokens_approx=150_000,
                context_limit=200_000,
                free_tokens=50_000,
            ),
            after_chat_reset=CompactionTokens(
                prompt_tokens_approx=12_000,
                context_limit=200_000,
                free_tokens=188_000,
            ),
        ),
    )
    step = build_compaction_step(event, step_id=7)
    intermediate = step["extra"]["compaction"]["tokens_intermediate"]
    assert intermediate["after_unwind"]["prompt_tokens_approx"] == 150_000
    assert intermediate["after_chat_reset"]["prompt_tokens_approx"] == 12_000


def test_build_compaction_step_with_failed_attempts() -> None:
    event = _minimal_event(
        strategy="terminus_short_fallback",
        llm_call_count=1,
        subagent_trajectory_ref=None,
        attempts=[
            CompactionAttempt(
                strategy="terminus_three_phase_subagent",
                error="timeout",
                llm_calls=2,
                phases_completed=["summary", "questions"],
            )
        ],
        mechanism=CompactionMechanism(
            unwind_applied=True,
            chat_reset_applied=True,
            messages_unwound_pairs=10,
            unwind_target_free_tokens=4000,
        ),
        observation_extra=CompactionObservationExtra(
            compaction_prompt_tokens=4200,
            compaction_completion_tokens=180,
            compaction_cost_usd=0.008,
        ),
    )
    step = build_compaction_step(event, step_id=5)
    attempts = step["extra"]["compaction"]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["strategy"] == "terminus_three_phase_subagent"
    assert attempts[0]["outcome"] == "failed"
    mechanism = step["extra"]["compaction"]["mechanism"]
    assert mechanism["unwind_applied"] is True
    assert mechanism["messages_unwound_pairs"] == 10
    obs_extra = step["observation"]["results"][0]["extra"]
    assert obs_extra["compaction_prompt_tokens"] == 4200


def test_llm_call_count_for_strategy() -> None:
    assert llm_call_count_for_strategy("terminus_three_phase_subagent") == 3
    assert llm_call_count_for_strategy("terminus_short_fallback") == 1
    assert llm_call_count_for_strategy("terminus_ultimate_fallback") == 0


def test_compaction_event_to_harbor_step() -> None:
    step, llm_call_count = compaction_event_to_harbor_step(_minimal_event(), step_id=3)
    assert llm_call_count == 3
    assert step.step_id == 3
    assert step.source == "system"
    assert step.extra is not None
    assert step.extra["context_management"]["type"] == "compaction"
    assert step.observation is not None
    assert step.observation.results[0].content == "handoff text"
    assert step.observation.results[0].subagent_trajectory_ref is not None
    assert step.observation.results[0].subagent_trajectory_ref[0].session_id == "sess-summarization-1-summary"
