# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenHands SDK condensation → ATIF compaction logging.

openhands-sdk is not a package dependency; the module skips when it is absent.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openhands.sdk", reason="openhands-sdk not installed")

from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.event.llm_convertible.message import MessageEvent
from openhands.sdk.llm import LLM, Message, TextContent

from nemo_evaluator.solvers.openhands_compaction import (
    build_compaction_events_from_run,
    enrich_trajectory_with_compaction,
    infer_openhands_trigger,
    is_hard_reset,
    splice_compaction_steps_into_trajectory,
)


def _user_message(text: str) -> MessageEvent:
    return MessageEvent(
        source="user",
        llm_message=Message(role="user", content=[TextContent(text=text)]),
    )


def _agent_message(text: str) -> MessageEvent:
    return MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text=text)]),
        llm_response_id="resp-agent",
    )


def _condensation(
    forgotten_ids: set[str],
    *,
    summary: str = "condensed summary",
    summary_offset: int = 1,
    response_id: str = "resp-condense",
) -> Condensation:
    return Condensation(
        forgotten_event_ids=forgotten_ids,
        summary=summary,
        summary_offset=summary_offset,
        llm_response_id=response_id,
    )


@pytest.fixture
def mock_llm() -> LLM:
    llm = MagicMock(spec=LLM)
    llm.metrics = SimpleNamespace(token_usages=[])
    return llm


def test_is_hard_reset_detects_full_view_forget() -> None:
    user = _user_message("one")
    agent = _agent_message("two")
    view = View.from_events([user, agent])
    condensation = _condensation({user.id, agent.id}, summary_offset=0)
    assert is_hard_reset(condensation, view) is True


def test_infer_trigger_condensation_request() -> None:
    user = _user_message("task")
    request = CondensationRequest()
    condensation = _condensation({user.id})
    events = [user, request, condensation]
    view = View.from_events(events[:2])
    trigger = infer_openhands_trigger(
        events,
        2,
        condensation,
        view,
        MagicMock(spec=LLM),
        max_size=80,
        max_tokens=None,
    )
    assert trigger == "openhands_condensation_request"


@patch("nemo_evaluator.solvers.openhands_compaction.get_total_token_count", side_effect=[120, 40, 30, 15])
def test_build_compaction_events_proactive_events(mock_count: MagicMock, mock_llm: LLM) -> None:
    events = [_user_message(f"msg-{idx}") for idx in range(90)]
    forgotten = {event.id for event in events[:40]}
    events.append(_condensation(forgotten))
    entries = build_compaction_events_from_run(
        events,
        mock_llm,
        context_limit=200_000,
        condenser_config={"max_size": 80},
    )
    assert len(entries) == 1
    _, compaction_event = entries[0]
    assert compaction_event.trigger == "openhands_condensation_events"
    assert compaction_event.strategy == "openhands_condensation"
    replacement = json.loads(compaction_event.replacement_content)
    assert isinstance(replacement, list)
    assert len(replacement) >= 1
    joined = "\n".join(message.get("content") or "" for message in replacement)
    assert "condensed summary" in joined
    assert "msg-40" in joined
    assert "msg-89" in joined
    assert "msg-0" not in joined
    assert compaction_event.openhands_extra is not None
    assert compaction_event.openhands_extra.reason == "events"
    assert compaction_event.openhands_extra.requirement == "soft"
    assert compaction_event.tokens_before.prompt_tokens_approx == 120
    assert compaction_event.tokens_after.prompt_tokens_approx == 40


@patch(
    "nemo_evaluator.solvers.openhands_compaction.get_total_token_count", side_effect=[150_000, 8_000, 150_000, 8_000]
)
def test_build_compaction_events_token_trigger(mock_count: MagicMock, mock_llm: LLM) -> None:
    user = _user_message("long context")
    condensation = _condensation({user.id})
    events = [user, condensation]
    entries = build_compaction_events_from_run(
        events,
        mock_llm,
        context_limit=200_000,
        condenser_config={"max_size": 10_000, "max_tokens": 100_000},
    )
    _, compaction_event = entries[0]
    assert compaction_event.trigger == "openhands_condensation_tokens"
    assert compaction_event.openhands_extra is not None
    assert compaction_event.openhands_extra.reason == "tokens"
    assert compaction_event.openhands_extra.requirement == "hard"


@patch("nemo_evaluator.solvers.openhands_compaction.get_total_token_count", side_effect=[50, 10, 50, 10])
def test_build_compaction_events_hard_reset(mock_count: MagicMock, mock_llm: LLM) -> None:
    user = _user_message("task")
    agent = _agent_message("work")
    condensation = _condensation({user.id, agent.id}, summary_offset=0)
    events = [user, agent, condensation]
    entries = build_compaction_events_from_run(events, mock_llm, context_limit=128_000)
    _, compaction_event = entries[0]
    assert compaction_event.trigger == "openhands_condensation_hard_reset"
    assert compaction_event.openhands_extra is not None
    assert compaction_event.openhands_extra.hard_reset is True
    assert compaction_event.openhands_extra.reason == "hard_reset"
    assert compaction_event.openhands_extra.requirement == "hard"


@patch("nemo_evaluator.solvers.openhands_compaction.get_total_token_count", side_effect=[100, 20])
def test_splice_compaction_steps_into_trajectory(mock_count: MagicMock, mock_llm: LLM) -> None:
    user = _user_message("hello")
    condensation = _condensation({user.id})
    events = [user, CondensationRequest(), condensation]
    entries = build_compaction_events_from_run(events, mock_llm, context_limit=128_000)
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "steps": [
            {
                "step_id": 1,
                "timestamp": user.timestamp,
                "source": "user",
                "message": "hello",
            }
        ],
    }
    enriched = splice_compaction_steps_into_trajectory(trajectory, entries)
    assert len(enriched["steps"]) == 2
    compaction_step = enriched["steps"][1]
    assert compaction_step["source"] == "system"
    assert compaction_step["extra"]["context_management"]["type"] == "compaction"
    assert compaction_step["extra"]["compaction"]["trigger"] == "openhands_condensation_request"
    assert compaction_step["extra"]["compaction"]["strategy_details"]["reason"] == "request"
    assert compaction_step["extra"]["compaction"]["strategy_details"]["requirement"] == "hard"
    replacement = json.loads(compaction_step["observation"]["results"][0]["content"])
    assert any("condensed summary" in (message.get("content") or "") for message in replacement)


@patch("nemo_evaluator.solvers.openhands_compaction.get_total_token_count", side_effect=[100, 20])
def test_enrich_trajectory_no_condensation_is_noop(mock_count: MagicMock, mock_llm: LLM) -> None:
    trajectory = {"steps": [{"step_id": 1, "source": "user", "message": "hi"}]}
    result = enrich_trajectory_with_compaction(trajectory, [_user_message("hi")], mock_llm, 128_000)
    assert result is trajectory
    assert len(result["steps"]) == 1


@patch(
    "nemo_evaluator.solvers.openhands_compaction.get_total_token_count",
    side_effect=[100, 50, 100, 50, 80, 30, 80, 30],
)
def test_multiple_condensations_produce_multiple_events(mock_count: MagicMock, mock_llm: LLM) -> None:
    user1 = _user_message("task 1")
    agent1 = _agent_message("reply 1")
    cond1 = _condensation({user1.id}, summary="summary-1", response_id="resp-c1")
    user2 = _user_message("task 2")
    cond2 = _condensation({agent1.id}, summary="summary-2", response_id="resp-c2")
    events = [user1, agent1, cond1, user2, cond2]
    entries = build_compaction_events_from_run(events, mock_llm, context_limit=200_000)
    assert len(entries) == 2
    assert entries[0][1].compaction_index == 1
    assert entries[1][1].compaction_index == 2
    assert entries[0][1].strategy == "openhands_condensation"
    assert entries[1][1].strategy == "openhands_condensation"


def test_observation_extra_populated_from_llm_metrics(mock_llm: LLM) -> None:
    from types import SimpleNamespace

    usage = SimpleNamespace(
        response_id="resp-c1",
        prompt_tokens=4000,
        completion_tokens=200,
        cost=0.005,
    )
    mock_llm.metrics = SimpleNamespace(token_usages=[usage])

    from nemo_evaluator.solvers.openhands_compaction import _observation_extra_for_response

    obs_extra = _observation_extra_for_response(mock_llm, "resp-c1")
    assert obs_extra is not None
    assert obs_extra.compaction_prompt_tokens == 4000
    assert obs_extra.compaction_completion_tokens == 200
    assert abs(obs_extra.compaction_cost_usd - 0.005) < 1e-9


def test_observation_extra_none_when_response_id_missing(mock_llm: LLM) -> None:
    from nemo_evaluator.solvers.openhands_compaction import _observation_extra_for_response

    assert _observation_extra_for_response(mock_llm, None) is None
    assert _observation_extra_for_response(mock_llm, "nonexistent") is None


@patch(
    "nemo_evaluator.solvers.openhands_compaction.get_total_token_count",
    side_effect=[100, 20, 100, 20],
)
def test_second_condensation_segment_does_not_include_first(mock_count: MagicMock, mock_llm: LLM) -> None:
    """The second Condensation's trigger inference only looks at events since the first."""
    user = _user_message("task")
    cond1 = _condensation({user.id}, summary="s1")
    request = CondensationRequest()
    cond2 = _condensation(set(), summary="s2")
    events = [user, cond1, request, cond2]
    entries = build_compaction_events_from_run(events, mock_llm, context_limit=200_000)
    assert len(entries) == 2
    assert entries[1][1].trigger == "openhands_condensation_request"


@patch(
    "nemo_evaluator.solvers.openhands_compaction.get_total_token_count",
    side_effect=[100, 20],
)
def test_splice_compaction_step_id_renumbered(mock_count: MagicMock, mock_llm: LLM) -> None:
    user = _user_message("hi")
    cond = _condensation({user.id})
    entries = build_compaction_events_from_run([user, cond], mock_llm, context_limit=128_000)
    trajectory = {
        "steps": [
            {"step_id": 5, "timestamp": user.timestamp, "source": "user", "message": "hi"},
        ]
    }
    enriched = splice_compaction_steps_into_trajectory(trajectory, entries)
    for i, step in enumerate(enriched["steps"], start=1):
        assert step["step_id"] == i
