# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for openhands_compaction.py that run without openhands-sdk installed.

The module-level try/except ImportError guard means the module is importable
and several functions are usable with plain Python objects even when openhands
is absent.  This file covers those paths so codecov sees non-zero coverage on
every CI run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_evaluator.solvers.compaction_logging import CompactionEvent, CompactionTokens
from nemo_evaluator.solvers.openhands_compaction import (
    _observation_extra_for_response,
    _parse_timestamp,
    _resolve_condenser_config,
    _timestamp_sort_key,
    splice_compaction_steps_into_trajectory,
)


def _make_compaction_event(
    *,
    timestamp: str | None = "2025-01-01T00:00:00+00:00",
    compaction_index: int = 1,
) -> CompactionEvent:
    tokens = CompactionTokens(prompt_tokens_approx=1000, context_limit=128_000, free_tokens=127_000)
    return CompactionEvent(
        compaction_index=compaction_index,
        trigger="unknown",
        strategy="openhands_condensation",
        replacement_content="[]",
        tokens_before=tokens,
        tokens_after=tokens,
        llm_call_count=1,
        timestamp=timestamp,
    )


class TestParseTimestamp:
    def test_none_returns_none(self) -> None:
        assert _parse_timestamp(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_timestamp("") is None

    def test_z_suffix_accepted(self) -> None:
        result = _parse_timestamp("2025-01-15T10:00:00Z")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_utc_offset_accepted(self) -> None:
        result = _parse_timestamp("2025-06-01T12:30:00+00:00")
        assert result is not None
        assert result.hour == 12
        assert result.minute == 30

    def test_invalid_string_returns_none(self) -> None:
        assert _parse_timestamp("not-a-timestamp") is None

    def test_non_string_value_returns_none(self) -> None:
        assert _parse_timestamp("???") is None


class TestTimestampSortKey:
    def test_valid_sorts_before_none(self) -> None:
        valid_key = _timestamp_sort_key("2025-01-01T00:00:00Z")
        none_key = _timestamp_sort_key(None)
        assert valid_key < none_key

    def test_earlier_sorts_before_later(self) -> None:
        earlier = _timestamp_sort_key("2025-01-01T00:00:00Z")
        later = _timestamp_sort_key("2025-12-31T23:59:59Z")
        assert earlier < later

    def test_invalid_string_sorts_after_valid(self) -> None:
        valid_key = _timestamp_sort_key("2025-01-01T00:00:00Z")
        bad_key = _timestamp_sort_key("garbage")
        assert valid_key < bad_key

    def test_none_and_invalid_both_sort_last(self) -> None:
        valid_key = _timestamp_sort_key("2025-06-01T00:00:00Z")
        none_key = _timestamp_sort_key(None)
        bad_key = _timestamp_sort_key("bad")
        assert valid_key < none_key
        assert valid_key < bad_key


class TestResolveCondenserConfig:
    def test_none_returns_defaults(self) -> None:
        cfg = _resolve_condenser_config(None)
        assert cfg["max_size"] == 80
        assert cfg["max_tokens"] is None
        assert cfg["keep_first"] == 4

    def test_empty_dict_returns_defaults(self) -> None:
        cfg = _resolve_condenser_config({})
        assert cfg["max_size"] == 80
        assert cfg["max_tokens"] is None
        assert cfg["keep_first"] == 4

    def test_max_size_override(self) -> None:
        cfg = _resolve_condenser_config({"max_size": 50})
        assert cfg["max_size"] == 50
        assert cfg["max_tokens"] is None
        assert cfg["keep_first"] == 4

    def test_max_tokens_override(self) -> None:
        cfg = _resolve_condenser_config({"max_tokens": 100_000})
        assert cfg["max_tokens"] == 100_000
        assert cfg["max_size"] == 80

    def test_full_override(self) -> None:
        cfg = _resolve_condenser_config({"max_size": 100, "max_tokens": 50_000, "keep_first": 2})
        assert cfg["max_size"] == 100
        assert cfg["max_tokens"] == 50_000
        assert cfg["keep_first"] == 2

    def test_string_values_coerced_to_int(self) -> None:
        cfg = _resolve_condenser_config({"max_size": "60", "max_tokens": "100000", "keep_first": "3"})
        assert cfg["max_size"] == 60
        assert cfg["max_tokens"] == 100_000
        assert cfg["keep_first"] == 3


class TestObservationExtraForResponse:
    def test_none_response_id_returns_none(self) -> None:
        llm = SimpleNamespace(metrics=SimpleNamespace(token_usages=[]))
        assert _observation_extra_for_response(llm, None) is None

    def test_empty_string_response_id_returns_none(self) -> None:
        llm = SimpleNamespace(metrics=SimpleNamespace(token_usages=[]))
        assert _observation_extra_for_response(llm, "") is None

    def test_no_matching_usage_returns_none(self) -> None:
        usage = SimpleNamespace(response_id="other-id", prompt_tokens=100, completion_tokens=10, cost=0.001)
        llm = SimpleNamespace(metrics=SimpleNamespace(token_usages=[usage]))
        assert _observation_extra_for_response(llm, "resp-xyz") is None

    def test_matching_usage_populates_extra(self) -> None:
        usage = SimpleNamespace(response_id="resp-1", prompt_tokens=4000, completion_tokens=200, cost=0.005)
        llm = SimpleNamespace(metrics=SimpleNamespace(token_usages=[usage]))
        result = _observation_extra_for_response(llm, "resp-1")
        assert result is not None
        assert result.compaction_prompt_tokens == 4000
        assert result.compaction_completion_tokens == 200
        assert abs(result.compaction_cost_usd - 0.005) < 1e-9

    def test_missing_metrics_attribute_returns_none(self) -> None:
        llm = SimpleNamespace()
        assert _observation_extra_for_response(llm, "resp-1") is None

    def test_none_metrics_returns_none(self) -> None:
        llm = SimpleNamespace(metrics=None)
        assert _observation_extra_for_response(llm, "resp-1") is None

    def test_none_token_usages_returns_none(self) -> None:
        llm = SimpleNamespace(metrics=SimpleNamespace(token_usages=None))
        assert _observation_extra_for_response(llm, "resp-1") is None

    def test_first_match_wins_among_multiple_usages(self) -> None:
        usage_a = SimpleNamespace(response_id="resp-1", prompt_tokens=100, completion_tokens=10, cost=0.001)
        usage_b = SimpleNamespace(response_id="resp-1", prompt_tokens=999, completion_tokens=99, cost=0.999)
        llm = SimpleNamespace(metrics=SimpleNamespace(token_usages=[usage_a, usage_b]))
        result = _observation_extra_for_response(llm, "resp-1")
        assert result is not None
        assert result.compaction_prompt_tokens == 100


class TestSpliceCompactionStepsIntoTrajectory:
    def test_empty_entries_returns_same_object(self) -> None:
        trajectory = {"steps": [{"step_id": 1, "source": "user", "message": "hi"}]}
        result = splice_compaction_steps_into_trajectory(trajectory, [])
        assert result is trajectory

    def test_compaction_appended_after_earlier_step(self) -> None:
        user_step = {"step_id": 1, "timestamp": "2025-01-01T00:01:00Z", "source": "user", "message": "hi"}
        trajectory = {"steps": [user_step]}
        event = _make_compaction_event(timestamp="2025-01-01T00:02:00Z")
        result = splice_compaction_steps_into_trajectory(trajectory, [("2025-01-01T00:02:00Z", event)])
        assert len(result["steps"]) == 2
        assert result["steps"][0]["source"] == "user"
        assert result["steps"][1]["source"] == "system"

    def test_compaction_inserted_before_later_step(self) -> None:
        user_step = {"step_id": 1, "timestamp": "2025-01-01T00:10:00Z", "source": "user", "message": "hi"}
        trajectory = {"steps": [user_step]}
        event = _make_compaction_event(timestamp="2025-01-01T00:05:00Z")
        result = splice_compaction_steps_into_trajectory(trajectory, [("2025-01-01T00:05:00Z", event)])
        assert len(result["steps"]) == 2
        assert result["steps"][0]["source"] == "system"
        assert result["steps"][1]["source"] == "user"

    def test_step_ids_renumbered_from_one(self) -> None:
        steps = [
            {"step_id": 5, "timestamp": "2025-01-01T00:01:00Z", "source": "user", "message": "a"},
            {"step_id": 10, "timestamp": "2025-01-01T00:03:00Z", "source": "agent", "message": "b"},
        ]
        trajectory = {"steps": steps}
        event = _make_compaction_event(timestamp="2025-01-01T00:02:00Z")
        result = splice_compaction_steps_into_trajectory(trajectory, [("2025-01-01T00:02:00Z", event)])
        assert len(result["steps"]) == 3
        for i, step in enumerate(result["steps"], start=1):
            assert step["step_id"] == i

    def test_compaction_step_has_system_source_and_compaction_extra(self) -> None:
        trajectory = {"steps": []}
        event = _make_compaction_event(timestamp="2025-01-01T00:01:00Z")
        result = splice_compaction_steps_into_trajectory(trajectory, [("2025-01-01T00:01:00Z", event)])
        assert len(result["steps"]) == 1
        step = result["steps"][0]
        assert step["source"] == "system"
        assert step["extra"]["context_management"]["type"] == "compaction"
        assert step["extra"]["compaction"]["trigger"] == "unknown"
        assert step["extra"]["compaction"]["strategy"] == "openhands_condensation"

    def test_multiple_compaction_entries_ordered_by_timestamp(self) -> None:
        trajectory = {"steps": []}
        event_a = _make_compaction_event(timestamp="2025-01-01T00:02:00Z", compaction_index=1)
        event_b = _make_compaction_event(timestamp="2025-01-01T00:01:00Z", compaction_index=2)
        entries = [
            ("2025-01-01T00:02:00Z", event_a),
            ("2025-01-01T00:01:00Z", event_b),
        ]
        result = splice_compaction_steps_into_trajectory(trajectory, entries)
        assert len(result["steps"]) == 2
        assert result["steps"][0]["extra"]["compaction"]["compaction_index"] == 2
        assert result["steps"][1]["extra"]["compaction"]["compaction_index"] == 1

    def test_none_timestamp_event_appended_last(self) -> None:
        user_step = {"step_id": 1, "timestamp": "2025-01-01T00:01:00Z", "source": "user", "message": "hi"}
        trajectory = {"steps": [user_step]}
        event = _make_compaction_event(timestamp=None)
        result = splice_compaction_steps_into_trajectory(trajectory, [(None, event)])
        assert len(result["steps"]) == 2
        assert result["steps"][-1]["source"] == "system"

    @pytest.mark.parametrize(
        "steps_value",
        [None, []],
        ids=["steps_none", "steps_empty"],
    )
    def test_missing_or_empty_steps_list(self, steps_value: list | None) -> None:
        trajectory: dict = {"steps": steps_value}
        event = _make_compaction_event(timestamp="2025-01-01T00:01:00Z")
        result = splice_compaction_steps_into_trajectory(trajectory, [("2025-01-01T00:01:00Z", event)])
        assert len(result["steps"]) == 1
        assert result["steps"][0]["source"] == "system"
