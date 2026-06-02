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
"""Unit tests for the pure helpers in ``openclaw_helpers``.

These cover the deterministic output/workspace/transcript utilities that the
OpenClaw solver composes. The async solver loop is exercised elsewhere; here we
pin the small, pure pieces it relies on.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path, PurePosixPath

import pytest

from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import ErrorKind
from nemo_evaluator.solvers.openclaw_helpers import (
    _CONTAINER_SESSIONS_DIR,
    _CONTAINER_WORKSPACE,
    _coerce_timeout,
    _container_session_find_command,
    _container_workspace_find_command,
    _decode_b64_text,
    _finalize_openclaw_solve,
    _format_timeout_seconds,
    _merge_model_responses,
    _openclaw_error_kind,
    _openclaw_retryable_response_error,
    _parse_response,
    _parse_session_jsonl,
    _read_workspace_files,
    _redact_secret,
    _resolve_container_session_path,
    _resolve_local_session_path,
    _sandbox_agent_exec_timeout,
    _session_path_candidates_from_index,
    _unique_strings,
    _usage_int,
    _workspace_rel_is_skipped,
)


class TestScalarHelpers:
    @pytest.mark.parametrize(
        ("timeout", "expected"),
        [
            (90, "90s"),
            (60.0, "60s"),
            (1.5, "1.5s"),
            (0.25, "0.25s"),
            (12.500, "12.5s"),
        ],
        ids=["int", "int-from-float", "one-decimal", "two-decimals", "trailing-zeros-stripped"],
    )
    def test_format_timeout_seconds(self, timeout: float, expected: str) -> None:
        assert _format_timeout_seconds(timeout) == expected

    def test_sandbox_exec_timeout_adds_grace(self) -> None:
        # The sandbox-level exec timeout must exceed the agent timeout so the
        # agent's own SIGTERM wins the race and we capture its partial output.
        assert _sandbox_agent_exec_timeout(100.0) > 100.0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("30", 30.0),
            (12.5, 12.5),
            (0, None),
            (-5, None),
            ("not-a-number", None),
            (None, None),
        ],
        ids=["str", "float", "zero", "negative", "unparseable", "none"],
    )
    def test_coerce_timeout(self, value: object, expected: float | None) -> None:
        assert _coerce_timeout(value) == expected

    @pytest.mark.parametrize(
        ("text", "secret", "expected"),
        [
            ("auth=sk-abc123 done", "sk-abc123", "auth=[REDACTED] done"),
            ("nothing secret", None, "nothing secret"),
            ("empty secret", "", "empty secret"),
        ],
        ids=["redacted", "none-secret", "empty-secret"],
    )
    def test_redact_secret(self, text: str, secret: str | None, expected: str) -> None:
        assert _redact_secret(text, secret) == expected

    def test_decode_b64_text_tolerates_surrounding_whitespace(self) -> None:
        raw = "  " + base64.b64encode("héllo".encode()).decode() + "\n"
        assert _decode_b64_text(raw) == "héllo"

    def test_unique_strings_preserves_order_and_drops_blanks(self) -> None:
        assert _unique_strings(["a", "b", "a", "", "c", "b"]) == ["a", "b", "c"]


class TestWorkspacePathHelpers:
    @pytest.mark.parametrize(
        ("rel", "skipped"),
        [
            ("src/main.py", False),
            (".git/config", True),
            ("node_modules/pkg/index.js", True),
            ("a/__pycache__/b.pyc", True),
            ("results.txt", False),
        ],
        ids=["normal", "git", "node_modules", "nested-pycache", "plain-file"],
    )
    def test_workspace_rel_is_skipped(self, rel: str, skipped: bool) -> None:
        assert _workspace_rel_is_skipped(rel) is skipped
        # PurePosixPath input must behave identically to the string form.
        assert _workspace_rel_is_skipped(PurePosixPath(rel)) is skipped

    def test_container_workspace_find_command_quotes_workspace_and_prunes(self) -> None:
        cmd = _container_workspace_find_command()
        assert _CONTAINER_WORKSPACE in cmd
        assert "-prune" in cmd
        assert "node_modules" in cmd and ".git" in cmd
        assert cmd.rstrip().endswith("|| true")

    def test_container_session_find_command_sorts_newest_first(self) -> None:
        cmd = _container_session_find_command()
        assert _CONTAINER_SESSIONS_DIR in cmd
        assert "*.jsonl" in cmd and "*.ndjson" in cmd
        assert "sort -nr" in cmd

    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            ("/abs/path/session.jsonl", "/abs/path/session.jsonl"),
            ("session.jsonl", f"{_CONTAINER_SESSIONS_DIR}/session.jsonl"),
        ],
        ids=["absolute", "relative"],
    )
    def test_resolve_container_session_path(self, candidate: str, expected: str) -> None:
        assert _resolve_container_session_path(candidate) == expected

    def test_resolve_local_session_path(self, tmp_path: Path) -> None:
        assert _resolve_local_session_path(tmp_path, "s.jsonl") == tmp_path / "s.jsonl"
        absolute = tmp_path / "elsewhere" / "s.jsonl"
        assert _resolve_local_session_path(tmp_path, str(absolute)) == absolute


class TestSessionPathCandidatesFromIndex:
    def test_invalid_json_returns_empty(self) -> None:
        assert _session_path_candidates_from_index("not json", "abc") == []

    def test_extracts_jsonl_paths_and_dedups(self) -> None:
        raw = json.dumps(
            {
                "path": "/sessions/abc.jsonl",
                "irrelevant": "notes.txt",
                "dup": "/sessions/abc.jsonl",
            }
        )
        assert _session_path_candidates_from_index(raw, "abc") == ["/sessions/abc.jsonl"]

    def test_matches_session_id_inside_nested_record(self) -> None:
        raw = json.dumps({"sessions": [{"id": "target", "sessionFile": "target.ndjson"}]})
        assert "target.ndjson" in _session_path_candidates_from_index(raw, "target")


class TestReadWorkspaceFiles:
    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        assert _read_workspace_files(tmp_path / "does-not-exist") == ""

    def test_reads_only_new_text_files(self, tmp_path: Path) -> None:
        (tmp_path / "new.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "pre.txt").write_text("old", encoding="utf-8")
        (tmp_path / "image.bin").write_text("binarydata", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")

        out = _read_workspace_files(tmp_path, pre_existing={"pre.txt"})

        assert "--- new.txt ---\nhello" in out
        assert "old" not in out  # pre-existing file excluded
        assert "binarydata" not in out  # non-text extension excluded
        assert "secret" not in out  # skip-dir excluded

    def test_skips_files_over_size_cap(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("x" * 50_001, encoding="utf-8")
        assert _read_workspace_files(tmp_path) == ""


class TestParseResponse:
    def test_extracts_text_usage_steps_and_extra(self) -> None:
        output = {
            "payloads": [
                {"text": "first"},
                {"text": "boom", "isError": True},
                {"mediaUrl": "http://img/1.png"},
            ],
            "meta": {
                "durationMs": 1500,
                "usage": {"inputTokens": 10, "outputTokens": 20},
                "agentMeta": {"model": "m", "provider": "p", "sessionId": "s1"},
            },
        }
        text, model_response, steps, oc_extra = _parse_response(output)

        assert text == "first\nboom"
        assert model_response.prompt_tokens == 10
        assert model_response.completion_tokens == 20
        assert model_response.total_tokens == 30  # derived when not reported
        assert model_response.latency_ms == 1500.0
        assert len(steps) == 3
        assert steps[-1]["extra"]["media_urls"] == ["http://img/1.png"]
        assert oc_extra is not None
        assert oc_extra["payload_errors"] == ["boom"]
        assert oc_extra["openclaw_session_id"] == "s1"

    def test_empty_output_yields_no_text_and_no_extra(self) -> None:
        text, model_response, steps, oc_extra = _parse_response({})
        assert text == ""
        assert steps == []
        assert oc_extra is None
        assert model_response.content == ""

    @pytest.mark.parametrize(
        ("usage", "keys", "expected"),
        [
            ({"inputTokens": 5}, ("inputTokens", "prompt_tokens"), 5),
            ({"prompt_tokens": "7"}, ("inputTokens", "prompt_tokens"), 7),
            ({"inputTokens": None}, ("inputTokens",), None),
            ({"inputTokens": "bad"}, ("inputTokens",), None),
            ({}, ("inputTokens",), None),
        ],
        ids=["first-key", "fallback-key-coerced", "explicit-null", "unparseable", "missing"],
    )
    def test_usage_int(self, usage: dict, keys: tuple[str, ...], expected: int | None) -> None:
        assert _usage_int(usage, *keys) == expected


class TestMergeModelResponses:
    def test_empty_list_returns_bare_content(self) -> None:
        merged = _merge_model_responses([], content="only-content")
        assert merged.content == "only-content"
        assert merged.total_tokens is None

    def test_sums_present_usage_and_keeps_last_metadata(self) -> None:
        responses = [
            ModelResponse(
                content="a", model="m1", prompt_tokens=1, completion_tokens=2, total_tokens=3, latency_ms=100.0
            ),
            ModelResponse(
                content="b",
                model="m2",
                finish_reason="stop",
                prompt_tokens=4,
                completion_tokens=None,
                total_tokens=5,
                latency_ms=50.0,
            ),
        ]
        merged = _merge_model_responses(responses, content="merged")

        assert merged.content == "merged"
        assert merged.model == "m2"  # metadata from the final turn
        assert merged.finish_reason == "stop"
        assert merged.prompt_tokens == 5
        assert merged.completion_tokens == 2  # None contributions ignored
        assert merged.total_tokens == 8
        assert merged.reasoning_tokens is None  # absent on every turn
        assert merged.latency_ms == 150.0  # always summed


class TestErrorClassification:
    @pytest.mark.parametrize(
        ("message", "return_code", "expected"),
        [
            ("anything", 124, ErrorKind.INFRA),
            ("anything", 137, ErrorKind.INFRA),
            ("agent timed out after 60s", None, ErrorKind.INFRA),
            ("hit ECONNRESET mid-stream", None, ErrorKind.INFRA),
            ("model said no", 0, ErrorKind.GRACEFUL),
        ],
        ids=["sigkill-124", "oom-137", "timeout-text", "retryable-fragment", "graceful"],
    )
    def test_openclaw_error_kind(self, message: str, return_code: int | None, expected: ErrorKind) -> None:
        assert _openclaw_error_kind(message, return_code=return_code) is expected

    def test_empty_response_without_addendum_is_retryable(self) -> None:
        result = _openclaw_retryable_response_error("", has_file_addendum=False)
        assert result is not None
        _, kind = result
        assert kind is ErrorKind.INFRA

    def test_empty_response_with_addendum_is_not_retryable(self) -> None:
        # Files were written even though no visible text came back -> usable.
        assert _openclaw_retryable_response_error("", has_file_addendum=True) is None

    def test_retryable_fragment_returns_first_line(self) -> None:
        result = _openclaw_retryable_response_error("ECONNRESET\ntrailing detail")
        assert result == ("ECONNRESET", ErrorKind.INFRA)

    def test_agent_couldnt_respond_banner_is_retryable(self) -> None:
        result = _openclaw_retryable_response_error("⚠️ Agent couldn't generate a response now")
        assert result is not None
        assert result[1] is ErrorKind.INFRA

    def test_normal_response_is_not_retryable(self) -> None:
        assert _openclaw_retryable_response_error("the answer is 42") is None


class TestFinalizeOpenclawSolve:
    def _finalize(self, **overrides):
        kwargs = dict(
            mode="sandbox",
            task_id="t1",
            workspace_path=None,
            pre_existing_files=set(),
            response_texts=[],
            model_responses=[],
            envelope_steps=[],
            oc_extra=None,
            session_ids=[],
            effective_prompts=["solve it"],
            raw_session_jsonl="",
        )
        kwargs.update(overrides)
        return _finalize_openclaw_solve(**kwargs)

    def test_happy_path_builds_clean_result(self) -> None:
        result = self._finalize(
            response_texts=["the answer"],
            model_responses=[ModelResponse(content="the answer", model="m", total_tokens=10, latency_ms=5.0)],
        )
        assert result.response == "the answer"
        assert result.error is None
        assert result.error_kind is ErrorKind.NONE
        assert result.model_response is not None
        assert result.model_response.content == "the answer"
        assert result.trajectory  # ATIF steps were assembled

    def test_empty_output_surfaces_retryable_infra_error(self) -> None:
        result = self._finalize()
        assert result.error is not None
        assert result.error_kind is ErrorKind.INFRA


class TestParseSessionJsonlToolResults:
    """The assistant/thinking paths live in ``test_openclaw_transcript``; here we
    pin the tool-result attachment and malformed-input handling."""

    def test_tool_result_attaches_to_preceding_agent_step(self) -> None:
        raw = "\n".join(
            json.dumps(rec)
            for rec in [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "running ls"},
                        {"type": "tool_use", "id": "c1", "name": "bash", "input": {"cmd": "ls"}},
                    ],
                },
                {"role": "tool", "tool_use_id": "c1", "content": [{"type": "text", "text": "file1\nfile2"}]},
            ]
        )
        steps = _parse_session_jsonl(raw)
        assert len(steps) == 1
        assert steps[0]["tool_calls"][0]["tool_call_id"] == "c1"
        observation = steps[0]["observation"]["results"][0]
        assert observation["content"] == "file1\nfile2"
        assert observation["source_call_id"] == "c1"

    def test_orphan_tool_result_becomes_system_step(self) -> None:
        raw = json.dumps({"role": "tool", "tool_use_id": "x", "content": "orphaned output"})
        steps = _parse_session_jsonl(raw)
        assert steps == [{"source": "system", "message": "orphaned output", "extra": {"tool_use_id": "x"}}]

    def test_unwraps_message_envelope(self) -> None:
        raw = json.dumps(
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "wrapped"}]}}
        )
        steps = _parse_session_jsonl(raw)
        assert steps == [{"source": "agent", "message": "wrapped"}]

    def test_blank_and_malformed_lines_are_skipped(self) -> None:
        raw = "\n".join(
            [
                "",
                "{not valid json",
                json.dumps({"role": "assistant", "content": [{"type": "text", "text": "ok"}]}),
            ]
        )
        steps = _parse_session_jsonl(raw)
        assert len(steps) == 1
        assert steps[0]["message"] == "ok"
