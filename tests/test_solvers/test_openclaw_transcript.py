# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenClaw session transcript persistence.

The PinchBench ``grade()`` contract (and other task-authored scorers)
reads ``<workspace>/.nel_transcript.jsonl`` in the upstream OpenClaw
envelope shape (``{"type": "message", "message": {...}}``).  The solver
must write that file verbatim from the session JSONL it recovers from
the container / local session store.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemo_evaluator.solvers.openclaw import (
    _TRANSCRIPT_FILENAME,
    _parse_session_jsonl,
    _persist_session_transcript,
)


class TestPersistSessionTranscript:
    def test_writes_bytes_verbatim(self, tmp_path: Path) -> None:
        """Raw JSONL is written unchanged so the envelope shape survives.

        ``grade()`` functions iterate top-level records looking for
        ``event["type"] == "message"`` and then descend into
        ``event["message"]["content"]`` blocks.  Any transformation here
        would invalidate that contract.
        """
        raw = "\n".join(
            [
                json.dumps(
                    {"type": "message", "message": {"role": "user", "content": "hi"}},
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "ok"},
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "bash",
                                    "input": {"command": "ls"},
                                },
                            ],
                        },
                    },
                ),
            ],
        )

        _persist_session_transcript(tmp_path, raw)

        out = tmp_path / _TRANSCRIPT_FILENAME
        assert out.exists()
        assert out.read_text(encoding="utf-8") == raw

    def test_filename_matches_scorer_contract(self) -> None:
        """Scorer expects this exact filename; guard against drift."""
        assert _TRANSCRIPT_FILENAME == ".nel_transcript.jsonl"

    def test_missing_workspace_is_silent_noop(self, tmp_path: Path) -> None:
        """A non-existent workspace must not raise — best-effort by design."""
        missing = tmp_path / "does-not-exist"
        _persist_session_transcript(missing, "anything")
        assert not missing.exists()

    def test_not_a_directory_is_silent_noop(self, tmp_path: Path) -> None:
        """If the ``workspace`` path is a regular file, we must not crash."""
        not_dir = tmp_path / "file.txt"
        not_dir.write_text("x")
        _persist_session_transcript(not_dir, "anything")
        assert not_dir.read_text() == "x"

    def test_overwrites_existing_transcript(self, tmp_path: Path) -> None:
        """A previous stale transcript (resume, retry) is replaced cleanly."""
        existing = tmp_path / _TRANSCRIPT_FILENAME
        existing.write_text("stale content that must be replaced")

        new = json.dumps(
            {"type": "message", "message": {"role": "user", "content": "fresh"}},
        )
        _persist_session_transcript(tmp_path, new)
        assert existing.read_text(encoding="utf-8") == new

    def test_empty_transcript_produces_empty_file(self, tmp_path: Path) -> None:
        """Empty string is still written — scorer treats missing vs empty alike."""
        _persist_session_transcript(tmp_path, "")
        assert (tmp_path / _TRANSCRIPT_FILENAME).read_text(encoding="utf-8") == ""

    def test_unicode_and_control_chars_preserved(self, tmp_path: Path) -> None:
        """Tool outputs often include UTF-8 text and embedded newlines; must round-trip."""
        raw = json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "tool",
                    "content": "emoji 🦀, accent é, newline preserved line:\nline2",
                },
            },
        )
        _persist_session_transcript(tmp_path, raw)
        out = (tmp_path / _TRANSCRIPT_FILENAME).read_text(encoding="utf-8")
        assert json.loads(out)["message"]["content"].startswith("emoji 🦀")


class TestParseSessionJsonl:
    def test_captures_thinking_block(self) -> None:
        raw = json.dumps(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "the chain of thought"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
                ],
            },
        )
        steps = _parse_session_jsonl(raw)
        assert len(steps) == 1
        assert steps[0]["reasoning_content"] == "the chain of thought"
        assert steps[0]["message"] == "answer"
        assert steps[0]["tool_calls"][0]["tool_call_id"] == "t1"

    def test_redacted_thinking_is_marked(self) -> None:
        raw = json.dumps(
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "<opaque>"},
                    {"type": "text", "text": "ok"},
                ],
            },
        )
        steps = _parse_session_jsonl(raw)
        assert steps[0]["reasoning_content"] == "[redacted_thinking]"

    def test_multiple_thinking_blocks_joined(self) -> None:
        raw = json.dumps(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "first"},
                    {"type": "text", "text": "answer"},
                    {"type": "thinking", "thinking": "second"},
                ],
            },
        )
        steps = _parse_session_jsonl(raw)
        assert steps[0]["reasoning_content"] == "first\nsecond"

    def test_thinking_only_message_is_kept(self) -> None:
        raw = json.dumps(
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "just a thought"}],
            },
        )
        steps = _parse_session_jsonl(raw)
        assert len(steps) == 1
        assert steps[0]["reasoning_content"] == "just a thought"
        assert steps[0]["message"] == ""

    def test_no_reasoning_field_when_absent(self) -> None:
        raw = json.dumps(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "plain"}],
            },
        )
        steps = _parse_session_jsonl(raw)
        assert "reasoning_content" not in steps[0]
