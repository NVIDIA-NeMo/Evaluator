# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PinchBench judge helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from nemo_evaluator.benchmarks.pinchbench import (
    _CACHE_METADATA,
    _TRANSCRIPT_SUMMARY_MAX_CHARS,
    _WORKSPACE_BLOB_MAX_CHARS,
    _build_judge_prompt,
    _cache_key,
    _load_all_tasks,
    _make_pinchbench_judge_fn,
    _manifest_task_files,
    _parse_task,
    _read_workspace_files_for_judge,
    _seed_fn,
    _summarize_transcript,
)


class TestSummarizeTranscript:
    def test_user_content_dict_is_unwrapped(self) -> None:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello there"}],
                },
            }
        ]
        out = _summarize_transcript(transcript)
        assert "User: hello there" in out
        assert "{'type'" not in out and "'text':" not in out

    def test_user_content_plain_string_preserved(self) -> None:
        out = _summarize_transcript(
            [
                {
                    "type": "message",
                    "message": {"role": "user", "content": "plain"},
                }
            ]
        )
        assert out == "User: plain"

    def test_total_ceiling_applied_with_truncation_marker(self) -> None:
        big_text = "x" * 5000
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": big_text}],
                },
            }
            for _ in range(100)
        ]
        out = _summarize_transcript(transcript)
        assert "[transcript truncated]" in out
        assert len(out) <= _TRANSCRIPT_SUMMARY_MAX_CHARS + 64

    def test_empty_transcript_returns_empty_string(self) -> None:
        assert _summarize_transcript([]) == ""

    def test_non_dict_events_skipped(self) -> None:
        assert _summarize_transcript(["not a dict", None, 7]) == ""  # type: ignore[list-item]


class TestReadWorkspaceFiles:
    def test_binary_file_is_listed_not_dropped(self, tmp_path: Path) -> None:
        (tmp_path / "report.pdf").write_bytes(b"\xff\xfe\xfd\xfc fakepdf")
        out = _read_workspace_files_for_judge(str(tmp_path))
        assert "report.pdf" in out
        assert "binary or unreadable" in out

    def test_boilerplate_and_transcript_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "BOOTSTRAP.md").write_text("boilerplate")
        (tmp_path / ".nel_transcript.jsonl").write_text('{"type":"message"}')
        (tmp_path / "deliverable.md").write_text("the answer")

        out = _read_workspace_files_for_judge(str(tmp_path))
        assert "deliverable.md" in out
        assert "BOOTSTRAP.md" not in out
        assert ".nel_transcript.jsonl" not in out

    def test_total_cap_stops_listing_and_appends_marker(self, tmp_path: Path) -> None:
        for i in range(200):
            (tmp_path / f"f{i:03d}.txt").write_text("z" * 2000)

        out = _read_workspace_files_for_judge(str(tmp_path))
        assert "[workspace listing truncated]" in out
        assert len(out) <= _WORKSPACE_BLOB_MAX_CHARS + 4096

    def test_missing_workspace_returns_empty(self, tmp_path: Path) -> None:
        assert _read_workspace_files_for_judge("") == ""
        assert _read_workspace_files_for_judge(str(tmp_path / "nope")) == ""


class TestTaskMetadata:
    def test_frontmatter_runtime_metadata_is_preserved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("nemo_evaluator.benchmarks.pinchbench._ensure_cache", lambda **kwargs: tmp_path)
        task_file = tmp_path / "task_second_brain.md"
        task_file.write_text(
            """---
id: task_second_brain
name: Second Brain
category: memory
grading_type: hybrid
timeout_seconds: 321
multi_session: true
sessions:
  - id: store
    prompt: Save the facts.
  - id: recall
    new_session: true
    prompt: Read the facts.
prerequisites:
  - npm:@juppytt/fws
workspace_files: []
---

## Prompt

This is a multi-session task.

## Grading Criteria

- Creates a memory file.
""",
            encoding="utf-8",
        )

        row = _parse_task(task_file)
        seed = _seed_fn(row, 0)

        assert row["multi_session"] is True
        assert row["timeout_seconds"] == 321
        assert row["sessions"][1]["new_session"] is True
        assert row["prerequisites"] == ["npm:@juppytt/fws"]
        assert seed.metadata["timeout_seconds"] == 321
        assert seed.metadata["multi_session"] is True
        assert seed.metadata["sessions"] == row["sessions"]
        assert seed.metadata["prerequisites"] == row["prerequisites"]
        assert "pinchbench_source_ref" in seed.metadata

    def test_source_metadata_is_preserved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "task_source.md").write_text(
            """---
id: task_source
name: Source Metadata
category: metadata
grading_type: automated
workspace_files: []
---

## Prompt

Check metadata.

## Automated Checks

```python
def grade(transcript, workspace_path):
    return {"ok": 1.0}
```
""",
            encoding="utf-8",
        )
        source_url = "https://example.test/pinchbench/skill"
        source_ref = "abc123"
        _CACHE_METADATA[_cache_key(source_ref, source_url)] = {
            "source_url": source_url,
            "source_ref": source_ref,
            "archive_url": f"{source_url}/archive/{source_ref}.zip",
            "benchmark_version": "2.0.0",
        }
        monkeypatch.setattr("nemo_evaluator.benchmarks.pinchbench._ensure_cache", lambda **kwargs: tmp_path)

        rows = _load_all_tasks(source_ref=source_ref, source_url=source_url)

        assert rows[0]["pinchbench_source_ref"] == source_ref
        assert rows[0]["pinchbench_source_url"] == source_url
        assert rows[0]["pinchbench_benchmark_version"] == "2.0.0"
        seed = _seed_fn(rows[0], 0)
        assert seed.metadata["pinchbench_source_ref"] == source_ref
        assert seed.metadata["pinchbench_benchmark_version"] == "2.0.0"

    def test_manifest_order_is_used_before_sorted_fallback(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        for name in ("task_zeta.md", "task_alpha.md", "task_extra.md"):
            (tasks_dir / name).write_text("---\nid: x\n---\n\n## Prompt\n\nx\n", encoding="utf-8")
        (tasks_dir / "manifest.yaml").write_text(
            """
run_first:
  - task_zeta
categories:
  core:
    - task_alpha.md
""",
            encoding="utf-8",
        )

        ordered = [path.name for path in _manifest_task_files(tasks_dir)]

        assert ordered == ["task_zeta.md", "task_alpha.md", "task_extra.md"]

    def test_judge_prompt_comes_from_upstream_grading_helper(self, tmp_path) -> None:
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "lib_grading.py").write_text(
            """
def _build_judge_prompt(task, transcript_summary, rubric, workspace_content=""):
    return (
        f"UPSTREAM:{task.prompt}|{task.expected_behavior}|{transcript_summary}|{rubric}|{workspace_content}\\n"
        "Do not penalize the agent's tool usage as a rule violation; "
        "the rules above apply only to you, the grader"
    )
""",
            encoding="utf-8",
        )

        prompt = _build_judge_prompt(
            {"prompt": "do work", "expected_behavior": "done"},
            "Agent used bash and wrote a file.",
            "Score it.",
            "### File: report.md\nok",
            source_root=tmp_path,
        )

        assert prompt.startswith("UPSTREAM:do work|done|")
        assert "Do not penalize the agent's tool usage as a rule violation" in prompt
        assert "the rules above apply only to you, the grader" in prompt


@dataclass
class _FakeResp:
    content: str
    model: str = "fake-judge"
    total_tokens: int = 10
    latency_ms: float = 1.0


class _AlwaysRaisingJudge:
    async def chat(self, *, prompt: str, system: str) -> _FakeResp:  # noqa: ARG002
        raise RuntimeError("gateway 429")


class _OkJudge:
    def __init__(self, content: str) -> None:
        self._content = content

    async def chat(self, *, prompt: str, system: str) -> _FakeResp:  # noqa: ARG002
        return _FakeResp(content=self._content)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_evaluator.benchmarks.pinchbench._JUDGE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr("nemo_evaluator.benchmarks.pinchbench._JUDGE_BASE_DELAY_S", 0.0)
    monkeypatch.setattr("nemo_evaluator.benchmarks.pinchbench._JUDGE_MAX_DELAY_S", 0.0)


class TestJudgeFnFallback:
    def test_hybrid_falls_back_to_automated_on_repeated_failure(self) -> None:
        fn = _make_pinchbench_judge_fn(
            judge_prompt="prompt",
            automated_score=0.7,
            grading_weights=None,
            grading_type="hybrid",
        )
        result = asyncio.run(fn(_AlwaysRaisingJudge()))
        assert result["reward"] == pytest.approx(0.7)
        assert result["judge"]["total"] is None
        assert "error" in result["judge"]
        assert result["details"]["hybrid_combine"]["judge_failed"] is True
        assert result["details"]["hybrid_combine"]["automated"] == pytest.approx(0.7)

    def test_llm_judge_returns_none_reward_on_failure(self) -> None:
        fn = _make_pinchbench_judge_fn(
            judge_prompt="prompt",
            automated_score=None,
            grading_weights=None,
            grading_type="llm_judge",
        )
        result = asyncio.run(fn(_AlwaysRaisingJudge()))
        assert result["reward"] is None
        assert "error" in result["judge"]

    def test_happy_path_parses_and_combines(self) -> None:
        payload = '{"scores": {"a": 1.0, "b": 0.5}, "total": 0.75, "notes": "ok"}'
        fn = _make_pinchbench_judge_fn(
            judge_prompt="prompt",
            automated_score=0.5,
            grading_weights={"automated": 0.5, "llm_judge": 0.5},
            grading_type="hybrid",
        )
        result = asyncio.run(fn(_OkJudge(payload)))
        assert result["reward"] == pytest.approx(0.625)
        assert result["judge"]["total"] == pytest.approx(0.75)
        assert result["details"]["hybrid_combine"]["judge"] == pytest.approx(0.75)
