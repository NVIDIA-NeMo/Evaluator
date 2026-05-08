# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``## Hints`` strip applied to Harbor ``instruction.md``."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_evaluator.environments.harbor import (
    HarborEnvironment,
    _is_swebench_multilingual,
    _strip_swebench_hints_block,
)


def test_strip_no_marker_returns_original():
    text = "Just a plain problem statement.\nNo hints here."
    out, stripped = _strip_swebench_hints_block(text)
    assert out == text
    assert stripped is False


def test_strip_removes_hints_block_exact_adapter_format():
    text = (
        "Fix the bug in foo.py.\n"
        "Repro steps:\n"
        "  1. call foo(1)\n"
        "  2. observe crash"
        "\n\n## Hints\n\n"
        "Maintainer: the fix is in bar.py line 42. Change x -> y."
    )
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is True
    assert "## Hints" not in out
    assert "Maintainer" not in out
    assert "fix is in bar.py" not in out
    assert out.endswith("observe crash")


def test_strip_ignores_hints_heading_with_wrong_surrounding_whitespace():
    text = "Some issue body.\n## Hints might be relevant here\nand more text."
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is False
    assert out == text


def test_strip_ignores_inline_hints_mention():
    text = "The word ## Hints appears but not as a heading."
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is False
    assert out == text


def test_strip_preserves_leading_content_exactly():
    body = "line1\nline2\n```py\nprint('x')\n```"
    text = body + "\n\n## Hints\n\nspoiler content"
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is True
    assert out == body


def test_strip_handles_marker_at_very_start():
    text = "\n\n## Hints\n\nnothing before."
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is True
    assert out == ""


def test_strip_handles_multiple_markers_uses_first():
    text = "body\n\n## Hints\n\nfirst\n\n## Hints\n\nsecond"
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is True
    assert out == "body"


def test_strip_empty_instruction_is_noop():
    out, stripped = _strip_swebench_hints_block("")
    assert stripped is False
    assert out == ""


def test_strip_handles_canary_empty_hints_body():
    """Adapter with an empty ``hints_text`` leaves a bare ``\\n\\n## Hints`` tail."""
    text = "Problem body.\n\n## Hints"
    out, stripped = _strip_swebench_hints_block(text)
    assert stripped is True
    assert out == "Problem body."


@pytest.mark.parametrize(
    "name",
    [
        "swebench_multilingual",
        "swebench_multilingual@1.0",
        "swebench-multilingual@2.0",
        "SWEBENCH_MULTILINGUAL@1.0",
    ],
)
def test_gate_matches_multilingual_variants(name):
    assert _is_swebench_multilingual(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "swebench-verified@1.0",
        "swebenchpro@1.0",
        "swesmith@1.0",
        "swe-gen-js@1.0",
        "swe-lancer-diamond@ic",
        "terminal-bench@2.0",
        "usaco@1.0",
        "ds",
        "",
    ],
)
def test_gate_rejects_non_multilingual_datasets(name):
    assert _is_swebench_multilingual(name) is False


def _make_task(root: Path, name: str, instruction: str, task_toml: str = "") -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "task.toml").write_text(task_toml)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


@pytest.mark.asyncio
async def test_seed_strips_multilingual_hints_block(tmp_path):
    dataset = tmp_path / "swebench_multilingual@1.0"
    _make_task(
        dataset,
        "apache__druid-12345",
        "Original bug report.\nSteps to reproduce.\n\n## Hints\n\nReviewer said: patch should go in X.java line 10.",
    )
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "## Hints" not in seed.prompt
    assert "Reviewer said" not in seed.prompt
    assert seed.prompt.endswith("Steps to reproduce.")


@pytest.mark.asyncio
async def test_seed_leaves_verified_instruction_untouched(tmp_path):
    dataset = tmp_path / "swebench-verified@1.0"
    _make_task(
        dataset,
        "django__django-12345",
        "Simplify signature of DatabaseOperations.execute_sql_flush().\nThe using argument can be dropped.",
    )
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "## Hints" not in seed.prompt
    assert "execute_sql_flush" in seed.prompt
    assert seed.prompt.startswith("Simplify signature")


@pytest.mark.asyncio
async def test_seed_gate_preserves_hints_for_non_multilingual_datasets(tmp_path):
    """Non-multilingual datasets never get their hint blocks stripped."""
    dataset = tmp_path / "swebenchpro@1.0"
    _make_task(
        dataset,
        "django__django-99999",
        "Problem body.\n\n## Hints\n\nsome body.",
    )
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "## Hints" in seed.prompt
    assert "some body" in seed.prompt


@pytest.mark.asyncio
async def test_seed_records_hints_stripped_in_metadata(tmp_path):
    dataset = tmp_path / "swebench_multilingual@1.0"
    _make_task(
        dataset,
        "apache__druid-12345",
        "Problem body.\n\n## Hints\n\nsome body.",
    )
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert seed.metadata.get("hints_stripped") is True


@pytest.mark.asyncio
async def test_seed_omits_hints_stripped_when_noop(tmp_path):
    dataset = tmp_path / "swebench_multilingual@1.0"
    _make_task(dataset, "apache__druid-noop", "No hints here at all.")
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "hints_stripped" not in seed.metadata


@pytest.mark.asyncio
async def test_seed_handles_adapter_canary_empty_hints_bytes(tmp_path):
    """Byte-exact regression guard for an adapter-written empty-hints tail."""
    dataset = tmp_path / "swebench_multilingual@1.0"
    instruction_bytes = "Problem body text.\n\n## Hints\n\n\n"
    _make_task(dataset, "apache__druid-canary", instruction_bytes)
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "## Hints" not in seed.prompt
    assert seed.prompt == "Problem body text."
    assert seed.metadata.get("hints_stripped") is True


@pytest.mark.asyncio
async def test_seed_handles_adapter_normal_hints_bytes(tmp_path):
    """Byte-exact regression guard for a normal (non-canary) adapter output."""
    dataset = tmp_path / "swebench_multilingual@1.0"
    instruction_bytes = (
        "Problem body text.\n"
        "Second line of body.\n\n## Hints\n\n"
        "@maintainer: this looks like bar.py line 42.\n"
        "reply: I'll patch it tomorrow.\n"
    )
    _make_task(dataset, "apache__druid-hints", instruction_bytes)
    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)
    assert "## Hints" not in seed.prompt
    assert "maintainer" not in seed.prompt.lower()
    assert "bar.py" not in seed.prompt
    assert seed.prompt == "Problem body text.\nSecond line of body."
    assert seed.metadata.get("hints_stripped") is True
