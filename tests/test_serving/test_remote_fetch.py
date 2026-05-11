# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_fetch_remote_results`` — lightweight-by-default fetch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_run_meta(tmp_path):
    local = tmp_path / "remote-out"
    local.mkdir()
    return SimpleNamespace(
        output_dir=str(local),
        details={
            "hostname": "cluster.example.com",
            "username": "tester",
            "remote_dir": "/lustre/runs/abc123",
        },
    )


class TestFetchRemoteResultsLightweight:
    def _invoke(self, include_results: bool, fake_run_meta):
        from nemo_evaluator.cli import eval as eval_cli

        captured_cmds: list[str] = []
        scp_calls: list[tuple] = []

        def fake_ssh_run(hostname, cmd, username=None, timeout=None):
            captured_cmds.append(cmd)
            files = [
                "/lustre/runs/abc123/gsm8k/eval-gsm8k-001.json",
                "/lustre/runs/abc123/gsm8k/results.jsonl",
            ]
            wants_results = "results.jsonl" in cmd
            kept = [f for f in files if f.endswith(".json") or (wants_results and f.endswith("results.jsonl"))]
            return "\n".join(kept) + "\n"

        with (
            patch.object(eval_cli, "_resolve_run_or_fail", return_value=fake_run_meta),
            patch("nemo_evaluator.executors.ssh.ssh_run", side_effect=fake_ssh_run),
            patch("nemo_evaluator.executors.ssh._ensure_master", return_value=None),
            patch(
                "nemo_evaluator.executors.ssh._ssh_target",
                return_value="tester@cluster.example.com",
            ),
            patch("nemo_evaluator.executors.ssh._ssh_opts", return_value=[]),
            patch(
                "nemo_evaluator.executors.ssh._run",
                side_effect=lambda cmd, **kw: scp_calls.append(tuple(cmd)),
            ),
        ):
            tmp, _ = eval_cli._fetch_remote_results("abc123", include_results=include_results)
        return captured_cmds, scp_calls, Path(tmp)

    def test_default_skips_results_jsonl(self, fake_run_meta):
        cmds, scp_calls, _ = self._invoke(include_results=False, fake_run_meta=fake_run_meta)
        assert len(cmds) == 1
        assert "eval-*.json" in cmds[0]
        assert "results.jsonl" not in cmds[0]
        scp_sources = [c[-2] for c in scp_calls]
        assert any("eval-gsm8k-001.json" in s for s in scp_sources)
        assert not any("results.jsonl" in s for s in scp_sources)

    def test_include_results_fetches_both(self, fake_run_meta):
        cmds, scp_calls, _ = self._invoke(include_results=True, fake_run_meta=fake_run_meta)
        assert "eval-*.json" in cmds[0]
        assert "results.jsonl" in cmds[0]
        scp_sources = [c[-2] for c in scp_calls]
        assert any("eval-gsm8k-001.json" in s for s in scp_sources)
        assert any("results.jsonl" in s for s in scp_sources)
