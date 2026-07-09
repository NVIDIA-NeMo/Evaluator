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
"""Behavioural tests for the git future-history scrub command builder.

The builder returns a POSIX shell snippet run in the agent sandbox before the
agent starts. These tests execute that snippet against real throwaway git
repositories to prove the anti-cheat invariant: after the scrub, no commit that
is not an ancestor of the base commit is reachable, and the future objects are
physically gone. Skip cases (no repo / no HEAD) must exit 0.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nemo_evaluator.sandbox.git_scrub import build_git_history_scrub_cmd


def _git(cwd: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(cwd),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_scrub(wd: Path):
    return subprocess.run(
        ["/bin/sh", "-c", build_git_history_scrub_cmd(str(wd))],
        capture_output=True,
        text=True,
    )


def _make_repo_with_future(tmp_path: Path) -> tuple[Path, str, str]:
    """Repo whose HEAD is left at the *base* commit while a 'gold' fix commit
    plus a hidden-tests tag live in the future history (reachable via
    other refs), mimicking a SWE-bench task image."""
    wd = tmp_path / "repo"
    wd.mkdir()
    _git(wd, "init", "-q")
    (wd / "a.txt").write_text("base\n")
    _git(wd, "add", "-A")
    _git(wd, "commit", "-qm", "base")
    base = _git(wd, "rev-parse", "HEAD")
    (wd / "a.txt").write_text("fixed\n")
    _git(wd, "add", "-A")
    _git(wd, "commit", "-qm", "gold fix + hidden tests")
    gold = _git(wd, "rev-parse", "HEAD")
    _git(wd, "tag", "gold-fix")
    _git(wd, "branch", "future")
    # Leave the working tree checked out at base, as the task image would.
    _git(wd, "checkout", "-q", base)
    return wd, base, gold


@pytest.mark.skipif(
    subprocess.run(["which", "git"], capture_output=True).returncode != 0,
    reason="git not available",
)
class TestGitHistoryScrub:
    def test_future_commit_becomes_unreachable_and_deleted(self, tmp_path: Path):
        wd, base, gold = _make_repo_with_future(tmp_path)

        result = _run_scrub(wd)

        assert result.returncode == 0, result.stderr
        assert "NEL_GIT_CLEANUP" in result.stdout
        assert "non_ancestor_left=0" in result.stdout
        # git log --all must no longer reveal the gold fix.
        log_all = _git(wd, "log", "--all", "--oneline")
        assert gold[:8] not in log_all
        assert "gold fix" not in log_all
        # git show <gold_sha> must fail — object physically pruned.
        shown = subprocess.run(
            ["git", "-C", str(wd), "cat-file", "-e", gold],
            capture_output=True,
        )
        assert shown.returncode != 0
        # base history is intact.
        assert _git(wd, "rev-parse", "HEAD") == base

    def test_invariant_enforced_via_exit_code(self, tmp_path: Path):
        """The snippet self-enforces: a generic runner that only checks the
        exit code must see success (0) once the scrub completes cleanly."""
        wd, _base, _gold = _make_repo_with_future(tmp_path)
        assert _run_scrub(wd).returncode == 0

    def test_workdir_with_space_does_not_fail_open(self, tmp_path: Path):
        """A workdir path containing a space must still scrub — cd/safe.directory
        are quoted, so it must not silently hit the no_workdir skip (fail-open)."""
        spaced = tmp_path / "dir with space"
        spaced.mkdir()
        wd, _base, gold = _make_repo_with_future(spaced)
        assert " " in str(wd)

        result = _run_scrub(wd)

        assert result.returncode == 0, result.stderr
        assert "skipped" not in result.stdout  # did NOT fail open to no_workdir
        assert "non_ancestor_left=0" in result.stdout
        shown = subprocess.run(["git", "-C", str(wd), "cat-file", "-e", gold], capture_output=True)
        assert shown.returncode != 0

    def test_skips_non_git_dir_without_error(self, tmp_path: Path):
        wd = tmp_path / "plain"
        wd.mkdir()
        result = _run_scrub(wd)
        assert result.returncode == 0
        assert "skipped" in result.stdout
        assert "not_a_git_repo" in result.stdout

    def test_skips_missing_workdir_without_error(self, tmp_path: Path):
        result = _run_scrub(tmp_path / "does-not-exist")
        assert result.returncode == 0
        assert "skipped" in result.stdout
        assert "no_workdir" in result.stdout
