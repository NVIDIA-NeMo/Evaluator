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
"""Git future-history scrub — anti-cheat pre-agent command for SWE-style evals.

SWE-bench task images ship the repository with the *solution* present in the
git history: the gold fix and hidden tests sit in commits that are descendants
of the base commit. An agent can read them with ``git log --all`` /
``git show <fix_sha>`` and copy the solution instead of solving the issue.

:func:`build_git_history_scrub_cmd` returns a POSIX shell snippet, run once in
the agent sandbox before the agent starts, that makes every commit which is not
an ancestor of the base commit unreachable *and* physically deletes the objects
(refs/tags/remotes/reflogs/packed-refs removed + ``git gc --prune=now``).

The working tree is deliberately left untouched — SWE images carry env-patch
edits and pre-built artifacts in the workdir, and verification runs in a fresh
container with the agent's diff re-applied against the recorded base commit, so
scrubbing the agent container's history cannot affect scoring.

The snippet is self-enforcing: it emits a single ``NEL_GIT_CLEANUP`` audit line
and exits non-zero if any non-ancestor commit survived, so a caller that only
checks the return code still catches an incomplete scrub. Skip conditions
(missing workdir, not a git repo, no HEAD) exit 0 — they are expected for
non-git tasks and must not fail the rollout.
"""

from __future__ import annotations


def build_git_history_scrub_cmd(workdir: str) -> str:
    """Return the git-history scrub shell command for *workdir*.

    ``workdir`` is the repository root inside the agent sandbox (the sandbox
    spec's ``workdir``); it is trusted config, not model input.
    """
    return f"""
_T0=$(date +%s 2>/dev/null || echo 0)
if ! cd "{workdir}" 2>/dev/null; then echo "NEL_GIT_CLEANUP skipped reason=no_workdir wd={workdir}"; exit 0; fi
git config --global --add safe.directory "{workdir}" 2>/dev/null
if ! git rev-parse --git-dir >/dev/null 2>&1; then echo "NEL_GIT_CLEANUP skipped reason=not_a_git_repo wd={workdir}"; exit 0; fi
_BASE=$(git rev-parse HEAD 2>/dev/null)
if [ -z "$_BASE" ]; then echo "NEL_GIT_CLEANUP skipped reason=no_head wd={workdir}"; exit 0; fi
_BEFORE=$(git rev-list --all --count 2>/dev/null || echo '?')
mkdir -p .git/refs/heads
echo "$_BASE" > .git/refs/heads/_nel_work
git symbolic-ref HEAD refs/heads/_nel_work 2>/dev/null
rm -f .git/packed-refs .git/ORIG_HEAD .git/FETCH_HEAD .git/MERGE_HEAD
find .git/refs -type f ! -path '*/heads/_nel_work' -delete 2>/dev/null
rm -rf .git/refs/tags .git/refs/remotes .git/logs
for _r in $(git remote 2>/dev/null); do git remote remove "$_r" 2>/dev/null; done
git reflog expire --expire=now --expire-unreachable=now --all 2>/dev/null
git gc --prune=now --quiet 2>/dev/null
_GC_RC=$?
_AFTER=$(git rev-list --all --count 2>/dev/null || echo '?')
_LEFT=$(git rev-list --all --not "$_BASE" --count 2>/dev/null || echo '?')
_ELAPSED=$(( $(date +%s 2>/dev/null || echo 0) - _T0 ))
echo "NEL_GIT_CLEANUP base=$_BASE commits_before=$_BEFORE commits_after=$_AFTER non_ancestor_left=$_LEFT gc_rc=$_GC_RC elapsed_s=$_ELAPSED"
[ "$_LEFT" = 0 ] || exit 3
""".strip()
