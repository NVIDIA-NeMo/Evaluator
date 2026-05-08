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
"""Tests for orchestration/artifact_access.py."""

from __future__ import annotations

import logging
from unittest.mock import patch

from nemo_evaluator.orchestration.artifact_access import (
    apply_artifact_access,
    chmod_a_rx,
    warn_if_parent_chain_lacks_execute,
)


def _mode(path) -> int:
    return path.stat().st_mode & 0o777


class TestArtifactAccess:
    def test_only_targets_current_run_artifacts(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()
        root.chmod(0o700)
        stale = root / "stale-private.txt"
        stale.write_text("do not expose")
        stale.chmod(0o600)
        bench_dir = root / "bench"
        bench_dir.mkdir()
        bench_dir.chmod(0o700)
        results = bench_dir / "results.jsonl"
        results.write_text("{}\n")
        results.chmod(0o600)

        apply_artifact_access(root_dir=root, artifact_dirs={bench_dir}, artifact_files={results})

        assert _mode(root) == 0o755
        assert _mode(bench_dir) == 0o755
        assert _mode(results) == 0o755
        assert _mode(stale) == 0o600

    def test_apply_checks_parent_chain(self, tmp_path):
        root = tmp_path / "run"
        root.mkdir()

        with patch("nemo_evaluator.orchestration.artifact_access.warn_if_parent_chain_lacks_execute") as warn:
            apply_artifact_access(root_dir=root, artifact_dirs=set(), artifact_files=set())

        warn.assert_called_once_with(root)

    def test_missing_path_is_ignored(self, tmp_path):
        chmod_a_rx(tmp_path / "missing")

    def test_chmod_failure_logs_warning(self, tmp_path, caplog):
        path = tmp_path / "artifact"
        path.write_text("data")

        with patch.object(type(path), "chmod", side_effect=OSError("denied")):
            chmod_a_rx(path)

        assert "Failed to apply artifact access" in caplog.text
        assert str(path) in caplog.text

    def test_parent_chain_without_execute_warns(self, tmp_path, caplog):
        parent = tmp_path / "private-parent"
        root = parent / "run"
        root.mkdir(parents=True)
        parent.chmod(0o700)
        caplog.set_level(logging.WARNING)

        try:
            warn_if_parent_chain_lacks_execute(root, stop_at=parent)
        finally:
            parent.chmod(0o755)

        assert "Artifact access may be blocked" in caplog.text
        assert str(parent) in caplog.text
        assert "lack a+x" in caplog.text

    def test_parent_chain_execute_without_read_is_enough(self, tmp_path, caplog):
        parent = tmp_path / "traversable-parent"
        root = parent / "run"
        root.mkdir(parents=True)
        parent.chmod(0o711)
        caplog.set_level(logging.WARNING)

        try:
            warn_if_parent_chain_lacks_execute(root, stop_at=parent)
        finally:
            parent.chmod(0o755)

        assert "Artifact access may be blocked" not in caplog.text
