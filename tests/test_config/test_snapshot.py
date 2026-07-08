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
"""Tests for the composed-config reproducibility snapshot."""

import yaml

from nemo_evaluator.config.snapshot import (
    SNAPSHOT_FILENAME,
    build_provenance,
    build_snapshot_text,
    record_output_dir_override,
    write_config_snapshot,
)


class _StubConfig:
    """Minimal stand-in for EvalConfig: just the snapshot private attrs."""

    def __init__(self, raw=None, provenance=None):
        self._composed_raw = raw
        self._snapshot_provenance = provenance or {}


class TestSnapshotText:
    def test_header_and_reparseable_body(self):
        raw = {"services": {"m": {"type": "api", "url": "${U}/v1", "model": "x", "api_key": "${KEY}"}}}
        text = build_snapshot_text(raw, build_provenance(source_config="/tmp/c.yaml"))
        assert text.startswith("#")
        assert "nemo-evaluator version" in text
        assert "source config: /tmp/c.yaml" in text
        # Env refs are the safe representation and must survive verbatim.
        assert "${KEY}" in text
        # Comments are ignored by YAML: the body must round-trip.
        assert yaml.safe_load(text) == raw


class TestWriteConfigSnapshot:
    def test_writes_composed_raw(self, tmp_path):
        raw = {"services": {"m": {"type": "api", "api_key": "${KEY}"}}}
        cfg = _StubConfig(raw=raw, provenance={"run_id": "r1"})
        path = write_config_snapshot(cfg, tmp_path)
        assert path == tmp_path / SNAPSHOT_FILENAME
        text = path.read_text()
        assert "${KEY}" in text
        assert "run_id: r1" in text

    def test_skipped_in_inner_execution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEL_INNER_EXECUTION", "1")
        cfg = _StubConfig(raw={"a": 1})
        assert write_config_snapshot(cfg, tmp_path) is None
        assert not (tmp_path / SNAPSHOT_FILENAME).exists()

    def test_no_snapshot_without_composed_raw(self, tmp_path):
        """Quick mode / programmatic configs hold resolved secrets — no snapshot."""
        cfg = _StubConfig(raw=None)
        assert write_config_snapshot(cfg, tmp_path) is None
        assert not (tmp_path / SNAPSHOT_FILENAME).exists()

    def test_never_raises(self):
        """A broken config must not raise (snapshot is best-effort)."""

        class _Broken:
            _composed_raw = {"a": 1}
            output = None  # config.output.dir access will fail

        assert write_config_snapshot(_Broken(), None) is None


class TestForceOverwrite:
    def test_fresh_run_overwrites_stale_snapshot(self, tmp_path):
        write_config_snapshot(_StubConfig(raw={"a": 1}), tmp_path, force=True)
        path = write_config_snapshot(_StubConfig(raw={"a": 2}), tmp_path, force=True)
        assert "a: 2" in path.read_text()

    def test_resume_preserves_original(self, tmp_path):
        first = write_config_snapshot(_StubConfig(raw={"a": 1}), tmp_path, force=True)
        original = first.read_text()
        write_config_snapshot(_StubConfig(raw={"a": 2}), tmp_path, force=False)
        assert first.read_text() == original


class TestOutputDirOverride:
    def test_cli_override_reflected_in_snapshot(self, tmp_path):
        """-o overrides config.output.dir after compose; the snapshot must match."""
        cfg = _StubConfig(raw={"output": {"dir": "./from_yaml"}})
        record_output_dir_override(cfg, "/actual/run/dir")
        path = write_config_snapshot(cfg, tmp_path)
        text = path.read_text()
        assert "/actual/run/dir" in text
        assert "from_yaml" not in text
