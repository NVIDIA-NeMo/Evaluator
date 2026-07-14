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
    record_output_dir_override,
    write_config_snapshot,
)


class _StubConfig:
    """Minimal stand-in for EvalConfig: just the snapshot private attrs."""

    def __init__(self, raw=None, provenance=None):
        self._composed_raw = raw
        self._snapshot_provenance = provenance or {}


class TestWriteConfigSnapshot:
    def test_snapshot_content(self, tmp_path):
        """Header carries provenance; env refs survive; body round-trips."""
        raw = {"services": {"m": {"type": "api", "url": "${U}/v1", "api_key": "${KEY}"}}}
        prov = build_provenance(source_config="/tmp/c.yaml", run_id="r1")
        path = write_config_snapshot(_StubConfig(raw=raw, provenance=prov), tmp_path)
        assert path == tmp_path / SNAPSHOT_FILENAME
        text = path.read_text()
        assert text.startswith("#")
        assert "nemo-evaluator version" in text
        assert "source config: /tmp/c.yaml" in text
        assert "run_id: r1" in text
        assert "${KEY}" in text
        # Comments are ignored by YAML: the body must round-trip.
        assert yaml.safe_load(text) == raw

    def test_skipped_in_inner_execution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEL_INNER_EXECUTION", "1")
        assert write_config_snapshot(_StubConfig(raw={"a": 1}), tmp_path) is None
        assert not (tmp_path / SNAPSHOT_FILENAME).exists()

    def test_never_raises(self):
        """A broken config must not raise (snapshot is best-effort)."""

        class _Broken:
            _composed_raw = {"a": 1}
            output = None  # config.output.dir access will fail

        assert write_config_snapshot(_Broken(), None) is None

    def test_force_semantics(self, tmp_path):
        """Fresh runs overwrite a stale snapshot; resumes keep the original."""
        write_config_snapshot(_StubConfig(raw={"a": 1}), tmp_path, force=True)
        write_config_snapshot(_StubConfig(raw={"a": 2}), tmp_path, force=False)  # resume: kept
        path = tmp_path / SNAPSHOT_FILENAME
        assert "a: 1" in path.read_text()
        write_config_snapshot(_StubConfig(raw={"a": 3}), tmp_path, force=True)  # fresh run: overwritten
        assert "a: 3" in path.read_text()

    def test_cli_output_dir_override_reflected(self, tmp_path):
        """-o overrides config.output.dir after compose; the snapshot must match."""
        cfg = _StubConfig(raw={"output": {"dir": "./from_yaml"}})
        record_output_dir_override(cfg, "/actual/run/dir")
        text = write_config_snapshot(cfg, tmp_path).read_text()
        assert "/actual/run/dir" in text
        assert "from_yaml" not in text
