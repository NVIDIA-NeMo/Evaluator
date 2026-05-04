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
"""Tests for nemo_evaluator.executors.local_executor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from nemo_evaluator.executors.local_executor import (
    LocalExecutor,
    _pid_alive,
    _read_pid,
    _write_pid,
)


class TestPidHelpers:
    def test_write_and_read(self, tmp_path):
        _write_pid(tmp_path, 12345)
        assert _read_pid(tmp_path) == 12345

    def test_read_missing(self, tmp_path):
        assert _read_pid(tmp_path) is None

    def test_read_corrupt(self, tmp_path):
        (tmp_path / "nel.pid").write_text("not-a-number")
        assert _read_pid(tmp_path) is None

    def test_write_default_pid(self, tmp_path):
        _write_pid(tmp_path)
        pid = _read_pid(tmp_path)
        assert pid is not None and pid > 0

    @patch("nemo_evaluator.executors.local_executor.os.kill")
    def test_pid_alive_true(self, mock_kill):
        assert _pid_alive(1234) is True
        mock_kill.assert_called_once_with(1234, 0)

    @patch("nemo_evaluator.executors.local_executor.os.kill", side_effect=OSError)
    def test_pid_alive_false(self, mock_kill):
        assert _pid_alive(99999) is False


class TestLocalExecutor:
    def test_name(self):
        assert LocalExecutor.name == "local"

    def test_detect_true(self, tmp_path):
        (tmp_path / "nel.pid").write_text("1")
        assert LocalExecutor.detect(tmp_path) is True

    def test_detect_false(self, tmp_path):
        assert LocalExecutor.detect(tmp_path) is False

    def test_status_no_pid(self, tmp_path):
        ex = LocalExecutor()
        state = ex.status(tmp_path)
        assert state.running is False

    def test_status_with_pid(self, tmp_path):
        _write_pid(tmp_path, 99999)
        ex = LocalExecutor()
        with patch("nemo_evaluator.executors.local_executor._pid_alive", return_value=True):
            state = ex.status(tmp_path)
        assert state.running is True

    def test_stop_no_pid(self, tmp_path):
        ex = LocalExecutor()
        assert ex.stop(tmp_path) is False

    def test_dry_run(self, tmp_path):
        ex = LocalExecutor()
        config = MagicMock()
        config.benchmarks = [MagicMock(name="bench1", repeats=1)]
        config.output.dir = str(tmp_path)
        ex.run(config, dry_run=True)
