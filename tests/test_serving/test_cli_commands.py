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
"""Tests for CLI commands via Click CliRunner — no real execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from nemo_evaluator.cli.main import cli


@pytest.fixture()
def runner():
    return CliRunner()


class TestCLIStructure:
    def test_root_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "eval" in result.output

    def test_eval_help(self, runner):
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_eval_run_help(self, runner):
        result = runner.invoke(cli, ["eval", "run", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output

    def test_list_help(self, runner):
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0


class TestEvalRun:
    def test_no_config_no_bench_errors(self, runner):
        result = runner.invoke(cli, ["eval", "run"])
        assert result.exit_code != 0

    @patch("nemo_evaluator.cli.eval._load_config")
    @patch("nemo_evaluator.executors.get_executor")
    def test_dry_run(self, mock_get_exec, mock_load, runner, tmp_path):
        config = MagicMock()
        config.cluster.type = "local"
        config.output.dir = str(tmp_path)
        config.benchmarks = [MagicMock(name="bench1", repeats=1)]
        mock_load.return_value = config

        mock_executor = MagicMock()
        mock_get_exec.return_value = mock_executor

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("dummy: true")

        runner.invoke(cli, ["eval", "run", str(cfg_file), "--dry-run"])
        mock_executor.run.assert_called_once()
        call_kwargs = mock_executor.run.call_args
        assert call_kwargs.kwargs.get("dry_run") is True or call_kwargs[1].get("dry_run") is True


class TestValidateCommand:
    def test_validate_missing_file(self, runner):
        result = runner.invoke(cli, ["eval", "validate", "/nonexistent_xyz.yaml"])
        assert result.exit_code != 0


class TestListCommand:
    def test_list_benchmarks(self, runner):
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
