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

    def test_eval_status_help_matches_watch_interval(self, runner):
        result = runner.invoke(cli, ["eval", "status", "--help"])
        assert result.exit_code == 0
        assert "Refresh every 60s" in result.output

    def test_list_help(self, runner):
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0


class TestEvalRun:
    def test_no_config_no_bench_errors(self, runner):
        result = runner.invoke(cli, ["eval", "run"])
        assert result.exit_code != 0

    @patch("nemo_evaluator.executors.get_executor")
    def test_quick_mode_reads_nvidia_api_key(self, mock_get_exec, runner):
        mock_executor = MagicMock()
        mock_get_exec.return_value = mock_executor

        result = runner.invoke(
            cli,
            [
                "eval",
                "run",
                "--bench",
                "gsm8k",
                "--model-url",
                "https://integrate.api.nvidia.com/v1",
                "--model-id",
                "nvidia/nemotron-3-super-120b-a12b",
                "--dry-run",
            ],
            env={"NVIDIA_API_KEY": "nvapi-test-key"},
        )

        assert result.exit_code == 0
        config = mock_executor.run.call_args.args[0]
        assert config.services["model"].api_key == "nvapi-test-key"

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


class TestEvalLogs:
    def test_follow_local_slurm_uses_tail_without_ssh(self, runner):
        from nemo_evaluator.run_store import RunMeta

        run_meta = RunMeta(
            run_id="local-slurm-run",
            executor="slurm",
            output_dir="/shared/eval/run",
            started_at="2026-07-21T00:00:00+00:00",
            details={
                "job_id": "12345",
                "hostname": "",
                "remote_dir": "/shared/eval/run",
            },
        )

        with (
            patch("nemo_evaluator.cli.eval._resolve_run_or_fail", return_value=run_meta),
            patch("nemo_evaluator.cli.eval._get_executor_for_run", return_value=MagicMock()),
            patch(
                "nemo_evaluator.executors.slurm_executor._resolve_latest_job_id_from_meta",
                return_value="12345",
            ),
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(
                cli,
                ["eval", "logs", "--run-id", run_meta.run_id, "--follow", "--tail", "25"],
            )

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once_with(
            ["tail", "-n", "25", "-f", "/shared/eval/run/logs/slurm-12345.log"],
            check=False,
        )
        assert "ssh" not in mock_run.call_args.args[0]

    def test_follow_single_shard_local_slurm_uses_shard_zero_log(self, runner):
        from nemo_evaluator.run_store import RunMeta

        run_meta = RunMeta(
            run_id="single-shard-local-slurm-run",
            executor="slurm",
            output_dir="/shared/eval/run",
            started_at="2026-07-21T00:00:00+00:00",
            details={
                "job_id": "12345",
                "job_ids": ["12345"],
                "hostname": "",
                "remote_dir": "/shared/eval/run",
                "is_sharded": True,
            },
        )

        with (
            patch("nemo_evaluator.cli.eval._resolve_run_or_fail", return_value=run_meta),
            patch("nemo_evaluator.cli.eval._get_executor_for_run", return_value=MagicMock()),
            patch(
                "nemo_evaluator.executors.slurm_executor._resolve_latest_job_id",
                return_value="12345",
            ) as mock_resolve_latest,
            patch("subprocess.run") as mock_run,
        ):
            result = runner.invoke(
                cli,
                ["eval", "logs", "--run-id", run_meta.run_id, "--follow", "--shard", "0"],
            )

        assert result.exit_code == 0, result.output
        mock_resolve_latest.assert_called_once_with("", "/shared/eval/run/shard_0", "12345", None)
        mock_run.assert_called_once_with(
            ["tail", "-n", "50", "-f", "/shared/eval/run/shard_0/logs/slurm-12345.log"],
            check=False,
        )


class TestValidateCommand:
    def test_validate_missing_file(self, runner):
        result = runner.invoke(cli, ["eval", "validate", "/nonexistent_xyz.yaml"])
        assert result.exit_code != 0

    @patch("nemo_evaluator.environments.registry.get_environment")
    @patch("nemo_evaluator.engine.eval_loop.run_evaluation", side_effect=PermissionError("dataset cache is read-only"))
    def test_validate_reports_runtime_failures_without_traceback(self, _mock_run, mock_get_env, runner):
        mock_get_env.return_value = MagicMock()

        result = runner.invoke(
            cli,
            [
                "validate",
                "-b",
                "gsm8k",
                "--model-url",
                "https://example.test/v1",
                "--model-id",
                "test-model",
            ],
        )

        assert result.exit_code != 0
        assert "Validation failed for 'gsm8k'" in result.output
        assert "dataset cache is read-only" in result.output

    def test_validate_reads_nvidia_api_key(self, runner: CliRunner) -> None:
        async def _run_evaluation(*_args: object, **_kwargs: object) -> dict[str, list[object]]:
            return {"_results": []}

        with (
            patch("nemo_evaluator.environments.registry.get_environment", return_value=MagicMock()),
            patch("nemo_evaluator.engine.eval_loop.run_evaluation", side_effect=_run_evaluation),
            patch("nemo_evaluator.engine.model_client.ModelClient") as mock_model_client,
        ):
            result = runner.invoke(
                cli,
                [
                    "validate",
                    "-b",
                    "gsm8k",
                    "--model-url",
                    "https://example.test/v1",
                    "--model-id",
                    "test-model",
                ],
                env={"NVIDIA_API_KEY": "nvapi-validate-key"},
            )

        assert result.exit_code == 0, result.output
        mock_model_client.assert_called_once_with(
            base_url="https://example.test/v1",
            model="test-model",
            api_key="nvapi-validate-key",
        )


class TestListCommand:
    def test_list_benchmarks(self, runner):
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0

    @patch("nemo_evaluator.environments.skills.list_skills_benchmarks", side_effect=ImportError("missing skills"))
    def test_list_skills_handles_missing_optional_dependency(self, _mock_list_skills, runner):
        result = runner.invoke(cli, ["list", "--source", "skills"])

        assert result.exit_code == 0
        assert "nemo-skills not installed" in result.output
