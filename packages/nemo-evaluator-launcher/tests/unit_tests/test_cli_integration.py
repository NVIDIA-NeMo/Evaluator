# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Integration tests for the complete CLI workflow with dummy backend executor."""

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest
from nemo_evaluator_launcher.cli.export import ExportCmd
from nemo_evaluator_launcher.cli.ls_tasks import Cmd as LsCmd
from nemo_evaluator_launcher.cli.run import Cmd as RunCmd
from nemo_evaluator_launcher.cli.status import Cmd as StatusCmd
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.executors.base import ExecutionState
from omegaconf import OmegaConf

from tests.unit_tests.conftest import DummyExecutor, extract_invocation_id


class TestCLIWorkflowIntegration:
    """Test complete CLI workflow integration."""

    def test_complete_workflow_run_status_ls(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test complete workflow: run -> status -> ls."""
        # Step 1: Run evaluation
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {"type": "dummy", "output_dir": "/tmp/test_output"},
            "target": {
                "api_endpoint": {"api_key_name": "test_key", "model_id": "test_model"}
            },
            "evaluation": [
                {"name": "test_task_1", "overrides": {"param1": "value1"}},
                {"name": "test_task_2", "overrides": {"param2": "value2"}},
            ],
        }

        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.return_value = OmegaConf.create(config_dict)

            # Run evaluation
            run_cmd = RunCmd(config_name="test_config")
            run_cmd.execute()

            # Get the invocation ID from the print output
            invocation_id = extract_invocation_id(mock_print)

            # Verify invocation ID format
            assert len(invocation_id) == 8
            assert all(c in "0123456789abcdef" for c in invocation_id)

            # Step 2: Check status
            # Set job statuses
            job_ids = [f"{invocation_id}.0", f"{invocation_id}.1"]
            DummyExecutor.set_job_status(job_ids[0], ExecutionState.RUNNING)
            DummyExecutor.set_job_status(job_ids[1], ExecutionState.SUCCESS)

            # Check status by invocation ID
            status_cmd = StatusCmd(job_ids=[invocation_id], json=True)

            with (
                patch("json.dumps") as mock_json_dumps,
                patch("builtins.print"),
            ):
                status_cmd.execute()

                # Verify status output
                mock_json_dumps.assert_called_once()
                status_data = mock_json_dumps.call_args[0][0]
                assert len(status_data) == 2

                # Check first job status
                assert status_data[0]["invocation"] == invocation_id
                assert status_data[0]["job_id"] == job_ids[0]
                assert status_data[0]["status"] == "running"

                # Check second job status
                assert status_data[1]["invocation"] == invocation_id
                assert status_data[1]["job_id"] == job_ids[1]
                assert status_data[1]["status"] == "success"

            # Step 3: List tasks
            with (
                patch(
                    "nemo_evaluator_launcher.api.functional.load_tasks_mapping"
                ) as mock_load_tasks,
                patch("builtins.print") as mock_ls_print,
            ):
                mock_load_tasks.return_value = {
                    ("lm-eval", "test_task_1"): {
                        "task": "test_task_1",
                        "endpoint_type": "openai",
                        "harness": "lm-eval",
                        "container": "test-container:latest",
                    },
                    ("helm", "test_task_2"): {
                        "task": "test_task_2",
                        "endpoint_type": "anthropic",
                        "harness": "helm",
                        "container": "test-container:latest",
                    },
                }
                ls_cmd = LsCmd()
                ls_cmd.execute()

                # Verify ls output (JSON printed includes both tasks)
                assert mock_ls_print.call_count == 1
                printed = mock_ls_print.call_args[0][0]
                assert "test_task_1" in printed
                assert "test_task_2" in printed

    def test_workflow_with_multiple_evaluations(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test workflow with multiple evaluation runs."""
        # Run first evaluation
        config1 = {
            "deployment": {"type": "none"},
            "execution": {"type": "dummy", "output_dir": "/tmp/test1"},
            "target": {"api_endpoint": {"api_key_name": "key1", "model_id": "model1"}},
            "evaluation": [{"name": "task1", "overrides": {}}],
        }

        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.return_value = OmegaConf.create(config1)

            run_cmd1 = RunCmd(config_name="config1")
            run_cmd1.execute()

            invocation_id1 = extract_invocation_id(mock_print)

            # Run second evaluation
            config2 = {
                "deployment": {"type": "none"},
                "execution": {"type": "dummy", "output_dir": "/tmp/test2"},
                "target": {
                    "api_endpoint": {"api_key_name": "key2", "model_id": "model2"}
                },
                "evaluation": [{"name": "task2", "overrides": {}}],
            }

            mock_compose.return_value = OmegaConf.create(config2)

            with patch("builtins.print") as mock_print2:
                run_cmd2 = RunCmd(config_name="config2")
                run_cmd2.execute()

                invocation_id2 = extract_invocation_id(mock_print2)

                # Verify different invocation IDs
                assert invocation_id1 != invocation_id2

                # Set different statuses
                DummyExecutor.set_job_status(
                    f"{invocation_id1}.0", ExecutionState.SUCCESS
                )
                DummyExecutor.set_job_status(
                    f"{invocation_id2}.0", ExecutionState.FAILED
                )

                # Check status for both invocations
                status_cmd = StatusCmd(
                    job_ids=[invocation_id1, invocation_id2], json=True
                )

                with (
                    patch("json.dumps") as mock_json_dumps,
                    patch("builtins.print"),
                ):
                    status_cmd.execute()

                    status_data = mock_json_dumps.call_args[0][0]
                    assert len(status_data) == 2

                    # Check first invocation status
                    assert status_data[0]["invocation"] == invocation_id1
                    assert status_data[0]["status"] == "success"

                    # Check second invocation status
                    assert status_data[1]["invocation"] == invocation_id2
                    assert status_data[1]["status"] == "failed"

    def test_workflow_with_job_id_status_queries(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test workflow with individual job ID status queries."""
        # Run evaluation
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {"type": "dummy", "output_dir": "/tmp/test"},
            "target": {"api_endpoint": {"api_key_name": "key", "model_id": "model"}},
            "evaluation": [
                {"name": "task1", "overrides": {}},
                {"name": "task2", "overrides": {}},
            ],
        }

        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.return_value = OmegaConf.create(config_dict)

            run_cmd = RunCmd()
            run_cmd.execute()

            invocation_id = extract_invocation_id(mock_print)
            job_ids = [f"{invocation_id}.0", f"{invocation_id}.1"]

            # Set different statuses
            DummyExecutor.set_job_status(job_ids[0], ExecutionState.PENDING)
            DummyExecutor.set_job_status(job_ids[1], ExecutionState.RUNNING)

            # Query status by individual job IDs
            status_cmd = StatusCmd(job_ids=job_ids, json=True)

            with (
                patch("json.dumps") as mock_json_dumps,
                patch("builtins.print"),
            ):
                status_cmd.execute()

                status_data = mock_json_dumps.call_args[0][0]
                assert len(status_data) == 2

                # Check individual job statuses
                assert status_data[0]["job_id"] == job_ids[0]
                assert status_data[0]["status"] == "pending"
                assert status_data[1]["job_id"] == job_ids[1]
                assert status_data[1]["status"] == "running"

    def test_workflow_with_mixed_status_queries(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test workflow with mixed invocation ID and job ID queries."""
        # Run evaluation
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {"type": "dummy", "output_dir": "/tmp/test"},
            "target": {"api_endpoint": {"api_key_name": "key", "model_id": "model"}},
            "evaluation": [
                {"name": "task1", "overrides": {}},
                {"name": "task2", "overrides": {}},
            ],
        }

        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.return_value = OmegaConf.create(config_dict)

            run_cmd = RunCmd()
            run_cmd.execute()

            invocation_id = extract_invocation_id(mock_print)
            job_ids = [f"{invocation_id}.0", f"{invocation_id}.1"]

            # Set statuses
            DummyExecutor.set_job_status(job_ids[0], ExecutionState.SUCCESS)
            DummyExecutor.set_job_status(job_ids[1], ExecutionState.FAILED)

            # Query with mixed IDs: invocation ID and individual job ID
            status_cmd = StatusCmd(job_ids=[invocation_id, job_ids[0]], json=True)

            with (
                patch("json.dumps") as mock_json_dumps,
                patch("builtins.print"),
            ):
                status_cmd.execute()

                status_data = mock_json_dumps.call_args[0][0]
                # We expect 3 results: 2 from invocation ID query + 1 from job ID query
                # But the job ID query might return the same job twice (once from invocation, once from direct query)
                assert len(status_data) >= 2

                # Check that we get results for both query types
                invocation_results = [
                    r for r in status_data if r["invocation"] == invocation_id
                ]
                job_results = [r for r in status_data if r["job_id"] == job_ids[0]]

                assert len(invocation_results) >= 2
                assert len(job_results) >= 1

    def test_workflow_with_nonexistent_queries(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test workflow with queries for nonexistent jobs/invocations."""
        # Run evaluation
        config_dict = {
            "deployment": {"type": "none"},
            "execution": {"type": "dummy", "output_dir": "/tmp/test"},
            "target": {"api_endpoint": {"api_key_name": "key", "model_id": "model"}},
            "evaluation": [{"name": "task1", "overrides": {}}],
        }

        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.return_value = OmegaConf.create(config_dict)

            run_cmd = RunCmd()
            run_cmd.execute()

            _ = extract_invocation_id(mock_print)

            # Query nonexistent job and invocation
            status_cmd = StatusCmd(job_ids=["nonexistent_job", "12345678"])

            with patch("builtins.print") as mock_status_print:
                status_cmd.execute()

                # Verify print was called (table was output)
                assert mock_status_print.called

                # The actual status data is returned by get_status, let's check that directly
                from nemo_evaluator_launcher.api.functional import get_status

                status_data = get_status(["nonexistent_job", "12345678"])
                assert len(status_data) == 2

                # Check nonexistent job result
                assert status_data[0]["job_id"] == "nonexistent_job"
                assert status_data[0]["status"] == "not_found"
                assert status_data[0]["invocation"] is None

                # Check nonexistent invocation result
                assert status_data[1]["invocation"] == "12345678"
                assert status_data[1]["status"] == "not_found"
                assert status_data[1]["job_id"] is None

    def test_workflow_with_error_handling(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test workflow with error handling scenarios."""
        # Test run command with invalid config
        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.side_effect = Exception("Config error")

            run_cmd = RunCmd()

            with pytest.raises(Exception, match="Config error"):
                run_cmd.execute()

        # Test status command with executor error
        db = ExecutionDB()
        db.write_job(
            JobData(
                invocation_id="test123",
                job_id="test123.0",
                timestamp=1234567890,
                executor="dummy",
                data={"task_name": "test_task"},
            )
        )

        # Mock executor to raise error
        with patch(
            "nemo_evaluator_launcher.api.functional.get_executor"
        ) as mock_get_executor:
            mock_get_executor.side_effect = ValueError("Executor not found")

            status_cmd = StatusCmd(job_ids=["test123.0"])

            with patch("builtins.print") as mock_status_print:
                status_cmd.execute()

                # Verify print was called (table was output)
                assert mock_status_print.called

                # The actual status data is returned by get_status, let's check that directly
                from nemo_evaluator_launcher.api.functional import get_status

                status_data = get_status(["test123.0"])
                assert len(status_data) == 1
                assert status_data[0]["status"] == "error"

    def test_workflow_with_realistic_data(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test workflow with realistic data structures."""
        # Realistic config
        realistic_config = {
            "deployment": {"type": "none"},
            "execution": {"type": "dummy", "output_dir": "/tmp/realistic_output"},
            "target": {
                "api_endpoint": {"api_key_name": "OPENAI_API_KEY", "model_id": "gpt-4"}
            },
            "evaluation": [
                {
                    "name": "mmlu",
                    "overrides": {"model_args": "temperature=0.1", "num_fewshot": 5},
                },
                {
                    "name": "arc",
                    "overrides": {"model_args": "temperature=0.0", "num_fewshot": 0},
                },
            ],
        }

        with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
            mock_compose.return_value = OmegaConf.create(realistic_config)

            # Run evaluation
            run_cmd = RunCmd(
                config_name="realistic",
                override=["target.api_endpoint.model_id=gpt-4-turbo"],
            )
            run_cmd.execute()

            invocation_id = extract_invocation_id(mock_print)

            # Set realistic statuses
            job_ids = [f"{invocation_id}.0", f"{invocation_id}.1"]
            DummyExecutor.set_job_status(job_ids[0], ExecutionState.SUCCESS)
            DummyExecutor.set_job_status(job_ids[1], ExecutionState.RUNNING)

            # Check status
            status_cmd = StatusCmd(job_ids=[invocation_id], json=True)

            with (
                patch("json.dumps") as mock_json_dumps,
                patch("builtins.print"),
            ):
                status_cmd.execute()

                status_data = status_data = mock_json_dumps.call_args[0][0]
                assert len(status_data) == 2

                # Verify realistic data
                for result in status_data:
                    assert result["invocation"] == invocation_id
                    assert result["job_id"] in job_ids
                    assert result["status"] in ["success", "running"]
                    assert "task_name" in result["data"]
                    assert result["data"]["task_name"] in ["mmlu", "arc"]


class TestCLICommandLineInterface:
    """Test CLI command line interface directly."""

    def test_cli_run_command_line(
        self, mock_execdb, mock_api_endpoint_check, mock_print
    ):
        """Test CLI run command from command line."""
        from nemo_evaluator_launcher.cli.main import main

        with patch(
            "nemo_evaluator_launcher.cli.main.create_parser"
        ) as mock_create_parser:
            # Setup mock parser and arguments
            mock_parser = type("MockParser", (), {"parse_args": lambda: None})()
            mock_args = type(
                "Args",
                (),
                {
                    "command": "run",
                    "run": type(
                        "RunCmd",
                        (),
                        {
                            "config_name": "test_config",
                            "override": ["param=value"],
                            "execute": lambda self: None,
                        },
                    )(),
                },
            )()
            mock_parser.parse_args = lambda: mock_args
            mock_create_parser.return_value = mock_parser

            # Execute main
            main()

            # Verify create_parser was called
            mock_create_parser.assert_called_once()

    def test_cli_status_command_line(self, mock_execdb, mock_json_dumps, mock_print):
        """Test CLI status command from command line."""
        from nemo_evaluator_launcher.cli.main import main

        with patch(
            "nemo_evaluator_launcher.cli.main.create_parser"
        ) as mock_create_parser:
            # Setup mock parser and arguments
            mock_parser = type("MockParser", (), {"parse_args": lambda: None})()
            mock_args = type(
                "Args",
                (),
                {
                    "command": "status",
                    "status": type(
                        "StatusCmd",
                        (),
                        {"job_ids": ["abc123", "def456"], "execute": lambda self: None},
                    )(),
                },
            )()
            mock_parser.parse_args = lambda: mock_args
            mock_create_parser.return_value = mock_parser

            # Execute main
            main()

            # Verify create_parser was called
            mock_create_parser.assert_called_once()

    def test_cli_ls_tasks_command_line(self, mock_tasks_mapping_load, mock_print):
        """Test CLI ls command from command line."""
        from nemo_evaluator_launcher.cli.main import main

        with patch(
            "nemo_evaluator_launcher.cli.main.create_parser"
        ) as mock_create_parser:
            # Setup mock parser and arguments
            mock_parser = type("MockParser", (), {"parse_args": lambda: None})()
            mock_args = type(
                "Args",
                (),
                {
                    "command": "ls",
                    "tasks": type("LsTasksCmd", (), {"execute": lambda self: None})(),
                },
            )()
            mock_parser.parse_args = lambda: mock_args
            mock_create_parser.return_value = mock_parser

            # Execute main
            main()

            # Verify create_parser was called
            mock_create_parser.assert_called_once()


class TestCLIErrorScenarios:
    """Test CLI error scenarios and edge cases."""

    def test_cli_with_missing_required_argument(self):
        """Test CLI with missing required argument."""
        from nemo_evaluator_launcher.cli.main import main

        with patch(
            "nemo_evaluator_launcher.cli.main.create_parser"
        ) as mock_create_parser:
            mock_parser = type("MockParser", (), {"parse_args": lambda: None})()
            mock_parser.parse_args = lambda: (_ for _ in ()).throw(SystemExit(2))
            mock_create_parser.return_value = mock_parser

            with pytest.raises(SystemExit):
                main()

    def test_cli_with_invalid_command(self):
        """Test CLI with invalid command."""
        from nemo_evaluator_launcher.cli.main import main

        with patch(
            "nemo_evaluator_launcher.cli.main.create_parser"
        ) as mock_create_parser:
            # Setup mock with invalid command
            mock_parser = type("MockParser", (), {"parse_args": lambda: None})()
            mock_args = type("Args", (), {"command": "invalid"})()
            mock_parser.parse_args = lambda: mock_args
            mock_create_parser.return_value = mock_parser

            # Should not raise exception but also not execute any command
            main()

    def test_cli_with_execution_failure(
        self, mock_hydra_config, mock_api_endpoint_check
    ):
        """Test CLI with execution failure."""
        from nemo_evaluator_launcher.cli.main import main

        with patch(
            "nemo_evaluator_launcher.cli.main.create_parser"
        ) as mock_create_parser:
            # Setup mock with failing execute method
            mock_parser = type("MockParser", (), {"parse_args": lambda: None})()
            mock_run_cmd = type(
                "RunCmd",
                (),
                {
                    "execute": lambda self: (_ for _ in ()).throw(
                        Exception("Execution failed")
                    )
                },
            )()

            mock_args = type("Args", (), {"command": "run", "run": mock_run_cmd})()
            mock_parser.parse_args = lambda: mock_args
            mock_create_parser.return_value = mock_parser

            with pytest.raises(Exception, match="Execution failed"):
                main()


def test_cli_export_missing_metadata(monkeypatch, tmp_path):
    # Simulate exporter returning per-job dict without "metadata"
    def fake_export_results(ids, dest, config):
        return {
            "success": True,
            "invocation_id": ids[0],
            "jobs": {
                f"{ids[0]}.0": {"success": True, "message": "Updated Output"},
            },
        }

    monkeypatch.setattr(
        "nemo_evaluator_launcher.cli.export.export_results", fake_export_results
    )

    cmd = ExportCmd(
        invocation_ids=["2eaceed9"],
        dest="local",
        format="json",
        output_dir=str(tmp_path),
    )

    buf = StringIO()
    with redirect_stdout(buf):
        cmd.execute()

    out = buf.getvalue()
    assert "Export completed for 2eaceed9" in out
    assert "2eaceed9.0: Updated Output" in out
