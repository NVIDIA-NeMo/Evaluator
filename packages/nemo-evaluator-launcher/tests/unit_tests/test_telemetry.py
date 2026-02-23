# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the telemetry module."""

import os
import uuid
from unittest.mock import patch

import pytest


class TestTelemetryLevel:
    """Tests for get_telemetry_level function."""

    def test_default_level_is_full(self):
        """Test that default telemetry level is DEFAULT (2) when nothing is set."""
        from unittest.mock import MagicMock

        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
            with patch("nemo_evaluator.config.CONFIG_FILE", mock_path):
                assert get_telemetry_level() == TelemetryLevel.DEFAULT

    def test_level_0_disables_telemetry(self):
        """Test that level 0 disables telemetry."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "0"}):
            assert get_telemetry_level() == TelemetryLevel.OFF

    def test_level_1_is_minimal(self):
        """Test that level 1 enables minimal telemetry (no model_id)."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "1"}):
            assert get_telemetry_level() == TelemetryLevel.MINIMAL

    def test_level_2_is_full(self):
        """Test that level 2 enables full telemetry."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "2"}):
            assert get_telemetry_level() == TelemetryLevel.DEFAULT

    def test_invalid_value_falls_back_to_minimal(self):
        """Test that invalid values fall back to MINIMAL (1)."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import get_telemetry_level

        for invalid in ("true", "false", "abc", "3"):
            with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": invalid}):
                assert get_telemetry_level() == TelemetryLevel.MINIMAL


class TestSessionId:
    """Tests for session ID generation and retrieval."""

    def test_session_id_generation(self):
        """Test that generated session IDs are valid UUIDs."""
        from nemo_evaluator.telemetry import _generate_session_id

        session_id = _generate_session_id()
        # Should be a valid UUID
        uuid.UUID(session_id)  # Raises ValueError if invalid

    def test_session_id_uniqueness(self):
        """Test that each call generates a unique session ID."""
        from nemo_evaluator.telemetry import _generate_session_id

        ids = [_generate_session_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_get_session_id_from_env(self):
        """Test that session ID is read from environment variable."""
        from nemo_evaluator.telemetry import (
            TELEMETRY_SESSION_ID_ENV_VAR,
            get_session_id,
        )

        test_id = "test-session-123"
        with patch.dict(os.environ, {TELEMETRY_SESSION_ID_ENV_VAR: test_id}):
            assert get_session_id() == test_id

    def test_get_session_id_generates_new(self):
        """Test that a new session ID is generated when not in env."""
        from nemo_evaluator.telemetry import (
            TELEMETRY_SESSION_ID_ENV_VAR,
            get_session_id,
        )

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(TELEMETRY_SESSION_ID_ENV_VAR, None)
            session_id = get_session_id()
            # Should be a valid UUID
            uuid.UUID(session_id)


class TestEventSerialization:
    """Tests for event serialization."""

    def test_launcher_job_event_serialization(self):
        """Test that LauncherJobEvent serializes with correct camelCase aliases."""
        from nemo_evaluator.telemetry import StatusEnum

        from nemo_evaluator_launcher.telemetry import LauncherJobEvent

        event = LauncherJobEvent(
            executor_type="local",
            deployment_type="vllm",
            model="llama-3.1-8b",
            tasks=["mmlu", "ifeval"],
            exporters=["wandb"],
            status=StatusEnum.STARTED,
        )

        serialized = event.model_dump(by_alias=True)
        assert "executorType" in serialized
        assert "deploymentType" in serialized
        assert serialized["executorType"] == "local"
        assert serialized["deploymentType"] == "vllm"
        assert serialized["model"] == "llama-3.1-8b"
        assert serialized["tasks"] == ["mmlu", "ifeval"]
        assert serialized["exporters"] == ["wandb"]
        assert serialized["status"] == "started"


class TestTelemetryHandler:
    """Tests for TelemetryHandler."""

    def test_handler_initialization(self):
        """Test that TelemetryHandler initializes correctly."""
        from nemo_evaluator.telemetry import TelemetryHandler

        handler = TelemetryHandler(
            source_client_version="1.0.0",
            session_id="test-session",
        )
        assert handler._source_client_version == "1.0.0"
        assert handler._session_id == "test-session"
        assert handler._events == []
        assert handler._running is False

    def test_handler_enqueue_when_disabled(self):
        """Test that events are not enqueued when telemetry level is OFF."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import StatusEnum, TelemetryHandler

        from nemo_evaluator_launcher.telemetry import LauncherJobEvent

        handler = TelemetryHandler(telemetry_level=TelemetryLevel.OFF)
        event = LauncherJobEvent(
            executor_type="local",
            deployment_type="none",
            model="test",
            tasks=[],
            exporters=[],
            status=StatusEnum.STARTED,
        )
        handler.enqueue(event)
        assert len(handler._events) == 0

    def test_handler_enqueue_when_enabled(self):
        """Test that events are enqueued when telemetry level is DEFAULT."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import StatusEnum, TelemetryHandler

        from nemo_evaluator_launcher.telemetry import LauncherJobEvent

        handler = TelemetryHandler(telemetry_level=TelemetryLevel.DEFAULT)
        event = LauncherJobEvent(
            executor_type="local",
            deployment_type="none",
            model="test",
            tasks=[],
            exporters=[],
            status=StatusEnum.STARTED,
        )
        handler.enqueue(event)
        assert len(handler._events) == 1

    def test_handler_graceful_degradation(self):
        """Test that handler silently fails on invalid events."""
        from nemo_evaluator.config import TelemetryLevel
        from nemo_evaluator.telemetry import TelemetryHandler

        handler = TelemetryHandler(telemetry_level=TelemetryLevel.DEFAULT)
        handler.enqueue("not an event")  # type: ignore
        assert len(handler._events) == 0


class TestBuildPayload:
    """Tests for payload building."""

    def test_build_payload_structure(self):
        """Test that build_payload creates correct structure."""
        from datetime import datetime, timezone

        from nemo_evaluator.telemetry import QueuedEvent, StatusEnum, build_payload

        from nemo_evaluator_launcher.telemetry import LauncherJobEvent

        event = LauncherJobEvent(
            executor_type="local",
            deployment_type="vllm",
            model="test-model",
            tasks=["mmlu"],
            exporters=[],
            status=StatusEnum.SUCCESS,
        )
        queued = QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))

        payload = build_payload(
            [queued],
            source_client_version="1.0.0",
            session_id="test-session",
        )

        assert payload["clientVer"] == "1.0.0"
        assert payload["sessionId"] == "test-session"
        assert len(payload["events"]) == 1
        assert payload["events"][0]["name"] == "LauncherJobEvent"
        assert payload["eventSchemaVer"] == "1.0"


class TestConfigSubcommand:
    """Tests for the config CLI subcommand."""

    def test_config_set_telemetry_level(self, tmp_path):
        """Test config set telemetry.level writes config file."""
        from nemo_evaluator_launcher.cli.config import SetCmd

        config_file = tmp_path / "config.yaml"
        with patch("nemo_evaluator.config.CONFIG_FILE", config_file):
            SetCmd(key="telemetry.level", value="1").execute()

        import yaml

        data = yaml.safe_load(config_file.read_text())
        assert data["telemetry"]["level"] == 1

    def test_config_set_invalid_level(self, caplog):
        """Test config set with invalid level logs error."""
        import logging

        from nemo_evaluator_launcher.cli.config import SetCmd

        with caplog.at_level(logging.ERROR):
            SetCmd(key="telemetry.level", value="abc").execute()
        assert "Invalid" in caplog.text

    def test_config_set_unknown_key(self, caplog):
        """Test config set with unknown key logs error."""
        import logging

        from nemo_evaluator_launcher.cli.config import SetCmd

        with caplog.at_level(logging.ERROR):
            SetCmd(key="unknown.key", value="value").execute()
        assert "Unknown" in caplog.text

    def test_config_get_telemetry_level(self, caplog):
        """Test config get telemetry.level displays the effective level."""
        import logging

        from nemo_evaluator_launcher.cli.config import GetCmd

        with caplog.at_level(logging.INFO):
            with patch.dict(os.environ, {"NEMO_EVALUATOR_TELEMETRY_LEVEL": "1"}):
                GetCmd(key="telemetry.level").execute()

        assert "1" in caplog.text
        assert "MINIMAL" in caplog.text

    def test_config_get_unknown_key(self, caplog):
        """Test config get with unknown key logs error."""
        import logging

        from nemo_evaluator_launcher.cli.config import GetCmd

        with caplog.at_level(logging.ERROR):
            GetCmd(key="unknown.key").execute()
        assert "Unknown" in caplog.text

    def test_config_show_no_file(self, caplog):
        """Test config show when no config file exists."""
        import logging
        from unittest.mock import MagicMock

        from nemo_evaluator_launcher.cli.config import ShowCmd

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with caplog.at_level(logging.INFO):
            with patch("nemo_evaluator_launcher.cli.config.CONFIG_FILE", mock_path):
                ShowCmd().execute()

        assert "No config file" in caplog.text

    def test_config_show_with_file(self, tmp_path, caplog):
        """Test config show dumps file contents."""
        import logging

        from nemo_evaluator_launcher.cli.config import ShowCmd

        config_file = tmp_path / "config.yaml"
        config_file.write_text("telemetry:\n  level: 1\n")
        with caplog.at_level(logging.INFO):
            with patch("nemo_evaluator_launcher.cli.config.CONFIG_FILE", config_file):
                ShowCmd().execute()

        assert "telemetry:" in caplog.text
        assert "level: 1" in caplog.text


class TestLocalExecutorTelemetryPropagation:
    """Tests for telemetry env var propagation in LocalExecutor."""

    @pytest.fixture
    def sample_config(self, tmpdir):
        """Create a sample configuration for testing."""
        from omegaconf import OmegaConf

        config_dict = {
            "deployment": {"type": "none"},
            "execution": {
                "type": "local",
                "output_dir": str(tmpdir / "test_output"),
            },
            "target": {
                "api_endpoint": {
                    "api_key_name": "TEST_API_KEY",
                    "model_id": "test_model",
                    "url": "https://test.api.com/v1/chat/completions",
                }
            },
            "evaluation": {
                "tasks": [
                    {
                        "name": "test_task",
                        "nemo_evaluator_config": {
                            "config": {"params": {"param1": "value1"}}
                        },
                    },
                ],
            },
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def mock_tasks_mapping(self):
        """Mock tasks mapping for testing."""
        return {
            ("lm-eval", "test_task"): {
                "task": "test_task",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test-container:latest",
            },
        }

    def test_telemetry_env_vars_included_when_set(
        self, sample_config, mock_tasks_mapping, tmpdir
    ):
        """Test that telemetry env vars are included in Docker env vars when set."""
        os.environ["TEST_API_KEY"] = "test_key_value"
        os.environ["NEMO_EVALUATOR_TELEMETRY_SESSION_ID"] = "test-session-12345"
        os.environ["NEMO_EVALUATOR_TELEMETRY_LEVEL"] = "2"
        os.environ["NEMO_EVALUATOR_TELEMETRY_ENDPOINT"] = (
            "https://staging.example.com/v1.1/events/json"
        )

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping
                mock_get_task_def.return_value = mock_tasks_mapping[
                    ("lm-eval", "test_task")
                ]

                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="test-command",
                    debug="# Test command",
                )

                from nemo_evaluator_launcher.executors.local.executor import (
                    LocalExecutor,
                )

                invocation_id = LocalExecutor.execute_eval(sample_config, dry_run=True)

                # Verify scripts were created with telemetry env vars
                import pathlib

                output_base = pathlib.Path(sample_config.execution.output_dir)
                output_dir = None
                for item in output_base.iterdir():
                    if item.is_dir() and item.name.endswith(f"-{invocation_id}"):
                        output_dir = item
                        break

                assert output_dir is not None
                run_script = output_dir / "test_task" / "run.sh"
                assert run_script.exists()

                script_content = run_script.read_text()
                assert (
                    "NEMO_EVALUATOR_TELEMETRY_SESSION_ID=test-session-12345"  # pragma: allowlist secret
                    in script_content
                )
                assert "NEMO_EVALUATOR_TELEMETRY_LEVEL=2" in script_content
                assert (
                    "NEMO_EVALUATOR_TELEMETRY_ENDPOINT=https://staging.example.com/v1.1/events/json"
                    in script_content
                )

        finally:
            for env_var in [
                "TEST_API_KEY",
                "NEMO_EVALUATOR_TELEMETRY_SESSION_ID",
                "NEMO_EVALUATOR_TELEMETRY_LEVEL",
                "NEMO_EVALUATOR_TELEMETRY_ENDPOINT",
            ]:
                os.environ.pop(env_var, None)

    def test_telemetry_env_vars_not_included_when_not_set(
        self, sample_config, mock_tasks_mapping, tmpdir
    ):
        """Test that telemetry env vars are NOT included when not set."""
        os.environ["TEST_API_KEY"] = "test_key_value"
        # Explicitly remove telemetry env vars
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_SESSION_ID", None)
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_ENDPOINT", None)

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.local.executor.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping
                mock_get_task_def.return_value = mock_tasks_mapping[
                    ("lm-eval", "test_task")
                ]

                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="test-command",
                    debug="# Test command",
                )

                from nemo_evaluator_launcher.executors.local.executor import (
                    LocalExecutor,
                )

                invocation_id = LocalExecutor.execute_eval(sample_config, dry_run=True)

                # Verify scripts were created WITHOUT telemetry env vars
                import pathlib

                output_base = pathlib.Path(sample_config.execution.output_dir)
                output_dir = None
                for item in output_base.iterdir():
                    if item.is_dir() and item.name.endswith(f"-{invocation_id}"):
                        output_dir = item
                        break

                assert output_dir is not None
                run_script = output_dir / "test_task" / "run.sh"
                assert run_script.exists()

                script_content = run_script.read_text()
                assert "NEMO_EVALUATOR_TELEMETRY_SESSION_ID" not in script_content
                assert "NEMO_EVALUATOR_TELEMETRY_LEVEL" not in script_content
                assert "NEMO_EVALUATOR_TELEMETRY_ENDPOINT" not in script_content

        finally:
            os.environ.pop("TEST_API_KEY", None)


class TestSlurmExecutorTelemetryPropagation:
    """Tests for telemetry env var propagation in Slurm executor sbatch scripts."""

    @pytest.fixture
    def slurm_base_config(self):
        """Base configuration for SLURM tests."""
        return {
            "deployment": {
                "type": "vllm",
                "image": "test-image:latest",
                "command": "test-command",
                "served_model_name": "test-model",
                "port": 8000,
                "endpoints": {
                    "health": "/health",
                },
            },
            "execution": {
                "type": "slurm",
                "output_dir": "/test/output",
                "walltime": "01:00:00",
                "account": "test-account",
                "partition": "test-partition",
                "num_nodes": 1,
                "ntasks_per_node": 1,
                "subproject": "test-subproject",
                "env_vars": {},
                "mounts": {},
            },
            "evaluation": {
                "env_vars": {},
                "tasks": [{"name": "test_task"}],
            },
            "target": {
                "api_endpoint": {
                    "url": "https://test.api.com/v1/chat/completions",
                    "model_id": "test-model",
                }
            },
        }

    def test_sbatch_script_includes_telemetry_env_vars_when_set(
        self, slurm_base_config
    ):
        """Test that sbatch script includes telemetry env vars when set."""
        from pathlib import Path

        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.executors.slurm.executor import (
            _create_slurm_sbatch_script,
        )

        cfg = OmegaConf.create(slurm_base_config)
        task = OmegaConf.create({"name": "test_task"})

        os.environ["NEMO_EVALUATOR_TELEMETRY_SESSION_ID"] = "slurm-session-67890"
        os.environ["NEMO_EVALUATOR_TELEMETRY_LEVEL"] = "1"
        os.environ["NEMO_EVALUATOR_TELEMETRY_ENDPOINT"] = (
            "https://staging.example.com/v1.1/events/json"
        )

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
                ) as mock_get_command,
                patch(
                    "nemo_evaluator_launcher.common.helpers.get_served_model_name"
                ) as mock_get_model_name,
            ):
                mock_load_mapping.return_value = {}
                mock_get_task_def.return_value = {
                    "container": "test-eval-container:latest",
                    "required_env_vars": [],
                    "endpoint_type": "openai",
                    "task": "test_task",
                }

                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="test-eval-command",
                    debug="# Test command",
                )
                mock_get_model_name.return_value = "test-model"

                result = _create_slurm_sbatch_script(
                    cfg=cfg,
                    task=task,
                    eval_image="test:latest",
                    remote_task_subdir=Path("/tmp/test/test_task"),
                    invocation_id="test123",
                    job_id="test123.0",
                )

                script = result.cmd

                # Verify telemetry env vars are exported in the script
                assert (
                    "export NEMO_EVALUATOR_TELEMETRY_SESSION_ID=slurm-session-67890"
                    in script
                )
                assert "export NEMO_EVALUATOR_TELEMETRY_LEVEL=1" in script
                assert (
                    "export NEMO_EVALUATOR_TELEMETRY_ENDPOINT=https://staging.example.com/v1.1/events/json"
                    in script
                )

        finally:
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_SESSION_ID", None)
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_ENDPOINT", None)

    def test_sbatch_script_excludes_telemetry_env_vars_when_not_set(
        self, slurm_base_config
    ):
        """Test that sbatch script does NOT include telemetry env vars when not set."""
        from pathlib import Path

        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.executors.slurm.executor import (
            _create_slurm_sbatch_script,
        )

        cfg = OmegaConf.create(slurm_base_config)
        task = OmegaConf.create({"name": "test_task"})

        # Ensure env vars are not set
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_SESSION_ID", None)
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_ENDPOINT", None)

        with (
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.slurm.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_command,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_served_model_name"
            ) as mock_get_model_name,
        ):
            mock_load_mapping.return_value = {}
            mock_get_task_def.return_value = {
                "container": "test-eval-container:latest",
                "required_env_vars": [],
                "endpoint_type": "openai",
                "task": "test_task",
            }

            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_command.return_value = CmdAndReadableComment(
                cmd="test-eval-command",
                debug="# Test command",
            )
            mock_get_model_name.return_value = "test-model"

            result = _create_slurm_sbatch_script(
                cfg=cfg,
                task=task,
                eval_image="test:latest",
                remote_task_subdir=Path("/tmp/test/test_task"),
                invocation_id="test123",
                job_id="test123.0",
            )

            script = result.cmd

            # Verify telemetry env vars are NOT in the script
            assert "NEMO_EVALUATOR_TELEMETRY_SESSION_ID" not in script
            assert "NEMO_EVALUATOR_TELEMETRY_LEVEL" not in script
            assert "NEMO_EVALUATOR_TELEMETRY_ENDPOINT" not in script


class TestLeptonExecutorTelemetryPropagation:
    """Tests for telemetry env var propagation in Lepton executor."""

    def test_lepton_job_env_vars_include_telemetry_when_set(self, tmpdir):
        """Test that Lepton job env vars include telemetry when set."""
        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.executors.lepton.executor import LeptonExecutor

        config_dict = {
            "deployment": {"type": "none"},
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "tasks": {"node_group": "default", "env_vars": {}, "mounts": []},
                },
            },
            "evaluation": {
                "tasks": [{"name": "test_task"}],
            },
            "target": {
                "api_endpoint": {
                    "url": "https://test.endpoint.com/v1/chat/completions",
                    "model_id": "test-model",
                }
            },
        }
        cfg = OmegaConf.create(config_dict)

        mock_tasks_mapping = {
            ("lm-eval", "test_task"): {
                "task": "test_task",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            }
        }

        os.environ["NEMO_EVALUATOR_TELEMETRY_SESSION_ID"] = "lepton-session-abc123"
        os.environ["NEMO_EVALUATOR_TELEMETRY_LEVEL"] = "2"
        os.environ["NEMO_EVALUATOR_TELEMETRY_ENDPOINT"] = (
            "https://staging.example.com/v1.1/events/json"
        )

        captured_env_vars = {}

        def mock_create_job(**kwargs):
            # Capture the env_vars passed to create_lepton_job
            captured_env_vars.update(kwargs.get("env_vars", {}))
            return (True, None)

        try:
            with (
                patch(
                    "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
                ) as mock_load_mapping,
                patch(
                    "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
                ) as mock_gen_id,
                patch(
                    "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
                ) as mock_get_task_def,
                patch(
                    "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_job"
                ) as mock_create_job_fn,
                patch(
                    "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
                ) as mock_get_command,
                patch("builtins.print"),
            ):
                mock_load_mapping.return_value = mock_tasks_mapping
                mock_gen_id.return_value = "leptontest"
                mock_get_task_def.return_value = mock_tasks_mapping[
                    ("lm-eval", "test_task")
                ]
                mock_create_job_fn.side_effect = mock_create_job

                from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

                mock_get_command.return_value = CmdAndReadableComment(
                    cmd="test-command --output_dir /results",
                    debug="# Test command",
                )

                LeptonExecutor.execute_eval(cfg, dry_run=False)

                # Verify telemetry env vars were passed to create_lepton_job
                assert (
                    captured_env_vars.get("NEMO_EVALUATOR_TELEMETRY_SESSION_ID")
                    == "lepton-session-abc123"
                )
                assert captured_env_vars.get("NEMO_EVALUATOR_TELEMETRY_LEVEL") == "2"
                assert (
                    captured_env_vars.get("NEMO_EVALUATOR_TELEMETRY_ENDPOINT")
                    == "https://staging.example.com/v1.1/events/json"
                )

        finally:
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_SESSION_ID", None)
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
            os.environ.pop("NEMO_EVALUATOR_TELEMETRY_ENDPOINT", None)

    def test_lepton_job_env_vars_exclude_telemetry_when_not_set(self, tmpdir):
        """Test that Lepton job env vars exclude telemetry when not set."""
        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.executors.lepton.executor import LeptonExecutor

        config_dict = {
            "deployment": {"type": "none"},
            "execution": {
                "output_dir": str(tmpdir),
                "lepton_platform": {
                    "tasks": {"node_group": "default", "env_vars": {}, "mounts": []},
                },
            },
            "evaluation": {
                "tasks": [{"name": "test_task"}],
            },
            "target": {
                "api_endpoint": {
                    "url": "https://test.endpoint.com/v1/chat/completions",
                    "model_id": "test-model",
                }
            },
        }
        cfg = OmegaConf.create(config_dict)

        mock_tasks_mapping = {
            ("lm-eval", "test_task"): {
                "task": "test_task",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test:latest",
            }
        }

        # Ensure env vars are not set
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_SESSION_ID", None)
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_LEVEL", None)
        os.environ.pop("NEMO_EVALUATOR_TELEMETRY_ENDPOINT", None)

        captured_env_vars = {}

        def mock_create_job(**kwargs):
            # Capture the env_vars passed to create_lepton_job
            captured_env_vars.update(kwargs.get("env_vars", {}))
            return (True, None)

        with (
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.load_tasks_mapping"
            ) as mock_load_mapping,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.generate_invocation_id"
            ) as mock_gen_id,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.get_task_definition_for_job"
            ) as mock_get_task_def,
            patch(
                "nemo_evaluator_launcher.executors.lepton.executor.create_lepton_job"
            ) as mock_create_job_fn,
            patch(
                "nemo_evaluator_launcher.common.helpers.get_eval_factory_command"
            ) as mock_get_command,
            patch("builtins.print"),
        ):
            mock_load_mapping.return_value = mock_tasks_mapping
            mock_gen_id.return_value = "leptontest2"
            mock_get_task_def.return_value = mock_tasks_mapping[
                ("lm-eval", "test_task")
            ]
            mock_create_job_fn.side_effect = mock_create_job

            from nemo_evaluator_launcher.common.helpers import CmdAndReadableComment

            mock_get_command.return_value = CmdAndReadableComment(
                cmd="test-command --output_dir /results",
                debug="# Test command",
            )

            LeptonExecutor.execute_eval(cfg, dry_run=False)

            # Verify telemetry env vars were NOT passed to create_lepton_job
            assert "NEMO_EVALUATOR_TELEMETRY_SESSION_ID" not in captured_env_vars
            assert "NEMO_EVALUATOR_TELEMETRY_LEVEL" not in captured_env_vars
            assert "NEMO_EVALUATOR_TELEMETRY_ENDPOINT" not in captured_env_vars
