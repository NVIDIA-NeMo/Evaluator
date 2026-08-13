# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Tests for the select_not_null resolver and vllm deployment command construction."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.types import RunConfig

# NOTE(martas): import to register the resolver
from nemo_evaluator_launcher.configs import select_not_null  # noqa: F401


class TestSelectNotNullResolver:
    """Tests for the select_not_null custom OmegaConf resolver."""

    def test_string_value_returned_as_is(self):
        cfg = OmegaConf.create(
            {
                "some_config": {"some_field": "some_value"},
                "result": "${select_not_null:some_config.some_field,default_value}",
            }
        )
        assert cfg.result == "some_value"

    def test_null_value_returns_default(self):
        cfg = OmegaConf.create(
            {
                "some_config": {"some_field": None},
                "result": "${select_not_null:some_config.some_field,default_value}",
            }
        )
        assert cfg.result == "default_value"

    def test_missing_key_returns_default(self):
        cfg = OmegaConf.create(
            {
                "some_config": {"some_other_field": "some_value"},
                "result": "${select_not_null:some_config.some_field,default_value}",
            }
        )
        assert cfg.result == "default_value"


class TestVllmDeploymentCommand:
    """Tests for the vllm deployment command string construction."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self):
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    def _load_vllm_cfg(self, checkpoint_path, **kwargs):
        """Load config via Hydra with deployment=vllm, the same way nel run does."""
        overrides = [
            "deployment=vllm",
            "deployment.served_model_name=test-model",
            f"deployment.checkpoint_path={checkpoint_path}",
        ]
        for key, value in kwargs.items():
            if value is None:
                value = "null"
            overrides.append(f"++deployment.{key}={value}")
        return RunConfig.from_hydra(hydra_overrides=overrides)

    def test_checkpoint_path_set_hf_handle_missing_uses_checkpoint(self):
        """hf_model_handle absent from config → command falls back to /checkpoint."""
        cfg = self._load_vllm_cfg(checkpoint_path="/my/checkpoint")
        assert cfg.deployment.command.startswith("vllm serve /checkpoint")

    def test_checkpoint_path_set_hf_handle_null_uses_checkpoint(self):
        """hf_model_handle: null → command falls back to /checkpoint."""
        cfg = self._load_vllm_cfg(
            checkpoint_path="/my/checkpoint", hf_model_handle=None
        )
        assert cfg.deployment.command.startswith("vllm serve /checkpoint")

    def test_both_set_hf_handle_takes_precedence(self):
        """Both checkpoint_path and hf_model_handle set → hf_model_handle used."""
        cfg = self._load_vllm_cfg(
            checkpoint_path="/my/checkpoint",
            hf_model_handle="meta-llama/Llama-3.1-8B",
        )
        command = cfg.deployment.command
        assert "meta-llama/Llama-3.1-8B" in command
        assert command.startswith("vllm serve meta-llama/Llama-3.1-8B")


class TestVllmPdDeploymentCommand:
    """Tests for the vLLM prefill/decode deployment command."""

    @pytest.fixture(autouse=True)
    def clear_hydra(self):
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    def test_pd_command_defines_router_and_rank_roles(self):
        cfg = RunConfig.from_hydra(
            hydra_overrides=[
                "deployment=vllm_pd",
                "deployment.served_model_name=test-model",
                "deployment.checkpoint_path=/my/checkpoint",
                "deployment.prefill_nodes=1",
                "deployment.decode_nodes=1",
            ]
        )

        command = cfg.deployment.command

        assert "vllm-router" in command
        assert 'if [ "$PROC_ID" -eq 0 ]' in command
        assert 'elif [ "$PROC_ID" -lt "$PREFILL_NODES" ]' in command
        assert '"kv_role":"kv_producer"' in command
        assert '"kv_role":"kv_consumer"' in command
        assert "NODE_IPS_FILE" in command
        assert "/results/vllm_pd_node_ips.txt" in command
        assert "ALL_NODE_IPS" not in command
        assert "awk 'END { print NR }'" in command
        assert "VLLM_PD_ROLE=prefill-head-router" in command
        assert "VLLM_PD_ROLE=decode-worker" in command
        assert "write_role_status" in command
        assert "$ROLE_STATUS_DIR/$VLLM_PD_ROLE-$PROC_ID.status" in command
        assert "$ROLE_LOG_DIR/$VLLM_PD_ROLE-$PROC_ID.log" in command
        assert "exit_code=%s" in command
        assert "trap on_exit EXIT" in command
        assert "write_role_status failed" in command
        assert "write_runtime_provenance" in command
        assert "$RUNTIME_PROVENANCE_DIR/runtime.env" in command
        assert '"fastokens", "instanttensor", "nixl"' in command
        assert cfg.deployment.mount_checkpoint_to_evaluation is False
        assert cfg.deployment.role_status_dir == ""
        assert cfg.deployment.role_log_dir == ""
        assert cfg.deployment.runtime_provenance_dir == ""
        assert cfg.deployment.runtime_image_sha256 == ""
        assert cfg.deployment.node_ips_file == "/results/vllm_pd_node_ips.txt"

        result = subprocess.run(
            ["bash", "-n"], input=command, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr

    def test_pd_command_supports_a_router_lifecycle_hook(self):
        cfg = RunConfig.from_hydra(
            hydra_overrides=[
                "deployment=vllm_pd",
                "deployment.served_model_name=test-model",
                "deployment.checkpoint_path=/my/checkpoint",
                "deployment.prefill_nodes=1",
                "deployment.decode_nodes=1",
                "deployment.router_background_command='echo gym-coordinator'",
            ]
        )

        command = cfg.deployment.command

        assert "VLLM_PD_ROUTER_BACKGROUND_COMMAND" in command
        assert 'source "$router_background_command_file"' in command
        assert "wait_for_router_or_background" in command
        assert "echo gym-coordinator" in command

        result = subprocess.run(
            ["bash", "-n"], input=command, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr

    def test_pd_command_runs_router_lifecycle_hook(self, tmp_path: Path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "vllm").write_text("#!/bin/sh\nexec sleep 30\n")
        (bin_dir / "vllm-router").write_text(
            '#!/bin/sh\nif [ "${1:-}" = "--version" ]; then exit 0; fi\nexec sleep 30\n'
        )
        for executable in bin_dir.iterdir():
            executable.chmod(0o755)

        role_log_dir = tmp_path / "role-logs"
        hook_marker = role_log_dir / "router-background-hook-ran"
        node_ips_file = tmp_path / "node-ips.txt"
        node_ips_file.write_text("10.0.0.1\n10.0.0.2\n")
        cfg = RunConfig.from_hydra(
            hydra_overrides=[
                "deployment=vllm_pd",
                "deployment.served_model_name=test-model",
                "deployment.checkpoint_path=/my/checkpoint",
                "deployment.prefill_nodes=1",
                "deployment.decode_nodes=1",
                f"deployment.node_ips_file={node_ips_file}",
                f"deployment.role_log_dir={role_log_dir}",
                "deployment.router_background_command='touch \"$ROLE_LOG_DIR/router-background-hook-ran\"'",
            ]
        )
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "NODES_PER_INSTANCE": "2",
            "PROC_ID": "0",
        }

        result = subprocess.run(
            ["bash", "-c", cfg.deployment.command],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
        )

        assert result.returncode == 0, result.stderr
        assert hook_marker.exists()

    @pytest.mark.parametrize(
        (
            "proc_id",
            "vllm_exit_code",
            "expected_role",
            "expected_node_ip",
            "expected_state",
        ),
        [
            (0, 0, "prefill-head-router", "10.0.0.1", "stopped"),
            (1, 0, "decode-head", "10.0.0.2", "stopped"),
            (1, 7, "decode-head", "10.0.0.2", "failed"),
        ],
    )
    def test_pd_command_writes_terminal_role_status(
        self,
        tmp_path: Path,
        proc_id: int,
        vllm_exit_code: int,
        expected_role: str,
        expected_node_ip: str,
        expected_state: str,
    ):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for executable in ("vllm", "vllm-router"):
            command = bin_dir / executable
            command.write_text('#!/bin/sh\nexit "${FAKE_VLLM_EXIT:-0}"\n')
            command.chmod(0o755)

        status_dir = tmp_path / "role-status"
        log_dir = tmp_path / "role-logs"
        runtime_dir = tmp_path / "runtime"
        node_ips_file = tmp_path / "node-ips.txt"
        node_ips_file.write_text("10.0.0.1\n10.0.0.2\n")
        cfg = RunConfig.from_hydra(
            hydra_overrides=[
                "deployment=vllm_pd",
                "deployment.served_model_name=test-model",
                "deployment.checkpoint_path=/my/checkpoint",
                "deployment.prefill_nodes=1",
                "deployment.decode_nodes=1",
                f"deployment.role_status_dir={status_dir}",
                f"deployment.role_log_dir={log_dir}",
                f"deployment.runtime_provenance_dir={runtime_dir}",
                f"deployment.runtime_image_sha256={'a' * 64}",
                f"deployment.node_ips_file={node_ips_file}",
            ]
        )
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_VLLM_EXIT": str(vllm_exit_code),
            "NODES_PER_INSTANCE": "2",
            "PROC_ID": str(proc_id),
        }

        result = subprocess.run(
            ["bash", "-c", cfg.deployment.command],
            env=env,
            text=True,
            capture_output=True,
        )

        assert result.returncode == vllm_exit_code, result.stderr
        assert (status_dir / f"{expected_role}-{proc_id}.status").read_text() == (
            f"state={expected_state}\n"
            f"role={expected_role}\n"
            f"rank={proc_id}\n"
            f"node_ip={expected_node_ip}\n"
            f"exit_code={vllm_exit_code}\n"
        )
        assert (
            f"vllm_pd role={expected_role} rank={proc_id}"
            in (log_dir / f"{expected_role}-{proc_id}.log").read_text()
        )
        if proc_id == 0:
            provenance = (runtime_dir / "runtime.env").read_text()
            assert f"configured_image_sha256={'a' * 64}" in provenance
            assert "router_command_version=" in provenance
            assert "vllm=" in provenance
        else:
            assert not (runtime_dir / "runtime.env").exists()
