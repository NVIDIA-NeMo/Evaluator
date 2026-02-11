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
"""Tests for the CLI ls executors and ls deployments commands."""

import json
from contextlib import redirect_stdout
from io import StringIO

import pytest

from nemo_evaluator_launcher.cli.ls_deployments import Cmd as LsDeploymentsCmd
from nemo_evaluator_launcher.cli.ls_deployments import DEPLOYMENTS
from nemo_evaluator_launcher.cli.ls_executors import Cmd as LsExecutorsCmd
from nemo_evaluator_launcher.cli.ls_executors import EXECUTORS


class TestLsExecutorsCommand:
    """Tests for ls executors command."""

    def test_formatted_output(self):
        """Test formatted output contains expected executors."""
        cmd = LsExecutorsCmd(json=False)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue()

        # Verify all executors are listed
        assert "local" in result
        assert "slurm" in result
        assert "lepton" in result

        # Verify descriptions
        assert "Docker" in result
        assert "SLURM" in result or "HPC" in result

        # Verify required flags are shown
        assert "--output-dir" in result
        assert "--slurm-hostname" in result
        assert "--slurm-account" in result

    def test_json_output(self):
        """Test JSON output format."""
        cmd = LsExecutorsCmd(json=True)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue()
        parsed = json.loads(result)

        assert "executors" in parsed
        assert "local" in parsed["executors"]
        assert "slurm" in parsed["executors"]
        assert "lepton" in parsed["executors"]

        # Verify structure
        local = parsed["executors"]["local"]
        assert "description" in local
        assert "required" in local
        assert "default" in local
        assert local["default"] is True

    def test_executors_dict_structure(self):
        """Test EXECUTORS dict has correct structure."""
        assert "local" in EXECUTORS
        assert "slurm" in EXECUTORS
        assert "lepton" in EXECUTORS

        for name, info in EXECUTORS.items():
            assert "description" in info
            assert "required" in info
            assert "optional" in info
            assert "default" in info
            assert isinstance(info["required"], list)
            assert isinstance(info["optional"], list)
            assert isinstance(info["default"], bool)

    def test_only_one_default_executor(self):
        """Test that only one executor is marked as default."""
        defaults = [name for name, info in EXECUTORS.items() if info["default"]]
        assert len(defaults) == 1
        assert defaults[0] == "local"


class TestLsDeploymentsCommand:
    """Tests for ls deployments command."""

    def test_formatted_output(self):
        """Test formatted output contains expected deployments."""
        cmd = LsDeploymentsCmd(json=False)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue()

        # Verify all deployments are listed
        assert "none" in result
        assert "vllm" in result
        assert "nim" in result
        assert "sglang" in result

        # Verify descriptions
        assert "API endpoint" in result or "model deployment" in result
        assert "vLLM" in result

        # Verify required flags are shown
        assert "--model" in result
        assert "--checkpoint" in result
        assert "--model-name" in result

    def test_json_output(self):
        """Test JSON output format."""
        cmd = LsDeploymentsCmd(json=True)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue()
        parsed = json.loads(result)

        assert "deployments" in parsed
        assert "none" in parsed["deployments"]
        assert "vllm" in parsed["deployments"]
        assert "nim" in parsed["deployments"]
        assert "sglang" in parsed["deployments"]

        # Verify structure
        none_deployment = parsed["deployments"]["none"]
        assert "description" in none_deployment
        assert "required" in none_deployment
        assert "default" in none_deployment
        assert none_deployment["default"] is True

    def test_deployments_dict_structure(self):
        """Test DEPLOYMENTS dict has correct structure."""
        assert "none" in DEPLOYMENTS
        assert "vllm" in DEPLOYMENTS
        assert "nim" in DEPLOYMENTS
        assert "sglang" in DEPLOYMENTS

        for name, info in DEPLOYMENTS.items():
            assert "description" in info
            assert "required" in info
            assert "optional" in info
            assert "default" in info
            assert isinstance(info["required"], list)
            assert isinstance(info["optional"], list)
            assert isinstance(info["default"], bool)

    def test_only_one_default_deployment(self):
        """Test that only one deployment is marked as default."""
        defaults = [name for name, info in DEPLOYMENTS.items() if info["default"]]
        assert len(defaults) == 1
        assert defaults[0] == "none"

    def test_vllm_requires_checkpoint_or_hf_model(self):
        """Test vLLM shows checkpoint OR hf-model as required."""
        vllm_info = DEPLOYMENTS["vllm"]
        required_str = " ".join(vllm_info["required"])
        # Either checkpoint or hf-model should be mentioned as required
        assert "checkpoint" in required_str or "hf-model" in required_str


class TestLsCommandIntegration:
    """Integration tests for ls commands."""

    def test_executors_default_matches_local(self):
        """Test that local is correctly marked as default in formatted output."""
        cmd = LsExecutorsCmd(json=False)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue()
        # Local should be marked as default
        assert "local" in result and "default" in result

    def test_deployments_default_matches_none(self):
        """Test that none is correctly marked as default in formatted output."""
        cmd = LsDeploymentsCmd(json=False)

        output = StringIO()
        with redirect_stdout(output):
            cmd.execute()

        result = output.getvalue()
        # None should be marked as default
        assert "none" in result and "default" in result
