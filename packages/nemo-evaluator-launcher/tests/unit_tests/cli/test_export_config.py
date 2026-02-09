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
"""Tests for export CLI --config flag."""

import tempfile
from pathlib import Path

import pytest
import yaml

from nemo_evaluator_launcher.cli.export import ExportCmd


class TestExportConfigFlag:
    """Tests for --config flag in export CLI."""

    def test_load_flat_config(self):
        """Test loading a flat config structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "output_dir": "./test_results",
                "format": "csv",
                "copy_logs": True,
                "only_required": False,
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                config=config_path,
            )
            loaded_config = cmd._load_config_from_file(config_path)

            assert loaded_config["output_dir"] == "./test_results"
            assert loaded_config["format"] == "csv"
            assert loaded_config["copy_logs"] is True
            assert loaded_config["only_required"] is False
        finally:
            Path(config_path).unlink()

    def test_load_structured_config_with_dest(self):
        """Test loading a structured config with export.<dest> sections."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "export": {
                    "local": {
                        "output_dir": "./local_results",
                        "format": "json",
                    },
                    "wandb": {
                        "entity": "my-org",
                        "project": "my-project",
                    },
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Test local destination
            cmd_local = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                config=config_path,
            )
            loaded_config = cmd_local._load_config_from_file(config_path)
            assert loaded_config["output_dir"] == "./local_results"
            assert loaded_config["format"] == "json"

            # Test wandb destination
            cmd_wandb = ExportCmd(
                invocation_ids=["test123"],
                dest="wandb",
                config=config_path,
            )
            loaded_config = cmd_wandb._load_config_from_file(config_path)
            assert loaded_config["entity"] == "my-org"
            assert loaded_config["project"] == "my-project"
        finally:
            Path(config_path).unlink()

    def test_cli_args_override_config_file(self, monkeypatch):
        """Test that CLI arguments override config file values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "output_dir": "./config_results",
                "format": "csv",
                "copy_logs": False,
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Mock export_results to capture the config
            captured_config = {}

            def mock_export(ids, dest, config):
                captured_config.update(config)
                return {
                    "success": True,
                    "metadata": {"successful_jobs": 1, "failed_jobs": 0},
                }

            monkeypatch.setattr(
                "nemo_evaluator_launcher.api.functional.export_results", mock_export
            )

            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                config=config_path,
                output_dir="./cli_results",  # CLI override
                format="json",  # CLI override
                copy_logs=True,  # CLI override
            )
            cmd.execute()

            # Verify CLI args took precedence
            assert captured_config["output_dir"] == "./cli_results"
            assert captured_config["format"] == "json"
            assert captured_config["copy_logs"] is True
        finally:
            Path(config_path).unlink()

    def test_config_with_top_level_and_dest_specific(self):
        """Test config with both top-level and destination-specific settings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "copy_logs": True,  # Top-level default
                "export": {
                    "local": {
                        "output_dir": "./local_results",
                        "format": "json",
                    },
                },
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                config=config_path,
            )
            loaded_config = cmd._load_config_from_file(config_path)

            # Should have both top-level and dest-specific values
            assert loaded_config["copy_logs"] is True
            assert loaded_config["output_dir"] == "./local_results"
            assert loaded_config["format"] == "json"
        finally:
            Path(config_path).unlink()

    def test_config_file_not_dict(self):
        """Test error handling when config file is not a dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "dict"], f)
            config_path = f.name

        try:
            cmd = ExportCmd(
                invocation_ids=["test123"],
                dest="local",
                config=config_path,
            )

            with pytest.raises(
                ValueError, match="must contain a dictionary at the root level"
            ):
                cmd._load_config_from_file(config_path)
        finally:
            Path(config_path).unlink()
