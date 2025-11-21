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
"""Minimal tests for example configuration files with dry_run."""

import pathlib

import pytest
from hydra.core.global_hydra import GlobalHydra

from nemo_evaluator_launcher.api.functional import RunConfig, run_eval

# Get the examples directory path
EXAMPLES_DIR = (pathlib.Path(__file__).parent.parent.parent / "examples").absolute()

# Discover all YAML example files
EXAMPLE_YAMLS = [f.stem for f in EXAMPLES_DIR.glob("*.yaml") if f.is_file()]


class TestExampleConfigs:
    """Test that all example configuration files can be initialized with dry_run."""

    @pytest.mark.parametrize("config_name", EXAMPLE_YAMLS)
    def test_example_config_dry_run(self, config_name, mock_execdb, setup_env_vars):
        """Test that example configs can be loaded and run in dry_run mode."""
        # Skip lepton configs with deployment (they try to create real endpoints even in dry_run)
        if (
            config_name.startswith("lepton_")
            and config_name != "lepton_none_llama_3_1_8b_instruct"
        ):
            pytest.skip(
                "Lepton configs with deployment try to create endpoints even in dry_run mode"
            )

        # Clear Hydra instance
        GlobalHydra.instance().clear()

        try:
            # Build overrides: output_dir for all, plus slurm-specific overrides
            overrides = ["execution.output_dir=/tmp/test_output"]

            # Add slurm-specific overrides for configs starting with "slurm_"
            if config_name.startswith("slurm_"):
                overrides.extend(
                    [
                        "++execution.type=slurm",
                        "++execution.hostname=test-slurm-host",
                        "++execution.account=test-account",
                        "++deployment.checkpoint_path=null",
                    ]
                )
                # Auto-export specific overrides
                if "auto_export" in config_name:
                    overrides.extend(
                        [
                            "++execution.env_vars.export.PATH=/tmp/test/bin:$PATH",
                            "++export.mlflow.tracking_uri=http://test-mlflow:5000",
                        ]
                    )
            # Load configuration using RunConfig.from_hydra (same as CLI)
            cfg = RunConfig.from_hydra(
                config_name=config_name,
                hydra_overrides=overrides,
                config_dir=str(EXAMPLES_DIR),
            )

            # Run with dry_run - should not raise
            invocation_id = run_eval(cfg, dry_run=True)
            assert invocation_id is not None
            assert len(invocation_id) == 16  # Standard invocation ID length
        finally:
            GlobalHydra.instance().clear()
