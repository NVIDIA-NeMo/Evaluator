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
import pathlib
import time
from dataclasses import dataclass

from simple_parsing import field


@dataclass
class Cmd:
    """Run command parameters"""

    config_name: str = field(
        default="default",
        alias=["-c", "--config-name"],
        metadata={
            "help": "Config name to use. Consult `nemo_evaluator_launcher.configs`"
        },
    )
    config_dir: str | None = field(
        default=None,
        alias=["-d", "--config-dir"],
        metadata={
            "help": "Path to user config directory. If provided, searches here first, then falls back to internal configs."
        },
    )
    run_config_file: str | None = field(
        default=None,
        alias=["-f", "--run-config-file"],
        metadata={
            "help": "Path to a run config file to load directly (bypasses Hydra config loading)."
        },
    )
    override: list[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-o"],
        metadata={
            "help": "Hydra override in the form some.param.path=value (pass multiple `-o` for multiple overrides).",
        },
    )
    dry_run: bool = field(
        default=False,
        alias=["-n", "--dry-run"],
        metadata={"help": "Do not run the evaluation, just print the config."},
    )

    def execute(self) -> None:
        # Import heavy dependencies only when needed
        import yaml
        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.api.functional import RunConfig, run_eval

        # Load configuration either from Hydra or from a run config file
        if self.run_config_file:
            # Validate that run config file is not used with other config options
            if self.config_name != "default":
                raise ValueError("Cannot use --run-config-file with --config-name")
            if self.config_dir is not None:
                raise ValueError("Cannot use --run-config-file with --config-dir")
            if self.override:
                raise ValueError("Cannot use --run-config-file with --override")

            # Load from run config file
            with open(self.run_config_file, "r") as f:
                config_dict = yaml.safe_load(f)

            # Create RunConfig from the loaded data
            config = OmegaConf.create(config_dict)
        else:
            # Load the complete Hydra configuration
            config = RunConfig.from_hydra(
                config_name=self.config_name,
                hydra_overrides=self.override,
                config_dir=self.config_dir,
            )

        invocation_id = run_eval(config, self.dry_run)

        # Save the complete configuration to the raw_configs directory
        if not self.dry_run and invocation_id is not None:
            # Create ~/.nemo-evaluator/run_configs directory
            home_dir = pathlib.Path.home()
            run_configs_dir = home_dir / ".nemo-evaluator" / "run_configs"
            run_configs_dir.mkdir(parents=True, exist_ok=True)

            # Convert DictConfig to dict and save as YAML
            config_dict = OmegaConf.to_container(config, resolve=True)
            config_yaml = yaml.dump(
                config_dict, default_flow_style=False, sort_keys=False, indent=2
            )

            # Create config filename with invocation ID
            config_filename = f"{invocation_id}_config.yml"
            config_path = run_configs_dir / config_filename

            # Save the complete Hydra configuration
            with open(config_path, "w") as f:
                f.write("# Complete configuration from nemo-evaluator-launcher\n")
                f.write(
                    f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
                )
                f.write(f"# Invocation ID: {invocation_id}\n")
                f.write("#\n")
                f.write("# This is the complete raw configuration\n")
                f.write("#\n")
                f.write("# To rerun this exact configuration:\n")
                f.write(
                    f"# nemo-evaluator-launcher run --run-config-file {config_path}\n"
                )
                f.write("#\n")
                f.write(config_yaml)

            print(f"Complete run config saved to: {config_path}")

        if invocation_id is not None:
            print(f"to check status: nemo-evaluator-launcher status {invocation_id}")
            print(f"to kill all jobs: nemo-evaluator-launcher kill {invocation_id}")
            print(
                f"to kill individual jobs: nemo-evaluator-launcher kill <job_id> (e.g., {invocation_id}.0)"
            )
