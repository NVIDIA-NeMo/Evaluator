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
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.printing_utils import (
    bold,
    cyan,
    green,
    magenta,
    red,
)


def _redact_value(value: str) -> str:
    if len(value) <= 4:
        return "***"
    return "***" + value[-4:]


@dataclass
class Cmd:
    """Run command parameters"""

    config: str | None = field(
        default=None,
        alias=["--config"],
        metadata={
            "help": "Full path to config file. Uses Hydra by default (--config-mode=hydra). Use --config-mode=raw to load directly (bypasses Hydra)."
        },
    )
    config_mode: Literal["hydra", "raw"] = field(
        default="hydra",
        alias=["--config-mode"],
        metadata={
            "help": "Config loading mode: 'hydra' (default) uses Hydra config system, 'raw' loads config file directly bypassing Hydra."
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
    tasks: list[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-t"],
        metadata={
            "help": "Run only specific tasks from the config. Example: -t ifeval -t gsm8k"
        },
    )
    config_output: str | None = field(
        default=None,
        alias=["--config-output"],
        metadata={
            "help": "Directory to save the complete run config. Defaults to ~/.nemo-evaluator/run_configs/"
        },
    )
    env_file: str | None = field(
        default=None,
        alias=["--env-file"],
        metadata={
            "help": "Path to .env file to load environment variables from. "
            "If not specified, loads $PWD/.env if it exists."
        },
    )

    def _parse_requested_tasks(self) -> list[str]:
        """Parse -t arguments into a list of task names.

        Handles None values that can be appended when using nargs="?" with action="append".
        """
        requested_tasks = []
        for task_arg in self.tasks:
            # Skip None or empty values (can happen with nargs="?")
            if not task_arg:
                continue
            task_name = task_arg.strip()
            if task_name and task_name not in requested_tasks:
                requested_tasks.append(task_name)
        return requested_tasks

    def _load_env_file(self) -> None:
        """Load environment variables from a .env file.

        Uses --env-file path if specified, otherwise loads $PWD/.env if it exists.
        Logs loaded keys with redacted values and warns about shadowed keys.
        """
        if self.env_file:
            env_path = pathlib.Path(self.env_file)
            if not env_path.is_file():
                raise FileNotFoundError(
                    f"Specified --env-file '{self.env_file}' does not exist."
                )
        else:
            env_path = pathlib.Path.cwd() / ".env"
            if not env_path.is_file():
                return

        # Parse first to inspect keys before loading
        parsed = dotenv_values(env_path)

        shadowed_keys = set()
        for key, value in parsed.items():
            if value is None:
                continue
            if key in os.environ:
                shadowed_keys.add(key)
                logger.warning(
                    "Env file key already set in environment, skipping",
                    key=key,
                    env_file_value=_redact_value(value),
                    existing_value=_redact_value(os.environ[key]),
                )

        load_dotenv(env_path, override=False)

        loaded = {
            k: _redact_value(v)
            for k, v in parsed.items()
            if v is not None and k not in shadowed_keys
        }
        logger.info(
            "Loaded env file",
            path=str(env_path),
            loaded_keys=loaded,
            skipped_keys=shadowed_keys,
        )

    def execute(self) -> None:
        # Load .env file before anything else
        self._load_env_file()

        # Import heavy dependencies only when needed
        import yaml
        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.api.functional import (
            RunConfig,
            filter_tasks,
            run_eval,
        )

        # Validate config_mode value
        if self.config_mode not in ["hydra", "raw"]:
            raise ValueError(
                f"Invalid --config-mode value: {self.config_mode}. Must be 'hydra' or 'raw'."
            )

        # Validate that raw mode requires --config
        if self.config_mode == "raw" and self.config is None:
            raise ValueError(
                "--config-mode=raw requires --config to be specified. Raw mode loads config files directly."
            )

        # Parse requested tasks if -t is specified
        requested_tasks = self._parse_requested_tasks() if self.tasks else None

        # Load configuration either from Hydra or directly from a config file
        if self.config_mode == "raw" and self.config:
            # Validate that raw config loading is not used with other config options
            if self.override:
                raise ValueError(
                    "Cannot use --config-mode=raw with --override. Raw mode only works with --config."
                )

            # Load from config file directly (bypass Hydra)
            with open(self.config, "r") as f:
                config_dict = yaml.safe_load(f)

            # Create RunConfig from the loaded data
            config = OmegaConf.create(config_dict)
        else:
            # Load the complete Hydra configuration
            config = RunConfig.from_hydra(
                config=self.config,
                hydra_overrides=self.override,
            )

        # Apply task filtering if -t is specified
        if requested_tasks:
            config = filter_tasks(config, requested_tasks)
            logger.info(
                "Running filtered tasks",
                count=len(config.evaluation.tasks),
                tasks=[t.name for t in config.evaluation.tasks],
            )

        try:
            invocation_id = run_eval(config, self.dry_run)
        except Exception as e:
            print(red(f"✗ Job submission failed, see logs | Error: {e}"))
            logger.error("Job submission failed", error=e)
            raise

        # Save the complete configuration
        if not self.dry_run and invocation_id is not None:
            # Determine config output directory
            if self.config_output:
                # Use custom directory specified by --config-output
                config_dir = pathlib.Path(self.config_output)
            else:
                # Default to original location: ~/.nemo-evaluator/run_configs
                home_dir = pathlib.Path.home()
                config_dir = home_dir / ".nemo-evaluator" / "run_configs"

            # Ensure the directory exists
            config_dir.mkdir(parents=True, exist_ok=True)

            # Convert DictConfig to dict and save as YAML
            config_dict = OmegaConf.to_container(config, resolve=True)
            config_yaml = yaml.dump(
                config_dict, default_flow_style=False, sort_keys=False, indent=2
            )

            # Create config filename with invocation ID
            config_filename = f"{invocation_id}_config.yml"
            config_path = config_dir / config_filename

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
                    f"# nemo-evaluator-launcher run --config {config_path} --config-mode=raw\n"
                )
                f.write("#\n")
                f.write(config_yaml)

            print(bold(cyan("Complete run config saved to: ")) + f"\n  {config_path}\n")
            logger.info("Saved complete config", path=config_path)

        # Print general success message with invocation ID and helpful commands
        if invocation_id is not None and not self.dry_run:
            print(
                bold(cyan("To check status: "))
                + f"nemo-evaluator-launcher status {invocation_id}"
            )
            print(
                bold(cyan("To view job info: "))
                + f"nemo-evaluator-launcher info {invocation_id}"
            )
            print(
                bold(cyan("To kill all jobs: "))
                + f"nemo-evaluator-launcher kill {invocation_id}"
            )

            # Show actual job IDs and task names
            print(bold(cyan("To kill individual jobs:")))
            # Access tasks - will work after normalization in run_eval
            tasks = (
                config.evaluation.tasks
                if hasattr(config.evaluation, "tasks")
                else config.evaluation
            )
            for idx, task in enumerate(tasks):
                job_id = f"{invocation_id}.{idx}"
                print(f"  nemo-evaluator-launcher kill {job_id}  # {task.name}")

            print(
                magenta(
                    "(all commands accept shortened IDs as long as there are no conflicts)"
                )
            )
            print(
                bold(cyan("To print all jobs: ")) + "nemo-evaluator-launcher ls runs"
                "\n  (--since 1d or --since 6h for time span, see --help)"
            )

            print(
                green(
                    bold(
                        f"✓ Job submission successful | Invocation ID: {invocation_id}"
                    )
                )
            )

            # Show a one-time telemetry hint when telemetry is active
            from nemo_evaluator.config import TelemetryLevel
            from nemo_evaluator.telemetry import get_telemetry_level

            if get_telemetry_level() != TelemetryLevel.OFF:
                print(
                    magenta(
                        "Telemetry is active. Run 'nel config set telemetry.level 0' to disable."
                    )
                )

        # Done.
