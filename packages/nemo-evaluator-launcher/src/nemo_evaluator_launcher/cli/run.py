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
from typing import Literal

from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.printing_utils import (
    bold,
    cyan,
    green,
    magenta,
    red,
    yellow,
)


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
        alias=["-t", "--tasks"],
        metadata={
            "help": "Run only specific tasks from the config (comma-separated or multiple -t flags). Example: -t ifeval -t gsm8k or -t ifeval,gsm8k"
        },
    )
    config_output: str | None = field(
        default=None,
        alias=["--config-output"],
        metadata={
            "help": "Directory to save the complete run config. Defaults to ~/.nemo-evaluator/run_configs/"
        },
    )

    def _parse_requested_tasks(self) -> set[str]:
        """Parse --tasks arguments into a set of task names."""
        requested_tasks = set()
        for task_arg in self.tasks:
            for task_name in task_arg.split(","):
                task_name = task_name.strip()
                if task_name:
                    requested_tasks.add(task_name)
        return requested_tasks

    def _build_task_filter_overrides(
        self, config, requested_tasks: set[str]
    ) -> list[str]:
        """Build Hydra overrides to filter tasks based on requested task names.

        Returns list of Hydra override strings that will set unwanted tasks to null.
        The nulls are filtered out after config is loaded.

        This preserves task-specific overrides that were applied earlier.
        """
        # Get original tasks
        original_tasks = (
            config.evaluation.tasks
            if hasattr(config.evaluation, "tasks")
            else config.evaluation
        )

        # Find indices of tasks to remove (set to null)
        indices_to_remove = []
        found_names = set()
        for idx, task in enumerate(original_tasks):
            if task.name in requested_tasks:
                found_names.add(task.name)
            else:
                indices_to_remove.append(idx)

        # Fail if ANY requested tasks are not found
        not_found = requested_tasks - found_names
        if not_found:
            available = [task.name for task in original_tasks]
            raise ValueError(
                f"Requested task(s) not found in config: {sorted(not_found)}. "
                f"Available tasks: {available}"
            )

        # Build overrides to set unwanted tasks to null
        # Using evaluation.tasks.INDEX=null syntax
        # Note: Hydra's ~evaluation.tasks.INDEX deletion doesn't work with list indices
        overrides = []
        for idx in indices_to_remove:
            overrides.append(f"++evaluation.tasks.{idx}=null")

        logger.info(
            "Filtering tasks via Hydra overrides",
            keep_count=len(original_tasks) - len(indices_to_remove),
            remove_count=len(indices_to_remove),
            tasks=sorted(found_names),
            overrides=overrides,
        )

        return overrides

    def execute(self) -> None:
        # Import heavy dependencies only when needed
        import yaml
        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.api.functional import RunConfig, run_eval

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

        # Parse requested tasks if --tasks is specified
        requested_tasks = self._parse_requested_tasks() if self.tasks else None

        # Load configuration either from Hydra or directly from a config file
        if self.config_mode == "raw" and self.config:
            # Validate that raw config loading is not used with other config options
            if self.config_name != "default":
                raise ValueError(
                    "Cannot use --config-mode=raw with --config-name. Raw mode only works with --config."
                )
            if self.config_dir is not None:
                raise ValueError(
                    "Cannot use --config-mode=raw with --config-dir. Raw mode only works with --config."
                )
            if self.override:
                raise ValueError(
                    "Cannot use --config-mode=raw with --override. Raw mode only works with --config."
                )

            # Load from config file directly (bypass Hydra)
            with open(self.config, "r") as f:
                config_dict = yaml.safe_load(f)

            # Create RunConfig from the loaded data
            config = OmegaConf.create(config_dict)

            # Apply task filtering for raw mode (directly modify config)
            if requested_tasks:
                original_tasks = (
                    config.evaluation.tasks
                    if hasattr(config.evaluation, "tasks")
                    else config.evaluation
                )
                filtered_tasks = [
                    task for task in original_tasks if task.name in requested_tasks
                ]
                if not filtered_tasks:
                    available = [task.name for task in original_tasks]
                    raise ValueError(
                        f"No matching tasks found. Requested: {sorted(requested_tasks)}, "
                        f"Available: {available}"
                    )
                config.evaluation.tasks = filtered_tasks
                logger.info(
                    "Filtered tasks (raw mode)",
                    count=len(filtered_tasks),
                    tasks=[t.name for t in filtered_tasks],
                )
        else:
            # Handle --config parameter: split path into config_dir and config_name for Hydra
            if self.config:
                if self.config_name != "default":
                    raise ValueError("Cannot use --config with --config-name")
                if self.config_dir is not None:
                    raise ValueError("Cannot use --config with --config-dir")
                config_path = pathlib.Path(self.config)
                config_dir = str(config_path.parent)
                config_name = str(config_path.stem)
            else:
                config_dir = self.config_dir
                config_name = self.config_name

            # If tasks filter is requested, first load config to get task list,
            # then build task filter overrides and reload with them appended LAST
            if requested_tasks:
                # First load to get available tasks
                config = RunConfig.from_hydra(
                    config_dir=config_dir,
                    config_name=config_name,
                    hydra_overrides=self.override,
                )

                # Build task filter overrides (sets unwanted tasks to null)
                task_filter_overrides = self._build_task_filter_overrides(
                    config, requested_tasks
                )

                # Reload with task filter overrides appended LAST
                all_overrides = list(self.override) + task_filter_overrides
                config = RunConfig.from_hydra(
                    config_dir=config_dir,
                    config_name=config_name,
                    hydra_overrides=all_overrides,
                )

                # Filter out null tasks (tasks that were set to null by overrides)
                original_tasks = (
                    config.evaluation.tasks
                    if hasattr(config.evaluation, "tasks")
                    else config.evaluation
                )
                filtered_tasks = [t for t in original_tasks if t is not None]
                config.evaluation.tasks = filtered_tasks
            else:
                # Load the complete Hydra configuration normally
                config = RunConfig.from_hydra(
                    config_dir=config_dir,
                    config_name=config_name,
                    hydra_overrides=self.override,
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

        # Warn if both config_dir and config_name are provided (and config_name is not default)
        if (
            self.config is None
            and self.config_dir is not None
            and self.config_name != "default"
        ):
            joint_path = pathlib.Path(self.config_dir) / f"{self.config_name}.yaml"
            print(
                yellow(
                    f"Warning: Using --config-dir and --config-name together is deprecated. "
                    f"Please use --config {joint_path} instead."
                )
            )
