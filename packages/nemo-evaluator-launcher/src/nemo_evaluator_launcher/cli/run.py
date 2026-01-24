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
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal

from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.printing_utils import (
    bold,
    cyan,
    green,
    magenta,
    red,
)


@dataclass
class Cmd:
    """Run LLM evaluation.

    Supports both config files and direct CLI flags for a unified experience.

    Examples:
        # Quick API evaluation (no config file needed)
        nel run --model meta/llama-3.2-3b-instruct --task ifeval

        # Deploy with vLLM locally
        nel run --deployment vllm --checkpoint /path/to/model --model-name my-model --task ifeval

        # Deploy on SLURM cluster
        nel run --deployment vllm --executor slurm \\
            --slurm-hostname cluster.example.com --slurm-account my-account \\
            --output-dir /shared/results --checkpoint /path/to/model \\
            --model-name my-model --task ifeval

        # Use config file (existing behavior)
        nel run --config my_eval.yaml

        # Config file with flag overrides (flags take precedence)
        nel run --config base.yaml --model new-model --task gsm8k
    """

    # === Config file (existing behavior) ===
    config: str | None = field(
        default=None,
        alias=["--config"],
        metadata={
            "help": "Path to YAML config file. Uses Hydra by default. Use --config-mode=raw to bypass Hydra."
        },
    )
    config_mode: Literal["hydra", "raw"] = field(
        default="hydra",
        alias=["--config-mode"],
        metadata={
            "help": "Config loading mode: 'hydra' (default) or 'raw' (bypass Hydra)."
        },
    )
    override: list[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-o"],
        metadata={
            "help": "Hydra override: -o key=value (repeatable).",
        },
    )

    # === Target/API flags (for deployment=none) ===
    model: str | None = field(
        default=None,
        alias=["--model", "-m"],
        metadata={"help": "Model ID (e.g., meta/llama-3.2-3b-instruct)."},
    )
    url: str | None = field(
        default=None,
        alias=["--url", "-u"],
        metadata={"help": "API endpoint URL. Default: NVIDIA API Catalog."},
    )
    api_key_env: str | None = field(
        default=None,
        alias=["--api-key-env"],
        metadata={"help": "Environment variable name for API key. Default: NGC_API_KEY."},
    )

    # === Tasks ===
    task: list[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-t", "--task"],
        metadata={"help": "Task name (repeatable: -t ifeval -t gsm8k)."},
    )

    # === Executor selection ===
    executor: str | None = field(
        default=None,
        alias=["--executor", "-e"],
        metadata={"help": "Executor: local (default), slurm, lepton."},
    )

    # === Common execution flags ===
    output_dir: str | None = field(
        default=None,
        alias=["--output-dir"],
        metadata={"help": "Output directory for results."},
    )

    # === SLURM-specific flags ===
    slurm_hostname: str | None = field(
        default=None,
        alias=["--slurm-hostname"],
        metadata={"help": "SLURM cluster hostname (required for executor=slurm)."},
    )
    slurm_account: str | None = field(
        default=None,
        alias=["--slurm-account"],
        metadata={"help": "SLURM account (required for executor=slurm)."},
    )
    slurm_partition: str | None = field(
        default=None,
        alias=["--slurm-partition"],
        metadata={"help": "SLURM partition. Default: batch."},
    )
    slurm_walltime: str | None = field(
        default=None,
        alias=["--slurm-walltime"],
        metadata={"help": "SLURM walltime (HH:MM:SS). Default: 01:00:00."},
    )

    # === Deployment selection ===
    deployment: str | None = field(
        default=None,
        alias=["--deployment", "-d"],
        metadata={"help": "Deployment: none (default), vllm, nim, sglang."},
    )

    # === vLLM/SGLang deployment flags ===
    checkpoint: str | None = field(
        default=None,
        alias=["--checkpoint", "-c"],
        metadata={"help": "Path to model checkpoint (for vllm/sglang deployment)."},
    )
    model_name: str | None = field(
        default=None,
        alias=["--model-name"],
        metadata={"help": "Served model name (for vllm/sglang deployment)."},
    )
    tensor_parallel: int | None = field(
        default=None,
        alias=["--tp", "--tensor-parallel"],
        metadata={"help": "Tensor parallel size. Default: 1."},
    )
    data_parallel: int | None = field(
        default=None,
        alias=["--dp", "--data-parallel"],
        metadata={"help": "Data parallel size. Default: 1."},
    )
    hf_model_handle: str | None = field(
        default=None,
        alias=["--hf-model"],
        metadata={"help": "HuggingFace model handle (alternative to --checkpoint)."},
    )

    # === NIM deployment flags ===
    nim_model: str | None = field(
        default=None,
        alias=["--nim-model"],
        metadata={"help": "NIM model name (for nim deployment)."},
    )

    # === Evaluation flags ===
    limit_samples: int | None = field(
        default=None,
        alias=["--limit-samples", "-n"],
        metadata={"help": "Limit samples (for testing only - not for benchmarking!)."},
    )

    # === Utility flags ===
    dry_run: bool = field(
        default=False,
        alias=["--dry-run"],
        metadata={"help": "Print config without running."},
    )
    save_config: str | None = field(
        default=None,
        alias=["--save-config"],
        metadata={"help": "Save generated config to file without running."},
    )
    check: bool = field(
        default=False,
        alias=["--check"],
        metadata={"help": "Run deep validation (SSH, Docker, API connectivity)."},
    )
    config_output: str | None = field(
        default=None,
        alias=["--config-output"],
        metadata={
            "help": "Directory to save the complete run config. Defaults to ~/.nemo-evaluator/run_configs/"
        },
    )
    # Keep for backward compatibility (deprecated, use --task instead)
    tasks: list[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        metadata={"help": "Deprecated: use --task instead."},
    )

    def _parse_task_args(self) -> list[str]:
        """Parse task arguments from both --task and deprecated --tasks flags."""
        result = []
        for task_list in [self.task, self.tasks]:
            for task_arg in task_list:
                if not task_arg:
                    continue
                task_name = task_arg.strip()
                if task_name and task_name not in result:
                    result.append(task_name)
        return result

    def _has_any_direct_flags(self) -> bool:
        """Check if any direct CLI flags were provided (beyond config file)."""
        return any([
            self.model is not None,
            self.url is not None,
            self.api_key_env is not None,
            self.executor is not None,
            self.output_dir is not None,
            self.slurm_hostname is not None,
            self.slurm_account is not None,
            self.slurm_partition is not None,
            self.slurm_walltime is not None,
            self.deployment is not None,
            self.checkpoint is not None,
            self.model_name is not None,
            self.tensor_parallel is not None,
            self.data_parallel is not None,
            self.hf_model_handle is not None,
            self.nim_model is not None,
            self.limit_samples is not None,
            len(self._parse_task_args()) > 0,
        ])

    def _build_config_from_flags(self) -> dict[str, Any]:
        """Build a Hydra-compatible config dictionary from CLI flags."""
        # Determine executor and deployment types
        executor_type = self.executor or "local"
        deployment_type = self.deployment or "none"

        # Build defaults list for Hydra
        if executor_type == "local":
            execution_default = "local"
        else:
            execution_default = f"{executor_type}/default"

        config: dict[str, Any] = {
            "defaults": [
                {"execution": execution_default},
                {"deployment": deployment_type},
                "_self_",
            ],
            "execution": {},
            "deployment": {},
            "evaluation": {},
        }

        # Add execution config
        if self.output_dir:
            config["execution"]["output_dir"] = self.output_dir

        # Add SLURM-specific config
        if executor_type == "slurm":
            if self.slurm_hostname:
                config["execution"]["hostname"] = self.slurm_hostname
            if self.slurm_account:
                config["execution"]["account"] = self.slurm_account
            if self.slurm_partition:
                config["execution"]["partition"] = self.slurm_partition
            if self.slurm_walltime:
                config["execution"]["walltime"] = self.slurm_walltime

        # Add deployment config
        if deployment_type == "none":
            # API endpoint config
            api_endpoint: dict[str, Any] = {}
            if self.model:
                api_endpoint["model_id"] = self.model
            if self.url:
                api_endpoint["url"] = self.url
            else:
                # Default to NVIDIA API Catalog
                api_endpoint["url"] = "https://integrate.api.nvidia.com/v1/chat/completions"
            if self.api_key_env:
                api_endpoint["api_key_name"] = self.api_key_env
            else:
                api_endpoint["api_key_name"] = "NGC_API_KEY"

            if api_endpoint:
                config["target"] = {"api_endpoint": api_endpoint}
        elif deployment_type in ("vllm", "sglang"):
            if self.checkpoint:
                config["deployment"]["checkpoint_path"] = self.checkpoint
            if self.hf_model_handle:
                config["deployment"]["hf_model_handle"] = self.hf_model_handle
            if self.model_name:
                config["deployment"]["served_model_name"] = self.model_name
            if self.tensor_parallel is not None:
                config["deployment"]["tensor_parallel_size"] = self.tensor_parallel
            if self.data_parallel is not None:
                config["deployment"]["data_parallel_size"] = self.data_parallel
        elif deployment_type == "nim":
            if self.nim_model:
                config["deployment"]["model_name"] = self.nim_model

        # Add tasks
        task_names = self._parse_task_args()
        if task_names:
            config["evaluation"]["tasks"] = [{"name": t} for t in task_names]

        # Add limit_samples if specified
        if self.limit_samples is not None:
            config["evaluation"]["nemo_evaluator_config"] = {
                "config": {"params": {"limit_samples": self.limit_samples}}
            }

        return config

    def _apply_flag_overrides(self, config: Any) -> Any:
        """Apply CLI flag overrides to an existing config."""
        from omegaconf import OmegaConf

        overrides: dict[str, Any] = {}

        # Target/API overrides
        if self.model is not None:
            overrides.setdefault("target", {}).setdefault("api_endpoint", {})["model_id"] = self.model
        if self.url is not None:
            overrides.setdefault("target", {}).setdefault("api_endpoint", {})["url"] = self.url
        if self.api_key_env is not None:
            overrides.setdefault("target", {}).setdefault("api_endpoint", {})["api_key_name"] = self.api_key_env

        # Execution overrides
        if self.output_dir is not None:
            overrides.setdefault("execution", {})["output_dir"] = self.output_dir
        if self.slurm_hostname is not None:
            overrides.setdefault("execution", {})["hostname"] = self.slurm_hostname
        if self.slurm_account is not None:
            overrides.setdefault("execution", {})["account"] = self.slurm_account
        if self.slurm_partition is not None:
            overrides.setdefault("execution", {})["partition"] = self.slurm_partition
        if self.slurm_walltime is not None:
            overrides.setdefault("execution", {})["walltime"] = self.slurm_walltime

        # Deployment overrides
        if self.checkpoint is not None:
            overrides.setdefault("deployment", {})["checkpoint_path"] = self.checkpoint
        if self.hf_model_handle is not None:
            overrides.setdefault("deployment", {})["hf_model_handle"] = self.hf_model_handle
        if self.model_name is not None:
            overrides.setdefault("deployment", {})["served_model_name"] = self.model_name
        if self.tensor_parallel is not None:
            overrides.setdefault("deployment", {})["tensor_parallel_size"] = self.tensor_parallel
        if self.data_parallel is not None:
            overrides.setdefault("deployment", {})["data_parallel_size"] = self.data_parallel
        if self.nim_model is not None:
            overrides.setdefault("deployment", {})["model_name"] = self.nim_model

        # Task overrides - replace tasks if specified
        task_names = self._parse_task_args()
        if task_names:
            overrides.setdefault("evaluation", {})["tasks"] = [{"name": t} for t in task_names]

        # Limit samples override
        if self.limit_samples is not None:
            overrides.setdefault("evaluation", {}).setdefault("nemo_evaluator_config", {}).setdefault("config", {}).setdefault("params", {})["limit_samples"] = self.limit_samples

        # Merge overrides if any
        if overrides:
            config = OmegaConf.merge(config, overrides)

        return config

    def _validate_config(self, config: Any) -> bool:
        """Run pre-submission validation on the config.

        Returns True if validation passes, False otherwise.
        """
        from nemo_evaluator_launcher.common.validation import validate_config

        result = validate_config(config)

        if not result.valid:
            result.print_errors()
            return False

        # Print warnings if any
        if result.warnings:
            result.print_errors()  # This prints both errors and warnings

        return True

    def _run_deep_checks(self, config: Any) -> bool:
        """Run optional deep validation checks (SSH, Docker, API connectivity).

        Returns True if all checks pass, False otherwise.
        """
        print()
        print(bold("Running deep checks..."))
        print()

        all_passed = True
        executor_type = config.execution.get("type", "local") if hasattr(config, "execution") else "local"

        # Check environment variables
        if hasattr(config, "target") and hasattr(config.target, "api_endpoint"):
            api_key_env = config.target.api_endpoint.get("api_key_name", "NGC_API_KEY")
            if api_key_env:
                import os
                if os.environ.get(api_key_env):
                    print(f"  {green('OK')} Environment variable {api_key_env} is set")
                else:
                    print(f"  {cyan('!')} Environment variable {api_key_env} is not set")

        # Executor-specific checks
        if executor_type == "local":
            # Check Docker
            try:
                import subprocess
                result = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print(f"  {green('OK')} Docker is running")
                else:
                    print(f"  {red('x')} Docker is not running or not accessible")
                    all_passed = False
            except FileNotFoundError:
                print(f"  {red('x')} Docker is not installed")
                all_passed = False
            except subprocess.TimeoutExpired:
                print(f"  {cyan('!')} Docker check timed out")

        elif executor_type == "slurm":
            # Check SSH connectivity
            hostname = config.execution.get("hostname")
            if hostname:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", hostname, "echo", "ok"],
                        capture_output=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        print(f"  {green('OK')} SSH connection to {hostname} successful")
                    else:
                        print(f"  {red('x')} SSH connection to {hostname} failed")
                        all_passed = False
                except subprocess.TimeoutExpired:
                    print(f"  {red('x')} SSH connection to {hostname} timed out")
                    all_passed = False
                except Exception as e:
                    print(f"  {cyan('!')} SSH check failed: {e}")

        print()
        if all_passed:
            print(f"{green('All checks passed')}")
        else:
            print(f"{red('Some checks failed')}")
        print()

        return all_passed

    def execute(self) -> None:
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

        # Determine if we're using direct flags or config file
        has_direct_flags = self._has_any_direct_flags()
        has_config = self.config is not None

        # Load configuration
        if self.config_mode == "raw" and self.config:
            # Validate that raw config loading is not used with overrides
            if self.override:
                raise ValueError(
                    "Cannot use --config-mode=raw with -o overrides. Raw mode only works with --config."
                )
            if has_direct_flags:
                raise ValueError(
                    "Cannot use --config-mode=raw with direct flags. Use --config-mode=hydra (default) to combine config with flags."
                )

            # Load from config file directly (bypass Hydra)
            with open(self.config, "r") as f:
                config_dict = yaml.safe_load(f)

            # Create config from the loaded data
            config = OmegaConf.create(config_dict)
        elif has_config:
            # Load config file with Hydra
            config = RunConfig.from_hydra(
                config=self.config,
                hydra_overrides=self.override,
            )
            # Apply direct flag overrides on top
            if has_direct_flags:
                config = self._apply_flag_overrides(config)
        elif has_direct_flags:
            # Build config entirely from CLI flags
            config_dict = self._build_config_from_flags()
            # Use Hydra to process the config (handles defaults, etc.)
            config = RunConfig.from_hydra(
                config=None,  # Use internal default
                dict_overrides=config_dict,
                hydra_overrides=self.override,
            )
        else:
            # No config file and no direct flags - use Hydra default config
            # This maintains backward compatibility with existing code that calls
            # RunConfig.from_hydra() without arguments
            config = RunConfig.from_hydra(
                config=None,
                hydra_overrides=self.override,
            )

        # Run validation
        if not self._validate_config(config):
            sys.exit(1)

        # Run optional deep checks
        if self.check:
            if not self._run_deep_checks(config):
                print(red("Deep checks failed. Use --dry-run to see the config anyway."))
                if not self.dry_run:
                    sys.exit(1)

        # Save config without running if --save-config is specified
        if self.save_config:
            save_path = pathlib.Path(self.save_config)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            config_dict = OmegaConf.to_container(config, resolve=True)
            config_yaml = yaml.dump(
                config_dict, default_flow_style=False, sort_keys=False, indent=2
            )

            with open(save_path, "w") as f:
                f.write("# Generated by nemo-evaluator-launcher\n")
                f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
                f.write("#\n")
                f.write("# To run this configuration:\n")
                f.write(f"#   nel run --config {save_path}\n")
                f.write("#\n")
                f.write(config_yaml)

            print(f"Config saved to: {save_path}")
            return

        # Handle task filtering for backward compatibility with -t flag on config files
        requested_tasks = self._parse_task_args()
        if has_config and requested_tasks and not has_direct_flags:
            # Only filter if using config file and -t was used for filtering (not building)
            config = filter_tasks(config, requested_tasks)
            logger.info(
                "Running filtered tasks",
                count=len(config.evaluation.tasks),
                tasks=[t.name for t in config.evaluation.tasks],
            )

        try:
            invocation_id = run_eval(config, self.dry_run)
        except Exception as e:
            print(red(f"x Job submission failed, see logs | Error: {e}"))
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
                        f"OK Job submission successful | Invocation ID: {invocation_id}"
                    )
                )
            )

        # Done.
