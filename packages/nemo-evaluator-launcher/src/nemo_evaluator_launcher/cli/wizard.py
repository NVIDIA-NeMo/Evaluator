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
"""Interactive configuration wizard for NeMo Evaluator Launcher."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from simple_parsing import field

from nemo_evaluator_launcher.cli.ls_deployments import DEPLOYMENTS
from nemo_evaluator_launcher.cli.ls_executors import EXECUTORS
from nemo_evaluator_launcher.common.container_metadata import load_tasks_from_tasks_file


# Default interceptors enabled by wizard
DEFAULT_INTERCEPTORS = ["caching", "response_stats"]

# Available interceptors with descriptions
INTERCEPTORS = {
    "caching": "Cache API responses to avoid redundant calls",
    "response_stats": "Collect statistics about API responses",
    "request_logging": "Log API requests for debugging",
    "system_message": "Inject custom system message",
}

# Reasoning modes
REASONING_MODES = {
    "none": "No reasoning (default)",
    "think": "Enable reasoning with /think prompt",
    "no_think": "Explicitly disable reasoning with /no_think prompt",
    "custom": "Custom reasoning configuration",
}

# Export destinations
EXPORTERS = {
    "local": "Save as JSON/CSV files",
    "mlflow": "Export to MLflow tracking server",
    "wandb": "Export to Weights & Biases",
    "gsheets": "Export to Google Sheets",
}

# NVIDIA green color (#76b900) - using closest ANSI approximation
NVIDIA_GREEN = "#76b900"

# Custom style for questionary prompts - NVIDIA branding with minimal highlighting
WIZARD_STYLE = Style(
    [
        ("qmark", f"fg:{NVIDIA_GREEN} bold"),  # question mark
        ("question", "bold"),  # question text
        ("answer", f"fg:{NVIDIA_GREEN}"),  # submitted answer
        ("pointer", f"fg:{NVIDIA_GREEN} bold"),  # pointer for current item
        ("highlighted", ""),  # no background highlight for pointed item
        ("selected", f"fg:{NVIDIA_GREEN}"),  # selected checkbox items
        ("instruction", "fg:ansibrightblack"),  # instructions (gray)
    ]
)


@dataclass
class Cmd:
    """Interactive configuration wizard for creating evaluation configs.

    Guides you through creating a YAML config file step by step.

    Examples:
        # Run the wizard interactively
        nel wizard

        # Save config only without running
        nel wizard --save-only

        # Save to specific file
        nel wizard --output my-config.yaml
    """

    save_only: bool = field(
        default=False,
        action="store_true",
        alias=["--save-only"],
        metadata={"help": "Save config file without running evaluation"},
    )
    output: str = field(
        default="",
        alias=["--output", "-o"],
        metadata={"help": "Output config file path (skip filename prompt)"},
    )

    def execute(self) -> None:
        """Run the interactive wizard."""
        console = Console()
        config: dict[str, Any] = {}

        # Header
        console.print()
        console.print(
            Panel(
                "[bold]NeMo Evaluator Configuration Wizard[/bold]\n"
                "Let's set up your evaluation step by step.",
                title="Wizard",
            )
        )
        console.print()

        try:
            # Step 1: Executor
            config["executor"] = self._prompt_executor()
            if config["executor"] is None:
                return  # User cancelled

            # Step 2: Executor-specific config
            if config["executor"] == "slurm":
                slurm_config = self._prompt_slurm_config()
                if slurm_config is None:
                    return
                config.update(slurm_config)

            # Step 3: Deployment
            config["deployment"] = self._prompt_deployment()
            if config["deployment"] is None:
                return

            # Step 4: Deployment-specific config
            if config["deployment"] == "none":
                api_config = self._prompt_api_config()
                if api_config is None:
                    return
                config.update(api_config)
            elif config["deployment"] in ("vllm", "sglang"):
                vllm_config = self._prompt_vllm_config()
                if vllm_config is None:
                    return
                config.update(vllm_config)
            elif config["deployment"] == "nim":
                nim_config = self._prompt_nim_config()
                if nim_config is None:
                    return
                config.update(nim_config)

            # Step 5: Tasks (searchable multi-select)
            config["tasks"] = self._prompt_tasks()
            if config["tasks"] is None:
                return

            # Step 6: Adapters
            adapter_config = self._prompt_adapters()
            if adapter_config is None:
                return
            config.update(adapter_config)

            # Step 7: Interceptors (caching + response_stats ON by default)
            config["interceptors"] = self._prompt_interceptors()
            if config["interceptors"] is None:
                return

            # Step 8: Common options
            config["output_dir"] = self._prompt_output_dir()
            if config["output_dir"] is None:
                return

            config["limit_samples"] = self._prompt_limit_samples()

            # Step 9: Export destinations
            config["exporters"] = self._prompt_exporters()
            if config["exporters"] is None:
                return

            # Step 10: Summary
            self._show_summary(config)

            # Step 11: Always save config
            config_path = self.output
            if not config_path:
                config_path = questionary.text(
                    "Save config to:",
                    default="config.yaml",
                ).ask()
                if config_path is None:
                    return

            self._save_config(config, config_path)

            # Step 12: Action
            if self.save_only:
                return

            action = questionary.select(
                "What would you like to do?",
                choices=[
                    questionary.Choice("Run evaluation now", value="run"),
                    questionary.Choice("Exit (config saved)", value="exit"),
                ],
                style=WIZARD_STYLE,
            ).ask()

            if action == "run":
                self._run_evaluation(config_path)

        except KeyboardInterrupt:
            console.print("\n[yellow]Wizard cancelled.[/yellow]")

    def _prompt_executor(self) -> Optional[str]:
        """Prompt for executor selection."""
        choices = [
            questionary.Choice(
                title=f"{name} - {info['description']}"
                + (" (default)" if info["default"] else ""),
                value=name,
            )
            for name, info in EXECUTORS.items()
        ]
        return questionary.select(
            "Where do you want to run the evaluation?",
            choices=choices,
            default="local",
            style=WIZARD_STYLE,
        ).ask()

    def _prompt_deployment(self) -> Optional[str]:
        """Prompt for deployment selection."""
        choices = [
            questionary.Choice(
                title=f"{name} - {info['description']}"
                + (" (default)" if info["default"] else ""),
                value=name,
            )
            for name, info in DEPLOYMENTS.items()
        ]
        return questionary.select(
            "How will the model be served?",
            choices=choices,
            default="none",
            style=WIZARD_STYLE,
        ).ask()

    def _prompt_tasks(self) -> Optional[list[str]]:
        """Searchable multi-select for tasks."""
        tasks, _ = load_tasks_from_tasks_file()
        task_names = sorted(set(t.name for t in tasks))

        result = questionary.checkbox(
            "Select tasks to run:",
            choices=[questionary.Choice(name, value=name) for name in task_names],
            validate=lambda x: len(x) > 0 or "Select at least one task",
            use_search_filter=True,
            instruction="(type to search, space to select, enter to confirm)",
            style=WIZARD_STYLE,
        ).ask()

        return result

    def _prompt_slurm_config(self) -> Optional[dict[str, Any]]:
        """Prompt for SLURM-specific configuration."""
        hostname = questionary.text(
            "SLURM cluster hostname:",
            validate=lambda x: len(x) > 0 or "Hostname is required",
        ).ask()
        if hostname is None:
            return None

        account = questionary.text(
            "SLURM account:",
            validate=lambda x: len(x) > 0 or "Account is required",
        ).ask()
        if account is None:
            return None

        partition = questionary.text(
            "SLURM partition:",
            default="batch",
        ).ask()
        if partition is None:
            return None

        walltime = questionary.text(
            "Walltime (HH:MM:SS):",
            default="01:00:00",
        ).ask()
        if walltime is None:
            return None

        return {
            "slurm_hostname": hostname,
            "slurm_account": account,
            "slurm_partition": partition,
            "slurm_walltime": walltime,
        }

    def _prompt_api_config(self) -> Optional[dict[str, Any]]:
        """Prompt for API endpoint configuration."""
        model = questionary.text(
            "Model ID (e.g., meta/llama-3.2-3b-instruct):",
            validate=lambda x: len(x) > 0 or "Model ID is required",
        ).ask()
        if model is None:
            return None

        url = questionary.text(
            "API endpoint URL:",
            default="https://integrate.api.nvidia.com/v1/chat/completions",
        ).ask()
        if url is None:
            return None

        api_key_env = questionary.text(
            "Environment variable for API key:",
            default="NGC_API_KEY",
        ).ask()
        if api_key_env is None:
            return None

        return {
            "model": model,
            "url": url,
            "api_key_env": api_key_env,
        }

    def _prompt_vllm_config(self) -> Optional[dict[str, Any]]:
        """Prompt for vLLM/SGLang deployment configuration."""
        use_hf = questionary.confirm(
            "Use HuggingFace model? (No = local checkpoint)",
            default=True,
        ).ask()
        if use_hf is None:
            return None

        config: dict[str, Any] = {}
        if use_hf:
            hf_model = questionary.text(
                "HuggingFace model (e.g., meta-llama/Llama-3.2-3B-Instruct):",
                validate=lambda x: len(x) > 0 or "Model is required",
            ).ask()
            if hf_model is None:
                return None
            config["hf_model"] = hf_model
        else:
            checkpoint = questionary.text(
                "Path to model checkpoint:",
                validate=lambda x: len(x) > 0 or "Checkpoint path is required",
            ).ask()
            if checkpoint is None:
                return None
            config["checkpoint"] = checkpoint

        model_name = questionary.text(
            "Served model name:",
            validate=lambda x: len(x) > 0 or "Model name is required",
        ).ask()
        if model_name is None:
            return None
        config["model_name"] = model_name

        configure_tp = questionary.confirm(
            "Configure tensor parallelism?", default=False
        ).ask()
        if configure_tp is None:
            return None

        if configure_tp:
            tp_str = questionary.text(
                "Tensor parallel size:",
                default="1",
            ).ask()
            if tp_str is None:
                return None
            try:
                config["tensor_parallel"] = int(tp_str)
            except ValueError:
                config["tensor_parallel"] = 1

        return config

    def _prompt_nim_config(self) -> Optional[dict[str, Any]]:
        """Prompt for NIM deployment configuration."""
        nim_model = questionary.text(
            "NIM model name:",
            validate=lambda x: len(x) > 0 or "NIM model is required",
        ).ask()
        if nim_model is None:
            return None

        return {"nim_model": nim_model}

    def _prompt_adapters(self) -> Optional[dict[str, Any]]:
        """Configure adapters (reasoning, payload modifier)."""
        result: dict[str, Any] = {}

        configure = questionary.confirm("Configure adapters?", default=False).ask()
        if configure is None:
            return None
        if not configure:
            return result

        # Reasoning mode
        reasoning = questionary.select(
            "Enable reasoning mode?",
            choices=[
                questionary.Choice(desc, value=key)
                for key, desc in REASONING_MODES.items()
            ],
            default="none",
            style=WIZARD_STYLE,
        ).ask()
        if reasoning is None:
            return None

        if reasoning == "think":
            result["reasoning"] = {
                "process_reasoning_traces": True,
                "use_system_prompt": True,
                "custom_system_prompt": "/think",
            }
        elif reasoning == "no_think":
            result["reasoning"] = {
                "use_system_prompt": True,
                "custom_system_prompt": "/no_think",
            }
        elif reasoning == "custom":
            process_traces = questionary.confirm(
                "Process reasoning traces?", default=True
            ).ask()
            if process_traces is None:
                return None

            custom_prompt = questionary.text(
                "Custom system prompt for reasoning:"
            ).ask()
            if custom_prompt is None:
                return None

            result["reasoning"] = {
                "process_reasoning_traces": process_traces,
                "use_system_prompt": True,
                "custom_system_prompt": custom_prompt,
            }

        # Payload modifier
        add_payload = questionary.confirm(
            "Add payload modifiers?", default=False
        ).ask()
        if add_payload is None:
            return None

        if add_payload:
            payload_config = self._prompt_payload_modifier()
            if payload_config is None:
                return None
            if payload_config:
                result["payload_modifier"] = payload_config

        return result

    def _prompt_payload_modifier(self) -> Optional[dict[str, Any]]:
        """Configure payload modifier interceptor."""
        console = Console()
        console.print("[dim]Add custom parameters to API requests[/dim]")
        console.print("[dim]Enter parameter key (empty to finish)[/dim]")

        params: dict[str, Any] = {}
        while True:
            key = questionary.text("Parameter key (empty to finish):").ask()
            if key is None:
                return None
            if not key:
                break

            value = questionary.text(f"Value for '{key}':").ask()
            if value is None:
                return None

            # Try to parse as number/bool
            parsed_value: Union[bool, int, float, str] = value
            try:
                if value.lower() == "true":
                    parsed_value = True
                elif value.lower() == "false":
                    parsed_value = False
                elif "." in value:
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except ValueError:
                parsed_value = value

            params[key] = parsed_value

        return {"params_to_add": params} if params else {}

    def _prompt_interceptors(self) -> Optional[list[str]]:
        """Configure interceptors with caching and response_stats enabled by default."""
        choices = [
            questionary.Choice(
                f"{name} - {desc}",
                value=name,
                checked=(name in DEFAULT_INTERCEPTORS),
            )
            for name, desc in INTERCEPTORS.items()
        ]

        return questionary.checkbox(
            "Select interceptors (caching and response_stats enabled by default):",
            choices=choices,
            instruction="(space to toggle, enter to confirm)",
            style=WIZARD_STYLE,
        ).ask()

    def _prompt_output_dir(self) -> Optional[str]:
        """Prompt for output directory."""
        return questionary.text(
            "Output directory:",
            default="./results",
        ).ask()

    def _prompt_limit_samples(self) -> Optional[int]:
        """Prompt for sample limit (for testing)."""
        limit = questionary.confirm(
            "Limit samples? (for testing only)", default=False
        ).ask()
        if limit is None:
            return None
        if not limit:
            return None

        limit_str = questionary.text("Number of samples:", default="10").ask()
        if limit_str is None:
            return None

        try:
            return int(limit_str)
        except ValueError:
            return 10

    def _prompt_exporters(self) -> Optional[list[dict[str, Any]]]:
        """Select export destinations."""
        choices = [
            questionary.Choice(f"{name} - {desc}", value=name)
            for name, desc in EXPORTERS.items()
        ]

        selected = questionary.checkbox(
            "Export results to (optional):",
            choices=choices,
            instruction="(space to select, enter to confirm)",
            style=WIZARD_STYLE,
        ).ask()
        if selected is None:
            return None

        result: list[dict[str, Any]] = []
        for exporter in selected:
            exp_config: dict[str, Any] = {"dest": exporter}

            if exporter == "local":
                fmt = questionary.select(
                    "Output format:",
                    choices=["json", "csv", "yaml"],
                    default="json",
                    style=WIZARD_STYLE,
                ).ask()
                if fmt is None:
                    return None
                exp_config["format"] = fmt

            elif exporter == "mlflow":
                uri = questionary.text(
                    "MLflow tracking URI:",
                    validate=lambda x: len(x) > 0 or "Tracking URI is required",
                ).ask()
                if uri is None:
                    return None
                exp_config["tracking_uri"] = uri

            elif exporter == "wandb":
                entity = questionary.text("W&B entity:").ask()
                if entity is None:
                    return None
                exp_config["entity"] = entity

                project = questionary.text(
                    "W&B project:", default="nemo-evaluations"
                ).ask()
                if project is None:
                    return None
                exp_config["project"] = project

            elif exporter == "gsheets":
                spreadsheet_id = questionary.text(
                    "Google Sheets spreadsheet ID:",
                    validate=lambda x: len(x) > 0 or "Spreadsheet ID is required",
                ).ask()
                if spreadsheet_id is None:
                    return None
                exp_config["spreadsheet_id"] = spreadsheet_id

            result.append(exp_config)

        return result

    def _show_summary(self, config: dict[str, Any]) -> None:
        """Display configuration summary."""
        console = Console()
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Executor", config["executor"])
        table.add_row("Deployment", config["deployment"])

        if config["deployment"] == "none":
            table.add_row("Model", config.get("model", ""))
            url = config.get("url", "")
            if url and url != "https://integrate.api.nvidia.com/v1/chat/completions":
                table.add_row("URL", url)
        elif config["deployment"] in ("vllm", "sglang"):
            model_source = config.get("checkpoint") or config.get("hf_model", "")
            table.add_row("Checkpoint/HF", model_source)
            table.add_row("Model Name", config.get("model_name", ""))
        elif config["deployment"] == "nim":
            table.add_row("NIM Model", config.get("nim_model", ""))

        if config["executor"] == "slurm":
            table.add_row("SLURM Host", config.get("slurm_hostname", ""))
            table.add_row("SLURM Account", config.get("slurm_account", ""))

        tasks_str = ", ".join(config.get("tasks", [])[:3])
        if len(config.get("tasks", [])) > 3:
            tasks_str += f" (+{len(config['tasks']) - 3} more)"
        table.add_row("Tasks", tasks_str)

        if config.get("reasoning"):
            reasoning_prompt = config["reasoning"].get("custom_system_prompt", "")
            table.add_row("Reasoning", reasoning_prompt or "enabled")

        interceptors = config.get("interceptors", [])
        if interceptors:
            table.add_row("Interceptors", ", ".join(interceptors))

        exporters = config.get("exporters", [])
        if exporters:
            exporter_names = [e["dest"] for e in exporters]
            table.add_row("Export", ", ".join(exporter_names))

        table.add_row("Output", config.get("output_dir", ""))

        if config.get("limit_samples"):
            table.add_row("Limit Samples", str(config["limit_samples"]))

        console.print()
        console.print(Panel(table, title="Configuration Summary"))
        console.print()

    def _build_yaml_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Convert wizard config to proper YAML config structure."""
        executor = config["executor"]
        deployment = config["deployment"]

        yaml_config: dict[str, Any] = {
            "defaults": [
                {
                    "execution": executor
                    if executor == "local"
                    else f"{executor}/default"
                },
                {"deployment": deployment},
                "_self_",
            ],
            "execution": {
                "output_dir": config["output_dir"],
            },
        }

        # Add SLURM config
        if executor == "slurm":
            yaml_config["execution"].update(
                {
                    "hostname": config.get("slurm_hostname"),
                    "account": config.get("slurm_account"),
                    "partition": config.get("slurm_partition", "batch"),
                }
            )
            if config.get("slurm_walltime"):
                yaml_config["execution"]["walltime"] = config["slurm_walltime"]

        # Add target config for API endpoint
        if deployment == "none":
            api_endpoint: dict[str, Any] = {
                "model_id": config["model"],
                "url": config.get(
                    "url", "https://integrate.api.nvidia.com/v1/chat/completions"
                ),
                "api_key_name": config.get("api_key_env", "NGC_API_KEY"),
            }

            # Add adapter config if any
            adapter_config: dict[str, Any] = {}
            if config.get("reasoning"):
                adapter_config.update(config["reasoning"])

            # Add interceptors
            interceptors: list[dict[str, Any]] = []
            if config.get("payload_modifier") and config["payload_modifier"].get(
                "params_to_add"
            ):
                interceptors.append(
                    {
                        "name": "payload_modifier",
                        "config": config["payload_modifier"],
                    }
                )

            for interceptor in config.get("interceptors", []):
                interceptor_cfg: dict[str, Any] = {
                    "name": interceptor,
                    "enabled": True,
                }
                if interceptor == "caching":
                    interceptor_cfg["config"] = {
                        "cache_dir": "/results/cache",
                        "reuse_cached_responses": True,
                    }
                interceptors.append(interceptor_cfg)

            # Always add endpoint interceptor at the end
            interceptors.append({"name": "endpoint", "enabled": True})

            if adapter_config or interceptors:
                api_endpoint["adapter_config"] = adapter_config
                if interceptors:
                    api_endpoint["adapter_config"]["interceptors"] = interceptors

            yaml_config["target"] = {"api_endpoint": api_endpoint}

        # Add vLLM/SGLang deployment config
        elif deployment in ("vllm", "sglang"):
            yaml_config["deployment"] = {}
            if config.get("hf_model"):
                yaml_config["deployment"]["hf_model_handle"] = config["hf_model"]
            elif config.get("checkpoint"):
                yaml_config["deployment"]["checkpoint"] = config["checkpoint"]
            yaml_config["deployment"]["model_name"] = config.get("model_name")
            if config.get("tensor_parallel"):
                yaml_config["deployment"]["tensor_parallel"] = config["tensor_parallel"]

        # Add NIM deployment config
        elif deployment == "nim":
            yaml_config["deployment"] = {
                "nim_model": config.get("nim_model"),
            }

        # Add evaluation tasks
        yaml_config["evaluation"] = {
            "tasks": [{"name": task} for task in config["tasks"]],
        }

        if config.get("limit_samples"):
            yaml_config["evaluation"]["nemo_evaluator_config"] = {
                "config": {"params": {"limit_samples": config["limit_samples"]}}
            }

        return yaml_config

    def _save_config(self, config: dict[str, Any], path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        yaml_config = self._build_yaml_config(config)
        with open(path, "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        console = Console()
        console.print(f"\n[green]Config saved to {path}[/green]")
        console.print(
            f"[dim]  Run with: nemo-evaluator-launcher run --config {path}[/dim]\n"
        )

        # Show export commands if specified
        if config.get("exporters"):
            console.print("[dim]  Export after run:[/dim]")
            for exp in config["exporters"]:
                cmd = f"    nel export <id> --dest {exp['dest']}"
                if exp["dest"] == "local" and exp.get("format"):
                    cmd += f" --format {exp['format']}"
                elif exp["dest"] == "mlflow" and exp.get("tracking_uri"):
                    cmd += f" -o export.mlflow.tracking_uri={exp['tracking_uri']}"
                elif exp["dest"] == "wandb":
                    if exp.get("entity"):
                        cmd += f" -o export.wandb.entity={exp['entity']}"
                    if exp.get("project"):
                        cmd += f" -o export.wandb.project={exp['project']}"
                console.print(f"[dim]{cmd}[/dim]")
            console.print()

    def _run_evaluation(self, config_path: str) -> None:
        """Execute the evaluation with saved config."""
        import subprocess
        import sys

        console = Console()
        console.print("[bold]Starting evaluation...[/bold]\n")

        result = subprocess.run(
            [sys.executable, "-m", "nemo_evaluator_launcher.cli.main", "run", "--config", config_path]
        )
        sys.exit(result.returncode)
