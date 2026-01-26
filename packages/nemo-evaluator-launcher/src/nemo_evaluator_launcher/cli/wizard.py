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

import os
import random
import sys
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
from nemo_evaluator_launcher.cli.wizard_messages import MESSAGES
from nemo_evaluator_launcher.common.container_metadata import load_tasks_from_tasks_file
from nemo_evaluator_launcher.common.settings import (
    EnvVarSettings,
    SlurmProfile,
    get_settings,
)


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
    no_fun: bool = field(
        default=False,
        action="store_true",
        alias=["--no-fun"],
        metadata={"help": "Disable fun messages (or set NEL_WIZARD_NO_FUN=1)"},
    )

    def _supports_emoji(self) -> bool:
        """Check if terminal likely supports emoji."""
        if sys.platform == "win32":
            # Windows Terminal and newer consoles support emoji
            return "WT_SESSION" in os.environ or "TERM_PROGRAM" in os.environ
        # Most modern Unix terminals support emoji
        encoding = sys.stdout.encoding
        return encoding is not None and encoding.lower() in ("utf-8", "utf8")

    def _show_fun_message(
        self, console: Console, stage: str, **kwargs: Any
    ) -> None:
        """Show a fun message if enabled."""
        # Check disable conditions
        if self.no_fun or os.environ.get("NEL_WIZARD_NO_FUN"):
            return

        messages = MESSAGES.get(stage, [])
        if not messages:
            return

        emoji_msg, text_msg = random.choice(messages)
        msg = emoji_msg if self._supports_emoji() else text_msg
        console.print(f"[dim]{msg.format(**kwargs)}[/dim]")

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
            self._show_fun_message(console, "executor", executor=config["executor"])

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
            self._show_fun_message(
                console, "deployment", deployment=config["deployment"]
            )

            # Step 5: Tasks (searchable multi-select)
            config["tasks"] = self._prompt_tasks()
            if config["tasks"] is None:
                return
            self._show_fun_message(console, "tasks", count=len(config["tasks"]))

            # Step 5b: Environment variables
            env_config = self._prompt_env_vars(config["deployment"], config["tasks"])
            if env_config is None:
                return
            config.update(env_config)

            # Step 6: Adapters
            adapter_config = self._prompt_adapters()
            if adapter_config is None:
                return
            config.update(adapter_config)

            # Step 7: Common options
            config["output_dir"] = self._prompt_output_dir()
            if config["output_dir"] is None:
                return

            config["limit_samples"] = self._prompt_limit_samples()

            # Step 8b: Generation parameters
            gen_params = self._prompt_generation_params()
            if gen_params is None:
                return
            config["generation_params"] = gen_params

            # Step 9: Export destinations
            config["exporters"] = self._prompt_exporters()
            if config["exporters"] is None:
                return

            # Step 10: Summary
            self._show_summary(config)

            # Step 11: Always save config
            config_path = self.output
            if not config_path:
                config_path = self._prompt_config_path()
                if config_path is None:
                    return
            elif os.path.exists(config_path):
                # Even with --output flag, warn about overwrite
                console.print(f"[yellow]⚠ File '{config_path}' already exists.[/yellow]")
                overwrite = questionary.confirm(
                    "Overwrite?", default=False, style=WIZARD_STYLE
                ).ask()
                if not overwrite:
                    config_path = self._prompt_config_path(default=config_path)
                    if config_path is None:
                        return

            self._save_config(config, config_path)
            self._show_fun_message(console, "saved")

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
        """Searchable multi-select for tasks with preview panel."""
        from nemo_evaluator_launcher.cli.task_selector import run_task_selector

        tasks, _ = load_tasks_from_tasks_file()

        # Build task metadata dict for preview
        task_metadata = {}
        for task in tasks:
            if task.name not in task_metadata:
                # Extract all defaults from config.params
                params = task.defaults.get("config", {}).get("params", {})
                extra = params.get("extra", {})

                task_metadata[task.name] = {
                    "description": task.description or "No description available",
                    "harness": task.harness,
                    # Store all params for display
                    "params": {
                        "temperature": params.get("temperature"),
                        "top_p": params.get("top_p"),
                        "max_new_tokens": params.get("max_new_tokens"),
                        "parallelism": params.get("parallelism"),
                        "num_fewshot": extra.get("num_fewshot"),
                    },
                    "endpoint_types": task.defaults.get("config", {}).get(
                        "supported_endpoint_types", []
                    ),
                }

        task_names = sorted(task_metadata.keys())

        # Use custom task selector with preview
        result = run_task_selector(task_names, task_metadata, WIZARD_STYLE)

        return result if result else None

    def _prompt_env_vars(
        self, deployment: str, tasks: list[str]
    ) -> Optional[dict[str, Any]]:
        """Prompt for environment variables needed for the configuration."""
        settings = get_settings()
        saved_env = settings.get_env_vars()

        # Suggested deployment env vars (merge predefined with saved)
        deployment_suggestions = ["NGC_API_KEY", "HF_TOKEN", "HF_HOME"]
        if deployment in ("vllm", "sglang"):
            deployment_suggestions.append("VLLM_CACHE_ROOT")
        # Add any saved vars that aren't in suggestions
        for var in saved_env.deployment:
            if var not in deployment_suggestions:
                deployment_suggestions.append(var)

        # Suggested evaluation env vars (merge predefined with saved)
        eval_suggestions = ["HF_TOKEN"]

        # Check if tasks need JUDGE_API_KEY
        judge_tasks = [
            "ns_aime2025",
            "ns_aa_lcr",
            "ns_scicode",
            "ns_livecodebench",
            "simple_evals.AIME_2025",
            "simple_evals.AA_math_test_500",
            "aa_lcr.aa_lcr",
        ]
        if any(t in tasks for t in judge_tasks):
            eval_suggestions.append("JUDGE_API_KEY")
        # Add any saved vars that aren't in suggestions
        for var in saved_env.task:
            if var not in eval_suggestions:
                eval_suggestions.append(var)

        # Ask if user wants to configure env vars
        configure = questionary.confirm(
            "Configure environment variables?", default=False, style=WIZARD_STYLE
        ).ask()
        if configure is None:
            return None

        if not configure:
            return {"deployment_env_vars": [], "evaluation_env_vars": []}

        # Prompt for deployment env vars
        deployment_choices = [
            questionary.Choice(v, checked=v in saved_env.deployment)
            for v in deployment_suggestions
        ]
        deployment_vars = questionary.checkbox(
            "Select environment variables for deployment:",
            choices=deployment_choices,
            instruction="(space to toggle, enter to confirm)",
            style=WIZARD_STYLE,
        ).ask()
        if deployment_vars is None:
            return None

        # Allow adding custom deployment env vars
        add_custom_deploy = questionary.confirm(
            "Add additional deployment env vars?", default=False, style=WIZARD_STYLE
        ).ask()
        if add_custom_deploy is None:
            return None

        if add_custom_deploy:
            custom_vars = self._prompt_custom_env_vars()
            if custom_vars is None:
                return None
            deployment_vars = list(deployment_vars or []) + custom_vars

        # Prompt for evaluation env vars
        eval_choices = [
            questionary.Choice(v, checked=v in saved_env.task) for v in eval_suggestions
        ]
        eval_vars = questionary.checkbox(
            "Select environment variables for evaluation:",
            choices=eval_choices,
            instruction="(space to toggle, enter to confirm)",
            style=WIZARD_STYLE,
        ).ask()
        if eval_vars is None:
            return None

        # Allow adding custom evaluation env vars
        add_custom_eval = questionary.confirm(
            "Add additional evaluation env vars?", default=False, style=WIZARD_STYLE
        ).ask()
        if add_custom_eval is None:
            return None

        if add_custom_eval:
            custom_vars = self._prompt_custom_env_vars()
            if custom_vars is None:
                return None
            eval_vars = list(eval_vars or []) + custom_vars

        # Offer to save preferences
        save_prefs = questionary.confirm(
            "Save env var preferences for future use?", default=True, style=WIZARD_STYLE
        ).ask()
        if save_prefs is None:
            return None

        if save_prefs:
            settings.save_env_vars(
                EnvVarSettings(
                    deployment=deployment_vars or [],
                    task=eval_vars or [],
                )
            )

        return {
            "deployment_env_vars": deployment_vars or [],
            "evaluation_env_vars": eval_vars or [],
        }

    def _prompt_custom_env_vars(self) -> Optional[list[str]]:
        """Prompt for custom environment variable names."""
        console = Console()
        console.print("[dim]Enter env var names (empty to finish)[/dim]")

        custom_vars: list[str] = []
        while True:
            var_name = questionary.text(
                "Env var name (empty to finish):", style=WIZARD_STYLE
            ).ask()
            if var_name is None:
                return None
            if not var_name:
                break
            # Normalize to uppercase
            var_name = var_name.strip().upper()
            if var_name and var_name not in custom_vars:
                custom_vars.append(var_name)

        return custom_vars

    def _prompt_slurm_config(self) -> Optional[dict[str, Any]]:
        """Prompt for SLURM-specific configuration with profile support."""
        settings = get_settings()
        profiles = settings.list_slurm_profiles()
        selected_profile: Optional[SlurmProfile] = None

        # If profiles exist, offer selection
        if profiles:
            choices = [
                questionary.Choice(f"{name} (saved)", value=name) for name in profiles
            ]
            choices.append(questionary.Choice("Enter new configuration", value="_new_"))

            selection = questionary.select(
                "Select SLURM configuration:",
                choices=choices,
                style=WIZARD_STYLE,
            ).ask()
            if selection is None:
                return None

            if selection != "_new_":
                selected_profile = settings.get_slurm_profile(selection)

        # If profile selected, use its values
        if selected_profile:
            return {
                "slurm_hostname": selected_profile.hostname,
                "slurm_username": selected_profile.username or "${oc.env:USER}",
                "slurm_account": selected_profile.account or "",
                "slurm_partition": selected_profile.partition,
                "slurm_walltime": selected_profile.walltime,
                "slurm_gres": selected_profile.gres,
            }

        # Prompt for new configuration
        hostname = questionary.text(
            "SLURM cluster hostname:",
            validate=lambda x: len(x) > 0 or "Hostname is required",
            style=WIZARD_STYLE,
        ).ask()
        if hostname is None:
            return None

        username = questionary.text(
            "SLURM username:",
            default="${oc.env:USER}",
            style=WIZARD_STYLE,
        ).ask()
        if username is None:
            return None

        account = questionary.text(
            "SLURM account:",
            validate=lambda x: len(x) > 0 or "Account is required",
            style=WIZARD_STYLE,
        ).ask()
        if account is None:
            return None

        partition = questionary.text(
            "SLURM partition:",
            default="batch",
            style=WIZARD_STYLE,
        ).ask()
        if partition is None:
            return None

        walltime = questionary.text(
            "Walltime (HH:MM:SS):",
            default="01:00:00",
            style=WIZARD_STYLE,
        ).ask()
        if walltime is None:
            return None

        gres = questionary.text(
            "GRES (e.g., gpu:8, optional):",
            default="",
            style=WIZARD_STYLE,
        ).ask()
        if gres is None:
            return None

        # Offer to save as profile
        save_profile = questionary.confirm(
            "Save this configuration as a profile?",
            default=True,
            style=WIZARD_STYLE,
        ).ask()
        if save_profile is None:
            return None

        if save_profile:
            # Default profile name from hostname (extract short name)
            default_name = hostname.split(".")[0] if "." in hostname else hostname
            profile_name = questionary.text(
                "Profile name:",
                default=default_name,
                style=WIZARD_STYLE,
            ).ask()
            if profile_name:
                profile = SlurmProfile(
                    hostname=hostname,
                    username=username if username != "${oc.env:USER}" else None,
                    account=account,
                    partition=partition,
                    walltime=walltime,
                    gres=gres if gres else None,
                )
                settings.save_slurm_profile(profile_name, profile)
                Console().print(f"[green]Profile '{profile_name}' saved.[/green]")

        return {
            "slurm_hostname": hostname,
            "slurm_username": username,
            "slurm_account": account,
            "slurm_partition": partition,
            "slurm_walltime": walltime,
            "slurm_gres": gres if gres else None,
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

        # Ask for container image
        image = questionary.text(
            "Container image (e.g., vllm/vllm-openai:v0.10.2):",
            default="vllm/vllm-openai:v0.10.2",
            style=WIZARD_STYLE,
        ).ask()
        if image is None:
            return None
        config["image"] = image

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
        """Configure adapters (logging, reasoning, payload modifier)."""
        result: dict[str, Any] = {}

        configure = questionary.confirm("Configure adapters?", default=False).ask()
        if configure is None:
            return None
        if not configure:
            return result

        # Logging options
        enable_logging = questionary.confirm(
            "Enable request/response logging?", default=True, style=WIZARD_STYLE
        ).ask()
        if enable_logging is None:
            return None

        if enable_logging:
            logging_config: dict[str, Any] = {
                "use_request_logging": True,
                "max_logged_requests": 10,
                "use_response_logging": True,
                "max_logged_responses": 10,
                "log_failed_requests": True,
            }

            # Ask for custom limits
            customize_logging = questionary.confirm(
                "Customize logging limits?", default=False, style=WIZARD_STYLE
            ).ask()
            if customize_logging is None:
                return None

            if customize_logging:
                max_req = questionary.text(
                    "Max logged requests:", default="10", style=WIZARD_STYLE
                ).ask()
                if max_req is None:
                    return None
                try:
                    logging_config["max_logged_requests"] = int(max_req)
                except ValueError:
                    pass

                max_resp = questionary.text(
                    "Max logged responses:", default="10", style=WIZARD_STYLE
                ).ask()
                if max_resp is None:
                    return None
                try:
                    logging_config["max_logged_responses"] = int(max_resp)
                except ValueError:
                    pass

            result["logging"] = logging_config

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
                "use_reasoning": True,
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>",
                "use_system_prompt": True,
                "custom_system_prompt": "/think",
            }
        elif reasoning == "no_think":
            result["reasoning"] = {
                "use_system_prompt": True,
                "custom_system_prompt": "/no_think",
            }
        elif reasoning == "custom":
            use_reasoning = questionary.confirm(
                "Process reasoning traces?", default=True, style=WIZARD_STYLE
            ).ask()
            if use_reasoning is None:
                return None

            reasoning_config: dict[str, Any] = {"use_reasoning": use_reasoning}

            if use_reasoning:
                start_token = questionary.text(
                    "Reasoning start token:",
                    default="<think>",
                    style=WIZARD_STYLE,
                ).ask()
                if start_token is None:
                    return None
                reasoning_config["start_reasoning_token"] = start_token

                end_token = questionary.text(
                    "Reasoning end token:",
                    default="</think>",
                    style=WIZARD_STYLE,
                ).ask()
                if end_token is None:
                    return None
                reasoning_config["end_reasoning_token"] = end_token

            custom_prompt = questionary.text(
                "Custom system prompt for reasoning (optional):",
                style=WIZARD_STYLE,
            ).ask()
            if custom_prompt is None:
                return None

            if custom_prompt:
                reasoning_config["use_system_prompt"] = True
                reasoning_config["custom_system_prompt"] = custom_prompt

            result["reasoning"] = reasoning_config

        # Payload modifier
        add_payload = questionary.confirm(
            "Add payload modifiers?", default=False, style=WIZARD_STYLE
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
        """Configure payload modifier (params_to_add, params_to_remove)."""
        console = Console()
        result: dict[str, Any] = {}

        # params_to_add
        console.print("[dim]Add custom parameters to API requests[/dim]")
        console.print("[dim]Enter parameter key (empty to finish)[/dim]")
        console.print(
            "[dim]Use dot notation for nested keys (e.g., chat_template_kwargs.thinking)[/dim]"
        )

        params: dict[str, Any] = {}
        while True:
            key = questionary.text(
                "Parameter key (empty to finish):", style=WIZARD_STYLE
            ).ask()
            if key is None:
                return None
            if not key:
                break

            value = questionary.text(f"Value for '{key}':", style=WIZARD_STYLE).ask()
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

            # Handle nested keys (e.g., "chat_template_kwargs.thinking" -> nested dict)
            if "." in key:
                parts = key.split(".")
                current = params
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = parsed_value
            else:
                params[key] = parsed_value

        if params:
            result["params_to_add"] = params

        # params_to_remove
        console.print()
        console.print("[dim]Parameters to remove from API requests[/dim]")
        console.print("[dim]Enter parameter name (empty to finish)[/dim]")

        params_to_remove: list[str] = []
        while True:
            param = questionary.text(
                "Parameter to remove (empty to finish):", style=WIZARD_STYLE
            ).ask()
            if param is None:
                return None
            if not param:
                break
            params_to_remove.append(param)

        if params_to_remove:
            result["params_to_remove"] = params_to_remove

        return result

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

    def _prompt_generation_params(self) -> Optional[dict[str, Any]]:
        """Prompt for global generation parameters (temperature, top_p, max_new_tokens)."""
        # Ask if user wants to set global params
        set_global = questionary.confirm(
            "Set global generation parameters? (applies to all tasks)",
            default=False,
            style=WIZARD_STYLE,
        ).ask()

        if set_global is None:
            return None

        result: dict[str, Any] = {}

        if set_global:
            # Temperature
            temp_str = questionary.text(
                "Temperature (leave empty for task defaults):",
                default="",
                style=WIZARD_STYLE,
            ).ask()
            if temp_str is None:
                return None
            if temp_str:
                try:
                    result["temperature"] = float(temp_str)
                except ValueError:
                    pass

            # Top-p
            top_p_str = questionary.text(
                "Top-p (leave empty for task defaults):",
                default="",
                style=WIZARD_STYLE,
            ).ask()
            if top_p_str is None:
                return None
            if top_p_str:
                try:
                    result["top_p"] = float(top_p_str)
                except ValueError:
                    pass

            # Max new tokens
            max_tokens_str = questionary.text(
                "Max new tokens (leave empty for task defaults):",
                default="",
                style=WIZARD_STYLE,
            ).ask()
            if max_tokens_str is None:
                return None
            if max_tokens_str:
                try:
                    result["max_new_tokens"] = int(max_tokens_str)
                except ValueError:
                    pass

        return result

    def _prompt_config_path(self, default: str = "config.yaml") -> Optional[str]:
        """Prompt for config path with overwrite protection."""
        console = Console()
        while True:
            path = questionary.text(
                "Save config to:",
                default=default,
                style=WIZARD_STYLE,
            ).ask()
            if path is None:
                return None

            # Ensure .yaml extension
            if not path.endswith(".yaml") and not path.endswith(".yml"):
                path += ".yaml"

            if not os.path.exists(path):
                return path

            console.print(f"[yellow]⚠ File '{path}' already exists.[/yellow]")
            overwrite = questionary.confirm(
                "Overwrite?", default=False, style=WIZARD_STYLE
            ).ask()
            if overwrite is None:
                return None
            if overwrite:
                return path
            # Loop continues to ask for new filename
            default = path  # Use the same name as default for retry

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
            if config.get("image"):
                table.add_row("Image", config["image"])
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
            tokens = ""
            if config["reasoning"].get("use_reasoning"):
                start = config["reasoning"].get("start_reasoning_token", "<think>")
                end = config["reasoning"].get("end_reasoning_token", "</think>")
                tokens = f" [{start}...{end}]"
            table.add_row("Reasoning", (reasoning_prompt or "enabled") + tokens)

        # Show logging if enabled
        if config.get("logging"):
            logging_items = []
            if config["logging"].get("use_request_logging"):
                logging_items.append("requests")
            if config["logging"].get("use_response_logging"):
                logging_items.append("responses")
            if logging_items:
                table.add_row("Logging", ", ".join(logging_items))

        # Show payload modifier
        if config.get("payload_modifier"):
            if config["payload_modifier"].get("params_to_add"):
                params = ", ".join(config["payload_modifier"]["params_to_add"].keys())
                table.add_row("Params to Add", params)
            if config["payload_modifier"].get("params_to_remove"):
                params = ", ".join(config["payload_modifier"]["params_to_remove"])
                table.add_row("Params to Remove", params)

        # Show env vars if configured
        if config.get("deployment_env_vars"):
            table.add_row("Deploy Env Vars", ", ".join(config["deployment_env_vars"]))
        if config.get("evaluation_env_vars"):
            table.add_row("Eval Env Vars", ", ".join(config["evaluation_env_vars"]))

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
                    "username": config.get("slurm_username", "${oc.env:USER}"),
                    "account": config.get("slurm_account"),
                    "partition": config.get("slurm_partition", "batch"),
                }
            )
            if config.get("slurm_walltime"):
                yaml_config["execution"]["walltime"] = config["slurm_walltime"]
            if config.get("slurm_gres"):
                yaml_config["execution"]["gres"] = config["slurm_gres"]

        # Add target config for API endpoint (deployment == "none")
        if deployment == "none":
            api_endpoint: dict[str, Any] = {
                "model_id": config["model"],
                "url": config.get(
                    "url", "https://integrate.api.nvidia.com/v1/chat/completions"
                ),
                "api_key_name": config.get("api_key_env", "NGC_API_KEY"),
            }
            yaml_config["target"] = {"api_endpoint": api_endpoint}

        # Add vLLM/SGLang deployment config
        if deployment in ("vllm", "sglang"):
            yaml_config["deployment"] = {}
            if config.get("hf_model"):
                yaml_config["deployment"]["hf_model_handle"] = config["hf_model"]
            elif config.get("checkpoint"):
                yaml_config["deployment"]["checkpoint"] = config["checkpoint"]
            yaml_config["deployment"]["model_name"] = config.get("model_name")
            if config.get("tensor_parallel"):
                yaml_config["deployment"]["tensor_parallel"] = config["tensor_parallel"]
            if config.get("image"):
                yaml_config["deployment"]["image"] = config["image"]

        # Add NIM deployment config
        elif deployment == "nim":
            yaml_config["deployment"] = {
                "nim_model": config.get("nim_model"),
            }

        # Add evaluation tasks
        yaml_config["evaluation"] = {
            "tasks": [{"name": task} for task in config["tasks"]],
        }

        # Add global generation params and limit_samples
        gen_params = config.get("generation_params", {})

        if gen_params or config.get("limit_samples"):
            nemo_config = yaml_config["evaluation"].setdefault(
                "nemo_evaluator_config", {}
            )
            params = nemo_config.setdefault("config", {}).setdefault("params", {})

            if config.get("limit_samples"):
                params["limit_samples"] = config["limit_samples"]

            params.update(gen_params)

        # Build adapter_config for ALL deployments (flat format)
        adapter_config: dict[str, Any] = {}

        # Logging settings
        if config.get("logging"):
            adapter_config.update(config["logging"])

        # Reasoning settings
        if config.get("reasoning"):
            adapter_config.update(config["reasoning"])

        # params_to_add (from payload_modifier)
        if config.get("payload_modifier", {}).get("params_to_add"):
            adapter_config["params_to_add"] = config["payload_modifier"]["params_to_add"]

        # params_to_remove (from payload_modifier)
        if config.get("payload_modifier", {}).get("params_to_remove"):
            adapter_config["params_to_remove"] = config["payload_modifier"][
                "params_to_remove"
            ]

        # Add adapter_config to evaluation.nemo_evaluator_config.target.api_endpoint
        if adapter_config:
            nemo_config = yaml_config["evaluation"].setdefault(
                "nemo_evaluator_config", {}
            )
            target = nemo_config.setdefault("target", {})
            api_endpoint = target.setdefault("api_endpoint", {})
            api_endpoint["adapter_config"] = adapter_config

        # Add env vars to execution.env_vars (deployment and evaluation)
        if config.get("deployment_env_vars") or config.get("evaluation_env_vars"):
            env_vars = yaml_config["execution"].setdefault("env_vars", {})

            if config.get("deployment_env_vars"):
                env_vars["deployment"] = {var: f"${var}" for var in config["deployment_env_vars"]}

            if config.get("evaluation_env_vars"):
                env_vars["evaluation"] = {var: f"${var}" for var in config["evaluation_env_vars"]}

        return yaml_config

    def _save_config(self, config: dict[str, Any], path: str) -> None:
        """Save configuration to YAML file with readable formatting."""
        import yaml

        yaml_config = self._build_yaml_config(config)

        # Dump each top-level section separately with empty lines between them
        sections = []
        for key, value in yaml_config.items():
            section = yaml.dump(
                {key: value}, default_flow_style=False, sort_keys=False
            )
            sections.append(section.rstrip())

        with open(path, "w") as f:
            f.write("\n\n".join(sections) + "\n")

        console = Console()
        console.print(f"\n[green]Config saved to {path}[/green]")
        console.print(
            f"[dim]  Run with: nemo-evaluator-launcher run --config {path}[/dim]"
        )

        # Show hint about per-task overrides if global generation params were set
        if config.get("generation_params"):
            console.print(
                "[dim]  Tip: Edit YAML to add task-specific overrides under each task's nemo_evaluator_config[/dim]"
            )
        console.print()

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
