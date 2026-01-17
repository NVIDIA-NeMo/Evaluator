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

"""Interactive quickstart to generate a runnable YAML config.

The generated YAML references internal Hydra configs via top-level defaults
and includes inline comments with typical defaults or alternatives.
"""

import pathlib
import time
from dataclasses import dataclass
from importlib import resources as importlib_resources
from typing import Any

from jinja2 import BaseLoader, Environment
from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.printing_utils import bold, cyan, red, yellow


@dataclass
class Cmd:
    """Quickstart command parameters."""

    output: str | None = field(
        default=None,
        alias=["-o", "--output"],
        metadata={
            "help": "Directory to write the generated YAML (default: ./examples or .)"
        },
    )
    config_name: str | None = field(
        default=None,
        alias=["-n", "--config-name"],
        metadata={"help": "Filename (without .yaml). Default derived from choices."},
    )
    minimal: bool = field(
        default=False,
        alias=["--minimal"],
        metadata={"help": "Generate with minimal prompts (accept sensible defaults)."},
    )

    def execute(self) -> None:
        print(bold(cyan("NeMo Evaluator Launcher Quickstart")))
        print("This wizard will create a runnable YAML config.")

        exec_choices = self._list_execution_defaults() or [
            "local",
            "slurm/default",
            "lepton/default",
        ]
        dep_choices = self._list_deployment_types() or [
            "none",
            "vllm",
            "nim",
            "sglang",
            "trtllm",
            "generic",
        ]

        flow = self._build_flow(exec_choices, dep_choices)
        answers = self._run_flow(flow)

        yaml_text = self._render_yaml(answers)

        # Ask where to save the file and how to name it
        default_output_dir = self._determine_output_dir(self.output)
        save_dir_str = self._prompt_str(
            "Directory to save config", default=str(default_output_dir)
        )
        output_dir = pathlib.Path(save_dir_str).expanduser().resolve()

        default_cfg_name = self.config_name or self._derive_config_name(
            answers["execution_default"], answers["deployment_default"], answers
        )
        cfg_name = self._prompt_str(
            "Config filename (without .yaml)", default=default_cfg_name
        )
        cfg_path = output_dir / f"{cfg_name}.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml_text, encoding="utf-8")

        print(bold(cyan("Generated config:")), f"\n  {cfg_path}\n")
        print(
            bold(cyan("To dry-run: "))
            + f"nemo-evaluator-launcher run --run-config-file {cfg_path} --dry-run"
        )
        print(
            bold(cyan("To run: "))
            + f"nemo-evaluator-launcher run --run-config-file {cfg_path}"
        )
        logger.info("Quickstart config written", path=str(cfg_path))

    # ---------------------- helpers ----------------------
    def _prompt_str(
        self, prompt: str, default: str | None = None, required: bool = False
    ) -> str:
        suffix = f" [{default}]" if default not in (None, "") else ""
        while True:
            val = input(f"{prompt}{suffix}: ").strip()
            if not val and default is not None:
                return default
            if val:
                return val
            if required:
                print(red("This field is required."))

    def _prompt_int(self, prompt: str, default: int | None = None) -> int:
        suffix = f" [{default}]" if default is not None else ""
        while True:
            val = input(f"{prompt}{suffix}: ").strip()
            if not val and default is not None:
                return default
            try:
                return int(val)
            except Exception:
                print(red("Please enter an integer."))

    def _prompt_choice(
        self, prompt: str, choices: list[str], default: str | None = None
    ) -> str:
        options = ", ".join(choices)
        suffix = f" [{default}]" if default is not None else ""
        while True:
            val = input(f"{prompt} ({options}){suffix}: ").strip()
            if not val and default is not None:
                return default
            if val in choices:
                return val
            print(red(f"Please choose one of: {options}"))

    def _derive_output_dir(self, answers: dict[str, Any]) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = answers.get("model_id") or answers.get("served_model_name") or "results"
        base = str(base).split("/")[-1].replace(" ", "_")
        return f"{base}_results_{ts}"

    def _derive_config_name(
        self, exec_choice: str, dep_choice: str, answers: dict[str, Any]
    ) -> str:
        name_parts = [
            exec_choice.split("/")[0],
            dep_choice,
            (answers.get("served_model_name") or answers.get("model_id") or "model")
            .split("/")[-1]
            .replace(" ", "_"),
            "quickstart",
        ]
        return "_".join([p for p in name_parts if p])

    def _determine_output_dir(self, cli_output: str | None) -> pathlib.Path:
        if cli_output:
            return pathlib.Path(cli_output).expanduser().resolve()
        # Prefer ./examples if it exists
        cwd = pathlib.Path.cwd()
        examples_dir = cwd / "examples"
        return examples_dir if examples_dir.exists() else cwd

    def _render_yaml(self, a: dict[str, Any]) -> str:
        env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(self._MAIN_TEMPLATE)
        return template.render(a=a)

    class _Prompt:
        def __init__(
            self,
            key: str,
            text: str,
            default: Any | None = None,
            required: bool = False,
            choices: list[str] | None = None,
            kind: str = "str",
        ):
            self.key = key
            self.text = text
            self.default = default
            self.required = required
            self.choices = choices or []
            self.kind = kind

    class _Group:
        def __init__(self, condition, prompts: list["Cmd._Prompt"]):
            self.condition = condition
            self.prompts = prompts

    def _build_flow(self, exec_choices: list[str], dep_choices: list[str]):
        return [
            self._Prompt(
                "execution_default",
                "Select execution platform",
                default="local",
                choices=exec_choices,
                kind="choice",
            ),
            self._Prompt(
                "deployment_default",
                "Select deployment type",
                default="none",
                choices=dep_choices,
                kind="choice",
            ),
            self._Group(
                lambda a: a.get("deployment_default") == "none",
                [
                    self._Prompt(
                        "model_id", "model_id", default="meta/llama-3.1-8b-instruct"
                    ),
                    self._Prompt(
                        "url",
                        "target.api_endpoint.url",
                        default="https://integrate.api.nvidia.com/v1/chat/completions",
                    ),
                    self._Prompt(
                        "api_key_name", "api_key_name (env var)", default="NGC_API_KEY"
                    ),
                ],
            ),
            self._Group(
                lambda a: a.get("deployment_default") == "vllm",
                [
                    self._Prompt(
                        "served_model_name",
                        "served_model_name",
                        default="llama-3.1-8b-instruct",
                    ),
                    self._Prompt(
                        "hf_model_handle",
                        "HF model handle",
                        default="meta-llama/Llama-3.1-8B-Instruct",
                    ),
                ],
            ),
            self._Group(
                lambda a: a.get("deployment_default") == "sglang",
                [
                    self._Prompt(
                        "served_model_name",
                        "served_model_name",
                        default="llama-3.1-8b-instruct",
                    ),
                    self._Prompt(
                        "hf_model_handle",
                        "HF model handle",
                        default="meta-llama/Llama-3.1-8B-Instruct",
                    ),
                ],
            ),
            self._Group(
                lambda a: a.get("deployment_default") == "trtllm",
                [
                    self._Prompt(
                        "served_model_name",
                        "served_model_name",
                        default="meta-llama/Llama-3.1-8B-Instruct",
                    ),
                    self._Prompt(
                        "checkpoint_path",
                        "checkpoint_path (ABS path)",
                        default="/checkpoint",
                    ),
                ],
            ),
            self._Group(
                lambda a: a.get("deployment_default") == "generic",
                [
                    self._Prompt("image", "image (Docker image)", required=True),
                    self._Prompt(
                        "command", "command (server start command)", required=True
                    ),
                    self._Prompt("port", "port", default=8000, kind="int"),
                    self._Prompt(
                        "served_model_name", "served_model_name", default="model"
                    ),
                ],
            ),
            self._Group(
                lambda a: a.get("deployment_default") == "nim",
                [
                    self._Prompt(
                        "served_model_name",
                        "served_model_name",
                        default="meta/llama-3.1-8b-instruct",
                    ),
                ],
            ),
            self._Group(
                lambda a: str(a.get("execution_default", "")).startswith("slurm"),
                [
                    self._Prompt(
                        "slurm_hostname", "slurm.hostname (required)", required=True
                    ),
                    self._Prompt(
                        "slurm_account", "slurm.account (required)", required=True
                    ),
                    self._Prompt(
                        "slurm_output_dir",
                        "slurm.output_dir (ABS path, required)",
                        required=True,
                    ),
                    self._Prompt(
                        "slurm_walltime", "slurm.walltime", default="02:00:00"
                    ),
                ],
            ),
            self._Group(
                lambda a: str(a.get("execution_default", "")).startswith("lepton"),
                [
                    self._Prompt(
                        "lepton_task_node_group",
                        "lepton node_group for tasks (optional)",
                        default="",
                    ),
                ],
            ),
            self._Group(
                lambda a: a.get("execution_default") == "local",
                [
                    self._Prompt(
                        "execution_output_dir",
                        "execution.output_dir",
                        default="results",
                    ),
                ],
            ),
            self._Prompt(
                "request_timeout", "request_timeout (seconds)", default=3600, kind="int"
            ),
            self._Prompt("parallelism", "parallelism", default=1, kind="int"),
            self._Prompt(
                "use_system_prompt",
                "Use system prompt?",
                default="true",
                choices=["true", "false"],
                kind="choice",
            ),
            self._Group(
                lambda a: a.get("use_system_prompt") == "true",
                [
                    self._Prompt(
                        "custom_system_prompt",
                        "Custom system prompt",
                        default="Think step by step.",
                    )
                ],
            ),
            self._Prompt(
                "endpoint_type",
                "Endpoint type",
                default="chat",
                choices=["chat", "completions"],
                kind="choice",
            ),
            self._Prompt(
                "enable_caching",
                "Enable response caching?",
                default="no",
                choices=["no", "yes"],
                kind="choice",
            ),
            self._Group(
                lambda a: a.get("enable_caching") == "yes",
                [
                    self._Prompt("cache_dir", "cache_dir", default="/results/cache"),
                    self._Prompt(
                        "reuse_cached",
                        "Reuse cached responses?",
                        default="yes",
                        choices=["yes", "no"],
                        kind="choice",
                    ),
                ],
            ),
            self._Prompt(
                "enable_request_logging",
                "Enable request logging?",
                default="no",
                choices=["no", "yes"],
                kind="choice",
            ),
            self._Group(
                lambda a: a.get("enable_request_logging") == "yes",
                [
                    self._Prompt(
                        "max_requests",
                        "request_logging.max_requests",
                        default=1,
                        kind="int",
                    )
                ],
            ),
            self._Prompt(
                "enable_payload_modifier",
                "Enable payload modifier (advanced)?",
                default="no",
                choices=["no", "yes"],
                kind="choice",
            ),
            self._Group(
                lambda a: a.get("enable_payload_modifier") == "yes",
                [
                    self._Prompt(
                        "enable_thinking",
                        "payload_modifier.enable_thinking?",
                        default="false",
                        choices=["false", "true"],
                        kind="choice",
                    ),
                    self._Prompt(
                        "thinking_budget",
                        "payload_modifier.thinking_budget",
                        default=-1,
                        kind="int",
                    ),
                ],
            ),
            self._Prompt(
                "tasks_csv",
                "Tasks (comma-separated, e.g., ifeval,gpqa_diamond)",
                default="ifeval",
            ),
            self._Prompt(
                "mlflow_enabled",
                "Enable MLflow auto-export?",
                default="no",
                choices=["no", "yes"],
                kind="choice",
            ),
            self._Group(
                lambda a: a.get("mlflow_enabled") == "yes",
                [
                    self._Prompt(
                        "mlflow_tracking_uri",
                        "MLflow tracking_uri",
                        default="http://mlflow.nvidia.com:5003",
                    ),
                    self._Prompt(
                        "mlflow_experiment", "MLflow experiment_name", default="nv-eval"
                    ),
                ],
            ),
        ]

    def _run_flow(self, flow: list[Any]) -> dict[str, Any]:
        a: dict[str, Any] = {}
        print(yellow("Follow the prompts. Press Enter to accept defaults."))
        for node in flow:
            if isinstance(node, self._Group):
                if node.condition(a):
                    for p in node.prompts:
                        self._apply_prompt(a, p)
                continue
            self._apply_prompt(a, node)

        if a.get("execution_default") == "local" and (
            not a.get("execution_output_dir")
            or a.get("execution_output_dir") == "results"
        ):
            a["execution_output_dir"] = self._derive_output_dir(a)

        a["tasks"] = [
            t.strip() for t in str(a.get("tasks_csv", "")).split(",") if t.strip()
        ]
        a["use_system_prompt"] = True if a.get("use_system_prompt") == "true" else False
        a["enable_caching"] = True if a.get("enable_caching") == "yes" else False
        a["reuse_cached"] = True if a.get("reuse_cached") == "yes" else False
        a["enable_request_logging"] = (
            True if a.get("enable_request_logging") == "yes" else False
        )
        a["enable_payload_modifier"] = (
            True if a.get("enable_payload_modifier") == "yes" else False
        )
        a["enable_thinking"] = True if a.get("enable_thinking") == "true" else False
        a["mlflow_enabled"] = True if a.get("mlflow_enabled") == "yes" else False
        return a

    def _apply_prompt(self, a: dict[str, Any], p: "Cmd._Prompt") -> None:
        if p.kind == "choice":
            a[p.key] = self._prompt_choice(p.text, p.choices or [], p.default)
        elif p.kind == "int":
            a[p.key] = self._prompt_int(
                p.text, p.default if isinstance(p.default, int) else None
            )
        else:
            a[p.key] = self._prompt_str(
                p.text,
                default=str(p.default) if p.default is not None else None,
                required=p.required,
            )

    _MAIN_TEMPLATE = r"""
# specify default configs for execution and deployment
defaults:
  - execution: {{ a.execution_default }}
  - deployment: {{ a.deployment_default }}
  - _self_

# execution configuration
execution:
{% if a.execution_default == 'local' %}
  output_dir: {{ a.execution_output_dir }}
  # mode: sequential  # alternative: remove to use package default
{% elif a.execution_default.startswith('slurm') %}
  hostname: {{ a.slurm_hostname }}  # required
  username: ${oc.env:USER}  # default: $USER
  account: {{ a.slurm_account }}  # required
  output_dir: {{ a.slurm_output_dir }}  # required ABSOLUTE path
  walltime: {{ a.slurm_walltime }}  # default in example: 02:00:00
  # partition: backfill  # alternatives: batch, cpu_short, ...
  env_vars:
    deployment: {}  # add secrets if needed
    evaluation: {}
  mounts:
    deployment: {}  # source:target pairs
    evaluation: {}
    mount_home: true
{% elif a.execution_default.startswith('lepton') %}
  output_dir: results  # change as needed
  evaluation_tasks:
    timeout: 3600  # time to wait for endpoint
  lepton_platform:
    # You will likely need to edit these after generation
    tasks:
      # node_group: {{ a.lepton_task_node_group or '"nv-int-multiteam-nebius-h200-01"' }}
      # mounts and secrets can be set here
{% endif %}

{% if a.deployment_default == 'none' %}
# target endpoint configuration (no deployment)
target:
  api_endpoint:
    model_id: {{ a.model_id }}
    url: {{ a.url }}
    api_key_name: {{ a.api_key_name }}  # name of env var or secret

{% elif a.deployment_default == 'vllm' %}
# vLLM deployment configuration
deployment:
  served_model_name: {{ a.served_model_name }}  # used by client and server
  checkpoint_path: {{ a.hf_model_handle }}  # HF model handle or absolute checkpoint path
  tensor_parallel_size: 1  # alternatives: 1..N
  data_parallel_size: 1  # alternatives: 1..N
  # extra_args: "--max-model-len 32768"

{% elif a.deployment_default == 'sglang' %}
# SGLang deployment configuration
deployment:
  served_model_name: {{ a.served_model_name }}
  checkpoint_path: {{ a.hf_model_handle }}  # HF model handle or absolute checkpoint path
  tensor_parallel_size: 1  # alternatives: 1..N
  data_parallel_size: 1  # alternatives: 1..N
  # extra_args: "--max-model-len 32768"

{% elif a.deployment_default == 'trtllm' %}
# TensorRT-LLM deployment configuration
deployment:
  served_model_name: {{ a.served_model_name }}
  checkpoint_path: {{ a.checkpoint_path }}
  tensor_parallel_size: 4  # alternatives: 1..N
  pipeline_parallel_size: 1
  # extra_args: "--max-model-len 32768"

{% elif a.deployment_default == 'generic' %}
# Generic deployment configuration
deployment:
  image: {{ a.image }}
  command: {{ a.command }}
  port: {{ a.port }}
  served_model_name: {{ a.served_model_name }}
  extra_args: ""
  env_vars: {}

{% elif a.deployment_default == 'nim' %}
# NIM deployment configuration
deployment:
  image: nvcr.io/nim/{{ a.served_model_name.split('/')[-1] }}:latest
  served_model_name: {{ a.served_model_name }}
  # lepton_config:  # typically used with Lepton execution
  #   endpoint_name: my-endpoint
  #   resource_shape: gpu.1xh200

{% endif %}

# evaluation configuration
evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: {{ a.request_timeout }}
        parallelism: {{ a.parallelism }}
      target:
        api_endpoint:
          adapter_config:
            # Adapter config docs: https://github.com/NVIDIA-NeMo/Evaluator/tree/main/docs/nemo-evaluator-launcher/index.md
            use_reasoning: false  # true strips reasoning tokens, collects stats
            use_system_prompt: {{ 'true' if a.use_system_prompt else 'false' }}
            {% if a.use_system_prompt %}custom_system_prompt: "{{ a.custom_system_prompt | replace('\n', '\\n') }}"{% endif %}
            endpoint_type: {{ a.endpoint_type }}  # alternatives: chat, completions
            {% set add_interceptors = a.enable_payload_modifier or a.enable_request_logging or a.enable_caching %}
            {% if add_interceptors %}
            # Interceptors docs: https://github.com/NVIDIA-NeMo/Evaluator/tree/main/docs/nemo-evaluator-launcher/index.md
            interceptors:
              {% if a.enable_payload_modifier %}
              - name: "payload_modifier"
                config:
                  params_to_add:
                    extra_body:
                      chat_template_kwargs:
                        enable_thinking: {{ 'true' if a.enable_thinking else 'false' }}
                        thinking_budget: {{ a.thinking_budget }}
              {% endif %}
              {% if a.enable_request_logging %}
              - name: request_logging
                enabled: true
                config:
                  max_requests: {{ a.max_requests }}
              {% endif %}
              {% if a.enable_caching %}
              - name: caching
                enabled: true
                config:
                  cache_dir: {{ a.cache_dir }}
                  reuse_cached_responses: {{ 'true' if a.reuse_cached else 'false' }}
              {% endif %}
              - name: endpoint
                enabled: true
              - name: response_stats
                enabled: true
            {% endif %}
  tasks:
{% for name in a.tasks %}
    - name: {{ name }}
{% if name == 'gpqa_diamond' %}
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND  # request access on HF
{% endif %}
{% endfor %}

{% if a.mlflow_enabled %}
# Auto-export destinations
execution:
  auto_export:
    destinations: ["mlflow"]

# Exporter configurations (paired with auto-export)
export:
  mlflow:
    tracking_uri: "{{ a.mlflow_tracking_uri }}"
    experiment_name: "{{ a.mlflow_experiment }}"
{% endif %}
"""

    # Discover choices from installed package configs
    def _list_execution_defaults(self) -> list[str]:
        results: list[str] = []
        try:
            base = (
                importlib_resources.files("nemo_evaluator_launcher.configs")
                / "execution"
            )
            # Top-level YAMLs
            for entry in base.iterdir():
                if entry.is_file() and entry.name.endswith(".yaml"):
                    results.append(entry.name.replace(".yaml", ""))
                elif entry.is_dir():
                    for sub in entry.iterdir():
                        if sub.is_file() and sub.name.endswith(".yaml"):
                            results.append(
                                f"{entry.name}/{sub.name.replace('.yaml', '')}"
                            )
        except Exception:
            pass
        # Ensure stable and expected order if present
        unique = sorted(set(results))
        # Prefer common order
        ordered: list[str] = []
        for pref in ["local", "slurm/default", "lepton/default"]:
            if pref in unique:
                ordered.append(pref)
        for item in unique:
            if item not in ordered:
                ordered.append(item)
        return ordered

    def _list_deployment_types(self) -> list[str]:
        results: list[str] = []
        try:
            base = (
                importlib_resources.files("nemo_evaluator_launcher.configs")
                / "deployment"
            )
            for entry in base.iterdir():
                if entry.is_file() and entry.name.endswith(".yaml"):
                    results.append(entry.name.replace(".yaml", ""))
        except Exception:
            pass
        # Stable order with common first
        unique = sorted(set(results))
        ordered: list[str] = []
        for pref in ["none", "vllm", "nim", "sglang", "trtllm", "generic"]:
            if pref in unique:
                ordered.append(pref)
        for item in unique:
            if item not in ordered:
                ordered.append(item)
        return ordered
