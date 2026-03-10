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
"""Generic auxiliary deployment system for NeMo Evaluator Launcher.

Each auxiliary deployment is identified by a config key (e.g. "judge",
"user", "reranker") and an env_prefix used to derive shell variable
names for the generated sbatch script.
"""

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.env_vars import (
    EnvVarValue,
    _collect_top_level_env_vars,
    parse_env_var_value,
)
from nemo_evaluator_launcher.common.logging_utils import logger

_RESERVED_PREFIXES = {"SLURM", "PRIMARY", "MODEL", "SERVER"}

_REQUIRED_FIELDS = {"type", "image", "port", "served_model_name", "endpoints"}

_DEFAULT_VLLM_COMMAND_TEMPLATE = (
    "vllm serve {hf_model_handle}"
    " --tensor-parallel-size={tensor_parallel_size}"
    " --pipeline-parallel-size={pipeline_parallel_size}"
    " --data-parallel-size={data_parallel_size}"
    " --port={port}"
    " --served-model-name={served_model_name}"
    " --gpu-memory-utilization {gpu_memory_utilization}"
    " {extra_args}"
)

_PROXY_PORT_START = 5010


@dataclass
class AuxDeploymentState:
    name: str
    env_prefix: str
    cfg: DictConfig
    num_nodes: int
    num_instances: int
    env_vars: dict[str, EnvVarValue]
    reexport_cmd: str = ""
    nodes_var: str = ""
    nodelist_var: str = ""
    primary_node_var: str = ""
    pid_var: str = ""
    pids_var: str = ""
    proxy_port: int = 0
    proxy_pid_var: str = ""

    def __post_init__(self) -> None:
        if not self.nodes_var:
            self.nodes_var = f"{self.env_prefix}_NODES"
        if not self.nodelist_var:
            self.nodelist_var = f"{self.env_prefix}_NODELIST"
        if not self.primary_node_var:
            self.primary_node_var = f"{self.env_prefix}_PRIMARY_NODE"
        if not self.pid_var:
            self.pid_var = f"{self.env_prefix}_SERVER_PID"
        if not self.pids_var:
            self.pids_var = f"{self.env_prefix}_SERVER_PIDS"
        if not self.proxy_pid_var:
            self.proxy_pid_var = f"{self.env_prefix}_PROXY_PID"


def collect_auxiliary_deployment_env_vars(
    cfg: DictConfig,
    deploy_cfg: DictConfig,
) -> dict[str, EnvVarValue]:
    """Collect and parse env vars for an auxiliary deployment.

    Merges (last wins):
        cfg.env_vars -> deploy_cfg.env_vars
    """
    top_level_vars = _collect_top_level_env_vars(cfg)
    parsed: dict[str, EnvVarValue] = {}
    for target_name, raw_value in top_level_vars.items():
        parsed[target_name] = parse_env_var_value(str(raw_value))

    if deploy_cfg.get("env_vars"):
        for target_name, raw_value in deploy_cfg["env_vars"].items():
            parsed[target_name] = parse_env_var_value(str(raw_value))

    return parsed


def build_aux_deployment_states(cfg: DictConfig) -> list[AuxDeploymentState]:
    """Build AuxDeploymentState objects from cfg.auxiliary_deployments.

    Skips entries where type == "none". Assigns proxy ports starting from
    5010 for deployments with num_instances > 1.
    """
    aux_deployments = cfg.get("auxiliary_deployments")
    if not aux_deployments:
        return []

    states: list[AuxDeploymentState] = []
    next_proxy_port = _PROXY_PORT_START

    for name, deploy_cfg in aux_deployments.items():
        deploy_type = deploy_cfg.get("type", "none")
        if deploy_type == "none":
            logger.debug(f"Skipping auxiliary deployment '{name}' (type=none)")
            continue

        env_prefix = str(deploy_cfg.get("env_prefix", name.upper()))
        num_nodes = int(deploy_cfg.get("num_nodes", 1))
        num_instances = int(deploy_cfg.get("num_instances", 1))

        proxy_port = 0
        if num_instances > 1:
            proxy_port = next_proxy_port
            next_proxy_port += 1

        env_vars = collect_auxiliary_deployment_env_vars(cfg, deploy_cfg)

        state = AuxDeploymentState(
            name=name,
            env_prefix=env_prefix,
            cfg=deploy_cfg,
            num_nodes=num_nodes,
            num_instances=num_instances,
            env_vars=env_vars,
            proxy_port=proxy_port,
        )
        states.append(state)

        logger.debug(
            f"Built auxiliary deployment state for '{name}'",
            env_prefix=env_prefix,
            num_nodes=num_nodes,
            num_instances=num_instances,
            proxy_port=proxy_port,
        )

    return states


def validate_auxiliary_deployments(
    aux_list: list[AuxDeploymentState],
    primary_port: int,
) -> None:
    """Validate auxiliary deployment states for conflicts and correctness.

    Raises:
        ValueError: On duplicate prefixes, port conflicts, reserved prefixes,
            indivisible nodes/instances, or missing required fields.
    """
    seen_prefixes: set[str] = set()
    seen_ports: set[int] = {primary_port}

    for state in aux_list:
        # Reserved prefixes
        if state.env_prefix in _RESERVED_PREFIXES:
            raise ValueError(
                f"Auxiliary deployment '{state.name}' uses reserved "
                f"env_prefix '{state.env_prefix}'. "
                f"Reserved prefixes: {_RESERVED_PREFIXES}"
            )

        # Duplicate prefixes
        if state.env_prefix in seen_prefixes:
            raise ValueError(
                f"Duplicate env_prefix '{state.env_prefix}' in auxiliary deployments"
            )
        seen_prefixes.add(state.env_prefix)

        # Port conflicts
        port = int(state.cfg.get("port", 0))
        if port and port in seen_ports:
            raise ValueError(
                f"Auxiliary deployment '{state.name}' port {port} "
                "conflicts with another deployment"
            )
        if port:
            seen_ports.add(port)

        if state.proxy_port and state.proxy_port in seen_ports:
            raise ValueError(
                f"Auxiliary deployment '{state.name}' proxy port "
                f"{state.proxy_port} conflicts with another deployment"
            )
        if state.proxy_port:
            seen_ports.add(state.proxy_port)

        # num_nodes divisible by num_instances
        if state.num_instances > 1 and state.num_nodes % state.num_instances != 0:
            raise ValueError(
                f"Auxiliary deployment '{state.name}': num_nodes ({state.num_nodes}) "
                f"must be divisible by num_instances ({state.num_instances})"
            )

        # Required fields
        cfg_container = OmegaConf.to_container(state.cfg, resolve=False)
        missing = _REQUIRED_FIELDS - set(cfg_container.keys())
        if missing:
            raise ValueError(
                f"Auxiliary deployment '{state.name}' is missing "
                f"required fields: {missing}"
            )


def resolve_deployment_command(deploy_cfg: DictConfig) -> str:
    """Resolve the server launch command for a deployment config.

    Priority:
        1. deploy_cfg.command (literal user-provided command)
        2. deploy_cfg.command_template (format string with config values)
        3. Default vllm command template
    """
    if deploy_cfg.get("command"):
        return str(deploy_cfg.command)

    template = str(deploy_cfg.get("command_template", _DEFAULT_VLLM_COMMAND_TEMPLATE))
    cfg_dict = OmegaConf.to_container(deploy_cfg, resolve=True)
    return template.format(**cfg_dict)
