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
"""Utilities for working with configuration models."""

from typing import Any, Dict, Union

from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.models.config import RunConfig
from nemo_evaluator_launcher.models.deployment import (
    DeploymentConfig,
    GenericDeploymentConfig,
    NIMDeploymentConfig,
    NoDeploymentConfig,
    SGLangDeploymentConfig,
    TRTLLMDeploymentConfig,
    VLLMDeploymentConfig,
)
from nemo_evaluator_launcher.models.execution import (
    ExecutorConfig,
    LeptonExecutorConfig,
    LocalExecutorConfig,
    SlurmExecutorConfig,
)


def parse_executor_config(config: Union[Dict[str, Any], DictConfig]) -> ExecutorConfig:
    """Parse executor configuration from dict or DictConfig.

    Args:
        config: Executor configuration as dict or OmegaConf DictConfig

    Returns:
        Appropriate ExecutorConfig subclass instance

    Raises:
        ValueError: If executor type is unknown
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    exec_type = config.get("type", "local")

    if exec_type == "local":
        return LocalExecutorConfig(**config)
    elif exec_type == "slurm":
        return SlurmExecutorConfig(**config)
    elif exec_type == "lepton":
        return LeptonExecutorConfig(**config)
    else:
        raise ValueError(f"Unknown executor type: {exec_type}")


def parse_deployment_config(
    config: Union[Dict[str, Any], DictConfig],
) -> DeploymentConfig:
    """Parse deployment configuration from dict or DictConfig.

    Args:
        config: Deployment configuration as dict or OmegaConf DictConfig

    Returns:
        Appropriate DeploymentConfig subclass instance

    Raises:
        ValueError: If deployment type is unknown
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    deploy_type = config.get("type", "none")

    if deploy_type == "none":
        return NoDeploymentConfig(**config)
    elif deploy_type == "generic":
        return GenericDeploymentConfig(**config)
    elif deploy_type == "vllm":
        return VLLMDeploymentConfig(**config)
    elif deploy_type == "nim":
        return NIMDeploymentConfig(**config)
    elif deploy_type == "trtllm":
        return TRTLLMDeploymentConfig(**config)
    elif deploy_type == "sglang":
        return SGLangDeploymentConfig(**config)
    else:
        raise ValueError(f"Unknown deployment type: {deploy_type}")


def parse_run_config(config: Union[Dict[str, Any], DictConfig]) -> RunConfig:
    """Parse complete run configuration from dict or DictConfig.

    Args:
        config: Complete run configuration as dict or OmegaConf DictConfig

    Returns:
        RunConfig instance with parsed executor and deployment configs
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Parse nested configs
    parsed_config = config.copy()
    if "execution" in parsed_config:
        parsed_config["execution"] = parse_executor_config(parsed_config["execution"])
    if "deployment" in parsed_config:
        parsed_config["deployment"] = parse_deployment_config(
            parsed_config["deployment"]
        )

    return RunConfig(**parsed_config)


def config_to_dict(
    config: Union[RunConfig, ExecutorConfig, DeploymentConfig],
) -> Dict[str, Any]:
    """Convert a Pydantic config model to a dictionary.

    Args:
        config: Any Pydantic config model

    Returns:
        Dictionary representation
    """
    return config.model_dump(exclude_none=False, by_alias=True)


def config_to_omegaconf(
    config: Union[RunConfig, ExecutorConfig, DeploymentConfig],
) -> DictConfig:
    """Convert a Pydantic config model to OmegaConf DictConfig.

    Args:
        config: Any Pydantic config model

    Returns:
        OmegaConf DictConfig
    """
    return OmegaConf.create(config_to_dict(config))
