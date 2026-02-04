#!/usr/bin/env python3
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
"""
NEL Config Validator - Validates NeMo Evaluator Launcher configurations.

This module provides Pydantic models and validation functions for NEL config files.
It can be used as a library or as a CLI tool.

Usage:
    python verify_config.py <config_path>

Example:
    python verify_config.py my_config.yaml
"""

from __future__ import annotations

import sys
from typing import Annotated, Any, Literal, Optional, Union

from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# =============================================================================
# Helper Types
# =============================================================================


class StrictModel(BaseModel):
    """Base model with strict validation (forbid extra fields)."""

    model_config = ConfigDict(extra="forbid")


class FlexibleModel(BaseModel):
    """Base model with flexible validation (allow extra fields for extensibility)."""

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Auto Export Config
# =============================================================================


# Valid destinations for auto_export (must push to external services).
# Note: 'local' is excluded because in auto_export context (running on remote node),
# it would just copy files to another location on the same node, which is not useful.
VALID_AUTO_EXPORT_DESTINATIONS = {"mlflow", "wandb"}


class AutoExportConfig(StrictModel):
    """Auto-export configuration for execution."""

    destinations: list[str] = Field(
        default_factory=list,
        description="Export destinations (e.g., ['mlflow', 'wandb'])",
    )
    launcher_install_cmd: Optional[str] = Field(
        default=None, description="Custom command to install nemo-evaluator-launcher"
    )

    @field_validator("destinations", mode="after")
    @classmethod
    def validate_destinations(cls, v: list[str]) -> list[str]:
        """Validate that destinations only contain valid exporter names."""
        invalid = [d for d in v if d not in VALID_AUTO_EXPORT_DESTINATIONS]
        if invalid:
            raise ValueError(
                f"Invalid auto_export destination(s): {invalid}. "
                f"Valid options: {sorted(VALID_AUTO_EXPORT_DESTINATIONS)}"
            )
        return v


# =============================================================================
# Execution Configs (Discriminated Union by 'type')
# =============================================================================


class LocalExecutionConfig(StrictModel):
    """Local Docker execution configuration."""

    type: Literal["local"] = Field(description="Execution type: local")
    output_dir: str = Field(description="Directory for output files")
    extra_docker_args: str = Field(
        default="", description="Additional Docker arguments"
    )
    mode: str = Field(default="sequential", description="Execution mode")
    auto_export: Optional[AutoExportConfig] = Field(
        default=None, description="Auto-export configuration"
    )
    env_vars: Optional[dict[str, Any]] = Field(
        default=None, description="Environment variables"
    )
    mounts: Optional[dict[str, Any]] = Field(
        default=None, description="Volume mounts for containers"
    )


class SlurmDeploymentSettings(StrictModel):
    """SLURM deployment-specific settings."""

    n_tasks: Optional[int] = Field(
        default=None, description="Number of tasks for deployment srun"
    )


class SlurmProxyHaproxyConfig(StrictModel):
    """HAProxy configuration for SLURM proxy."""

    haproxy_port: int = Field(default=5009, description="HAProxy port")
    health_check_path: str = Field(default="/health", description="Health check path")
    health_check_status: int = Field(
        default=200, description="Expected health check status"
    )


class SlurmProxyConfig(FlexibleModel):
    """SLURM proxy configuration."""

    type: str = Field(default="haproxy", description="Proxy type")
    image: str = Field(default="haproxy:latest", description="Proxy image")
    config: Optional[SlurmProxyHaproxyConfig] = Field(
        default=None, description="Proxy-specific configuration"
    )


class SlurmMountsConfig(FlexibleModel):
    """SLURM mounts configuration."""

    deployment: Optional[dict[str, str]] = Field(
        default=None, description="Deployment mounts"
    )
    evaluation: Optional[dict[str, str]] = Field(
        default=None, description="Evaluation mounts"
    )
    mount_home: bool = Field(
        default=True, description="Whether to mount home directory"
    )


class SlurmEnvVarsConfig(FlexibleModel):
    """SLURM environment variables configuration."""

    deployment: Optional[dict[str, Any]] = Field(
        default=None, description="Deployment environment variables"
    )
    evaluation: Optional[dict[str, Any]] = Field(
        default=None, description="Evaluation environment variables"
    )
    export: Optional[dict[str, Any]] = Field(
        default=None, description="Export environment variables"
    )


class SlurmExecutionConfig(StrictModel):
    """SLURM cluster execution configuration."""

    type: Literal["slurm"] = Field(description="Execution type: slurm")
    hostname: str = Field(description="SLURM headnode (login) hostname (required)")
    username: str = Field(description="Cluster username")
    account: str = Field(description="SLURM account allocation (required)")
    output_dir: str = Field(
        description="Absolute path accessible on compute nodes (required)"
    )
    partition: str = Field(default="batch", description="SLURM partition")
    num_nodes: int = Field(default=1, description="Number of nodes")
    ntasks_per_node: int = Field(default=1, description="Tasks per node")
    gres: str = Field(default="gpu:8", description="Generic resources (e.g., gpu:8)")
    walltime: str = Field(default="01:00:00", description="Job wall time")
    max_walltime: Optional[str] = Field(
        default=None, description="Maximum total runtime across all resumes"
    )
    subproject: Optional[str] = Field(
        default=None, description="Subproject name for organization"
    )
    sbatch_comment: Optional[str] = Field(
        default=None, description="Optional comment for SLURM job"
    )
    deployment: Optional[SlurmDeploymentSettings] = Field(
        default=None, description="Deployment-specific SLURM configuration"
    )
    env_vars: Optional[SlurmEnvVarsConfig] = Field(
        default=None, description="Environment variables"
    )
    mounts: Optional[SlurmMountsConfig] = Field(
        default=None, description="Volume mounts"
    )
    proxy: Optional[SlurmProxyConfig] = Field(
        default=None, description="Proxy configuration for multi-node"
    )
    auto_export: Optional[AutoExportConfig] = Field(
        default=None, description="Auto-export configuration"
    )


class LeptonEvaluationTasks(FlexibleModel):
    """Lepton evaluation task settings."""

    resource_shape: str = Field(
        default="cpu.small", description="Resource shape for evaluation tasks"
    )
    timeout: int = Field(
        default=3600, description="Timeout for individual evaluation tasks"
    )
    use_shared_storage: bool = Field(
        default=True, description="Whether to use shared storage for results"
    )


class LeptonValueFrom(FlexibleModel):
    """Lepton secret/token reference."""

    secret_name_ref: Optional[str] = Field(
        default=None, description="Secret name reference"
    )
    token_name_ref: Optional[str] = Field(
        default=None, description="Token name reference"
    )


class LeptonEnvVar(FlexibleModel):
    """Lepton environment variable (can be direct value or reference)."""

    value_from: Optional[LeptonValueFrom] = Field(
        default=None, description="Value from secret/token reference"
    )


class LeptonApiToken(FlexibleModel):
    """Lepton API token configuration."""

    value_from: Optional[LeptonValueFrom] = Field(
        default=None, description="Token reference"
    )


class LeptonMount(FlexibleModel):
    """Lepton storage mount configuration."""

    from_: Optional[str] = Field(
        default=None, alias="from", description="Storage source"
    )
    path: Optional[str] = Field(default=None, description="Path on storage")
    mount_path: Optional[str] = Field(
        default=None, description="Mount path in container"
    )


class LeptonDeploymentPlatformDefaults(FlexibleModel):
    """Lepton deployment platform defaults."""

    health: Optional[dict[str, Any]] = Field(default=None, description="Health checks")
    liveness: Optional[dict[str, Any]] = Field(
        default=None, description="Liveness checks"
    )
    log: Optional[dict[str, Any]] = Field(default=None, description="Logging")
    metrics: Optional[dict[str, Any]] = Field(default=None, description="Metrics")
    routing_policy: Optional[dict[str, Any]] = Field(
        default=None, description="Routing policy"
    )
    queue_config: Optional[dict[str, Any]] = Field(
        default=None, description="Queue configuration"
    )
    enable_rdma: Optional[bool] = Field(default=None, description="Enable RDMA")
    user_security_context: Optional[dict[str, Any]] = Field(
        default=None, description="Security context"
    )
    image_pull_secrets: Optional[list[str]] = Field(
        default=None, description="Image pull secrets"
    )


class LeptonDeploymentConfig(FlexibleModel):
    """Lepton deployment configuration within platform."""

    node_group: Optional[str] = Field(
        default=None, description="Node group for endpoint deployments"
    )
    endpoint_readiness_timeout: Optional[int] = Field(
        default=None, description="Endpoint readiness timeout"
    )
    platform_defaults: Optional[LeptonDeploymentPlatformDefaults] = Field(
        default=None, description="Platform defaults"
    )


class LeptonTasksConfig(FlexibleModel):
    """Lepton task execution configuration."""

    node_group: Optional[str] = Field(
        default=None, description="Node group for evaluation tasks"
    )
    env_vars: Optional[dict[str, Any]] = Field(
        default=None, description="Environment variables for tasks"
    )
    mounts: Optional[list[LeptonMount]] = Field(
        default=None, description="Storage mounts for task execution"
    )
    image_pull_secrets: Optional[list[str]] = Field(
        default=None, description="Image pull secrets"
    )
    api_tokens: Optional[list[LeptonApiToken]] = Field(
        default=None, description="API tokens for tasks"
    )


class LeptonPlatformConfig(FlexibleModel):
    """Lepton platform infrastructure settings."""

    deployment: Optional[LeptonDeploymentConfig] = Field(
        default=None, description="Deployment configuration"
    )
    tasks: Optional[LeptonTasksConfig] = Field(
        default=None, description="Task execution configuration"
    )


class LeptonExecutionConfig(StrictModel):
    """Lepton cloud execution configuration."""

    type: Literal["lepton"] = Field(description="Execution type: lepton")
    output_dir: str = Field(description="Output directory")
    env_var_names: list[str] = Field(
        default_factory=list, description="Environment variable names to pass"
    )
    evaluation_tasks: Optional[LeptonEvaluationTasks] = Field(
        default=None, description="Evaluation task settings"
    )
    lepton_platform: Optional[LeptonPlatformConfig] = Field(
        default=None, description="Lepton platform settings"
    )


# Discriminated union for execution config
ExecutionConfig = Annotated[
    Union[LocalExecutionConfig, SlurmExecutionConfig, LeptonExecutionConfig],
    Field(discriminator="type"),
]


# =============================================================================
# Deployment Configs (Discriminated Union by 'type')
# =============================================================================


class EndpointsConfig(StrictModel):
    """API endpoints configuration for deployment."""

    chat: str = Field(
        default="/v1/chat/completions", description="Chat completions endpoint"
    )
    completions: str = Field(
        default="/v1/completions", description="Completions endpoint"
    )
    health: str = Field(default="/health", description="Health check endpoint")


class NoneDeploymentConfig(StrictModel):
    """No deployment - use external endpoint."""

    type: Literal["none"] = Field(description="Deployment type: none (external API)")


class LeptonMountsConfig(FlexibleModel):
    """Lepton mounts configuration for deployment."""

    enabled: bool = Field(default=False, description="Whether mounts are enabled")
    cache_path: Optional[str] = Field(default=None, description="Cache path on storage")
    mount_path: Optional[str] = Field(
        default=None, description="Mount path in container"
    )


class LeptonAutoScalerScaleDown(FlexibleModel):
    """Lepton auto-scaler scale-down configuration."""

    no_traffic_timeout: int = Field(
        default=3600, description="Timeout with no traffic before scaling down"
    )
    scale_from_zero: bool = Field(
        default=False, description="Whether to scale from zero"
    )


class LeptonAutoScaler(FlexibleModel):
    """Lepton auto-scaler configuration."""

    scale_down: Optional[LeptonAutoScalerScaleDown] = Field(
        default=None, description="Scale down configuration"
    )


class LeptonDeploymentConfigForDeployment(FlexibleModel):
    """Lepton-specific deployment settings (within deployment section)."""

    endpoint_name: Optional[str] = Field(
        default=None, description="Base name for Lepton endpoint"
    )
    resource_shape: Optional[str] = Field(
        default=None, description="GPU shape for endpoint"
    )
    min_replicas: int = Field(default=1, description="Minimum replicas")
    max_replicas: int = Field(default=1, description="Maximum replicas")
    api_tokens: Optional[list[LeptonApiToken]] = Field(
        default=None, description="API tokens"
    )
    auto_scaler: Optional[LeptonAutoScaler] = Field(
        default=None, description="Auto-scaling settings"
    )
    envs: Optional[dict[str, Any]] = Field(
        default=None, description="Environment variables for container"
    )
    mounts: Optional[LeptonMountsConfig] = Field(
        default=None, description="Storage mounts for model caching"
    )


class VLLMDeploymentConfig(StrictModel):
    """vLLM deployment configuration."""

    type: Literal["vllm"] = Field(description="Deployment type: vllm")
    image: str = Field(default="vllm/vllm-openai:latest", description="Docker image")
    checkpoint_path: Optional[str] = Field(
        default=None, description="Model checkpoint path"
    )
    hf_model_handle: Optional[str] = Field(
        default=None, description="HuggingFace model ID"
    )
    served_model_name: str = Field(description="Model name for serving")
    port: int = Field(default=8000, description="Server port")
    tensor_parallel_size: int = Field(default=8, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, description="Pipeline parallel size")
    data_parallel_size: int = Field(default=1, description="Data parallel size")
    data_parallel_size_local: Optional[int] = Field(
        default=None, description="Local data parallel size (GPUs per node)"
    )
    gpu_memory_utilization: float = Field(
        default=0.95, description="GPU memory utilization"
    )
    extra_args: str = Field(default="", description="Extra vLLM arguments")
    env_vars: dict[str, Any] = Field(
        default_factory=dict, description="Environment variables"
    )
    endpoints: Optional[EndpointsConfig] = Field(
        default=None, description="API endpoints configuration"
    )
    command: Optional[str] = Field(default=None, description="Custom command")
    pre_cmd: Optional[str] = Field(default=None, description="Pre-command to run")
    multiple_instances: Optional[bool] = Field(
        default=None, description="Enable HAProxy load balancing across nodes"
    )
    lepton_config: Optional[LeptonDeploymentConfigForDeployment] = Field(
        default=None, description="Lepton-specific deployment settings"
    )


class SGLangDeploymentConfig(StrictModel):
    """SGLang deployment configuration."""

    type: Literal["sglang"] = Field(description="Deployment type: sglang")
    image: str = Field(default="lmsysorg/sglang:latest", description="Docker image")
    checkpoint_path: Optional[str] = Field(
        default=None, description="Model checkpoint path"
    )
    hf_model_handle: Optional[str] = Field(
        default=None, description="HuggingFace model ID"
    )
    served_model_name: str = Field(description="Model name for serving")
    port: int = Field(default=8000, description="Server port")
    tensor_parallel_size: int = Field(default=8, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, description="Pipeline parallel size")
    data_parallel_size: int = Field(default=1, description="Data parallel size")
    extra_args: str = Field(default="", description="Extra SGLang arguments")
    env_vars: dict[str, Any] = Field(
        default_factory=dict, description="Environment variables"
    )
    endpoints: Optional[EndpointsConfig] = Field(
        default=None, description="API endpoints configuration"
    )
    command: Optional[str] = Field(default=None, description="Custom command")
    pre_cmd: Optional[str] = Field(default=None, description="Pre-command to run")


class NIMDeploymentConfig(StrictModel):
    """NIM deployment configuration.

    Note: pre_cmd is not supported for NIM deployments.
    """

    type: Literal["nim"] = Field(description="Deployment type: nim")
    image: str = Field(description="NIM container image")
    served_model_name: str = Field(description="Model name for serving")
    port: int = Field(default=8000, description="Server port")
    command: Optional[str] = Field(default=None, description="Custom command")
    endpoints: Optional[EndpointsConfig] = Field(
        default=None, description="API endpoints configuration"
    )
    lepton_config: Optional[LeptonDeploymentConfigForDeployment] = Field(
        default=None, description="Lepton-specific deployment settings"
    )


class TRTLLMDeploymentConfig(StrictModel):
    """TensorRT-LLM deployment configuration."""

    type: Literal["trtllm"] = Field(description="Deployment type: trtllm")
    image: str = Field(
        default="nvcr.io/nvidia/tensorrt-llm/release:1.0.0", description="Docker image"
    )
    checkpoint_path: Optional[str] = Field(
        default=None, description="Model checkpoint path"
    )
    served_model_name: str = Field(description="Model name for serving")
    port: int = Field(default=8000, description="Server port")
    tensor_parallel_size: int = Field(default=8, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, description="Pipeline parallel size")
    extra_args: str = Field(default="", description="Extra TensorRT-LLM arguments")
    endpoints: Optional[EndpointsConfig] = Field(
        default=None, description="API endpoints configuration"
    )
    command: Optional[str] = Field(default=None, description="Custom command")
    pre_cmd: Optional[str] = Field(default=None, description="Pre-command to run")


class GenericDeploymentConfig(StrictModel):
    """Generic server deployment configuration."""

    type: Literal["generic"] = Field(description="Deployment type: generic")
    image: str = Field(description="Docker image to use for deployment")
    command: str = Field(description="Command to run the server")
    pre_cmd: Optional[str] = Field(default=None, description="Pre-command to run")
    port: int = Field(default=8000, description="Server port")
    served_model_name: str = Field(description="Name of the served model")
    extra_args: str = Field(default="", description="Additional command line arguments")
    env_vars: dict[str, Any] = Field(
        default_factory=dict, description="Environment variables"
    )
    checkpoint_path: Optional[str] = Field(
        default=None, description="Path to model checkpoint"
    )
    endpoints: Optional[EndpointsConfig] = Field(
        default=None, description="API endpoints configuration"
    )


# Discriminated union for deployment config
DeploymentConfig = Annotated[
    Union[
        NoneDeploymentConfig,
        VLLMDeploymentConfig,
        SGLangDeploymentConfig,
        NIMDeploymentConfig,
        TRTLLMDeploymentConfig,
        GenericDeploymentConfig,
    ],
    Field(discriminator="type"),
]


# =============================================================================
# Target & Evaluation Configs
# =============================================================================


class InterceptorConfig(FlexibleModel):
    """Interceptor configuration within adapter_config."""

    name: str = Field(description="Interceptor name")
    enabled: bool = Field(default=True, description="Whether interceptor is enabled")
    config: Optional[dict[str, Any]] = Field(
        default=None, description="Interceptor-specific configuration"
    )


class DiscoveryConfig(StrictModel):
    """Adapter discovery configuration."""

    modules: list[str] = Field(default_factory=list, description="Modules to discover")
    dirs: list[str] = Field(default_factory=list, description="Directories to discover")


class PostEvalHookConfig(FlexibleModel):
    """Post-evaluation hook configuration."""

    name: str = Field(description="Hook name")
    enabled: bool = Field(default=True, description="Whether enabled")
    config: Optional[dict[str, Any]] = Field(
        default=None, description="Hook configuration"
    )


class AdapterConfig(StrictModel):
    """Adapter configuration for API endpoint."""

    discovery: Optional[DiscoveryConfig] = Field(
        default=None, description="Discovery configuration"
    )
    interceptors: Optional[list[InterceptorConfig]] = Field(
        default=None, description="List of interceptors in processing order"
    )
    post_eval_hooks: Optional[list[Union[PostEvalHookConfig, str]]] = Field(
        default=None, description="Post-evaluation hooks"
    )
    process_reasoning_traces: Optional[bool] = Field(
        default=None, description="Strip reasoning tokens and collect stats"
    )
    use_reasoning: Optional[bool] = Field(
        default=None,
        description="Enable reasoning mode (alias for process_reasoning_traces)",
    )
    use_system_prompt: Optional[bool] = Field(
        default=None, description="Use custom system prompt"
    )
    custom_system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt content"
    )
    use_response_logging: Optional[bool] = Field(
        default=None, description="Enable response logging"
    )
    max_logged_responses: Optional[int] = Field(
        default=None, description="Maximum responses to log"
    )
    use_request_logging: Optional[bool] = Field(
        default=None, description="Enable request logging"
    )
    max_logged_requests: Optional[int] = Field(
        default=None, description="Maximum requests to log"
    )
    use_caching: Optional[bool] = Field(
        default=None, description="Enable response caching"
    )
    tracking_requests_stats: Optional[bool] = Field(
        default=None, description="Track request statistics"
    )
    log_failed_requests: Optional[bool] = Field(
        default=None, description="Log failed requests"
    )
    params_to_add: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional parameters to add to requests (e.g., chat_template_kwargs)",
    )
    params_to_remove: Optional[list[str]] = Field(
        default=None, description="Parameters to remove from requests"
    )
    params_to_rename: Optional[dict[str, str]] = Field(
        default=None, description="Parameters to rename in requests"
    )


class ApiEndpointConfig(StrictModel):
    """API endpoint configuration.

    Matches ApiEndpoint from nemo_evaluator.api.api_dataclasses.
    """

    api_key: Optional[str] = Field(
        default=None, description="[DEPRECATED] Use api_key_name"
    )
    api_key_name: Optional[str] = Field(
        default=None, description="Environment variable name for API key"
    )
    model_id: Optional[str] = Field(default=None, description="Name of the model")
    stream: Optional[bool] = Field(
        default=None, description="Whether responses should be streamed"
    )
    type: Optional[str] = Field(
        default=None,
        description="Endpoint type (undefined/chat/completions/vlm/embedding)",
    )
    url: Optional[str] = Field(default=None, description="URL of the model endpoint")
    adapter_config: Optional[AdapterConfig] = Field(
        default=None, description="Adapter configuration"
    )


class TargetConfig(StrictModel):
    """Target configuration.

    Matches EvaluationTarget from nemo_evaluator.api.api_dataclasses.
    """

    api_endpoint: Optional[ApiEndpointConfig] = Field(
        default=None, description="API endpoint to be used for evaluation"
    )


class JudgeConfig(FlexibleModel):
    """Judge configuration for evaluation tasks."""

    model_id: Optional[str] = Field(default=None, description="Judge model ID")
    model: Optional[str] = Field(default=None, description="Judge model name")
    url: Optional[str] = Field(default=None, description="Judge endpoint URL")
    api_key: Optional[str] = Field(
        default=None, description="Judge API key env var name"
    )
    backend: Optional[str] = Field(default=None, description="Judge backend type")
    timeout: Optional[int] = Field(default=None, description="Judge request timeout")
    max_retries: Optional[int] = Field(default=None, description="Max retry attempts")
    temperature: Optional[float] = Field(default=None, description="Judge temperature")
    top_p: Optional[float] = Field(default=None, description="Judge top_p")
    max_tokens: Optional[int] = Field(default=None, description="Judge max tokens")
    parallelism: Optional[int] = Field(default=None, description="Judge parallelism")


class ParamsExtraConfig(FlexibleModel):
    """Extra parameters for evaluation (task-specific)."""

    n_samples: Optional[int] = Field(default=None, description="Number of samples")
    tokenizer: Optional[str] = Field(default=None, description="Tokenizer path or name")
    tokenizer_backend: Optional[str] = Field(
        default=None, description="Tokenizer backend (huggingface, tiktoken)"
    )
    judge: Optional[JudgeConfig] = Field(
        default=None, description="Judge configuration"
    )
    probes: Optional[str] = Field(
        default=None, description="Probes specification (for garak)"
    )


class ParamsConfig(StrictModel):
    """Parameters configuration for nemo_evaluator_config.

    Matches ConfigParams from nemo_evaluator.api.api_dataclasses.
    """

    limit_samples: Optional[Union[int, float]] = Field(
        default=None, description="Limit number of evaluation samples"
    )
    max_new_tokens: Optional[int] = Field(
        default=None, description="Max tokens to generate"
    )
    max_retries: Optional[int] = Field(
        default=None, description="Number of REST request retries"
    )
    parallelism: Optional[int] = Field(
        default=None, description="Parallelism to be used"
    )
    task: Optional[str] = Field(default=None, description="Name of the task")
    temperature: Optional[float] = Field(
        default=None, description="Float value between 0 and 1 for sampling temperature"
    )
    request_timeout: Optional[int] = Field(
        default=None, description="REST response timeout in seconds"
    )
    top_p: Optional[float] = Field(
        default=None, description="Float value between 0 and 1 for nucleus sampling"
    )
    extra: Optional[ParamsExtraConfig] = Field(
        default=None, description="Framework specific parameters"
    )
    # Additional fields used in NEL configs but not in core ConfigParams
    max_tokens: Optional[int] = Field(
        default=None, description="Alias for max_new_tokens (used by some configs)"
    )
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter")


class NemoEvaluatorInnerConfig(StrictModel):
    """Inner config section of nemo_evaluator_config."""

    params: Optional[ParamsConfig] = Field(
        default=None, description="Evaluation parameters"
    )
    extra: Optional[ParamsExtraConfig] = Field(
        default=None, description="Extra parameters (alternative location)"
    )


class NemoEvaluatorTargetConfig(StrictModel):
    """Target section within nemo_evaluator_config."""

    api_endpoint: Optional[ApiEndpointConfig] = Field(
        default=None, description="API endpoint configuration"
    )


class NemoEvaluatorConfig(StrictModel):
    """NeMo Evaluator configuration block."""

    config: Optional[NemoEvaluatorInnerConfig] = Field(
        default=None, description="Config section with params"
    )
    target: Optional[NemoEvaluatorTargetConfig] = Field(
        default=None, description="Target configuration"
    )


class TaskConfig(FlexibleModel):
    """Individual evaluation task configuration."""

    name: str = Field(description="Task name (e.g., ifeval, gsm8k, mmlu)")
    nemo_evaluator_config: Optional[NemoEvaluatorConfig] = Field(
        default=None, description="Task-specific evaluator config"
    )
    env_vars: Optional[dict[str, str]] = Field(
        default=None, description="Task-specific environment variables"
    )
    pre_cmd: Optional[str] = Field(default=None, description="Pre-command to run")
    container: Optional[str] = Field(default=None, description="Container override")
    dataset_dir: Optional[str] = Field(default=None, description="Dataset directory")
    dataset_mount_path: Optional[str] = Field(
        default=None, description="Dataset mount path"
    )


class EvaluationSectionConfig(FlexibleModel):
    """Evaluation section configuration."""

    nemo_evaluator_config: Optional[NemoEvaluatorConfig] = Field(
        default=None, description="Global evaluator config for all tasks"
    )
    tasks: list[TaskConfig] = Field(
        default_factory=list, description="List of evaluation tasks"
    )
    env_vars: Optional[dict[str, str]] = Field(
        default=None, description="Environment variables for all evaluation tasks"
    )
    pre_cmd: Optional[str] = Field(default=None, description="Pre-command to run")


# =============================================================================
# Export Configs
# =============================================================================


class MLflowExportConfig(StrictModel):
    """MLflow export configuration."""

    tracking_uri: str = Field(description="MLflow server URL (http://hostname:port)")
    experiment_name: Optional[str] = Field(
        default=None, description="MLflow experiment name"
    )
    description: Optional[str] = Field(default=None, description="Run description")
    log_metrics: Optional[list[str]] = Field(
        default=None, description="Subset of metrics to log"
    )
    log_logs: Optional[bool] = Field(default=None, description="Log evaluation logs")
    log_config_params: Optional[bool] = Field(
        default=None, description="Log flattened config as MLflow params"
    )
    only_required: Optional[bool] = Field(
        default=None, description="Only log required metrics"
    )
    tags: Optional[dict[str, str]] = Field(
        default=None, description="MLflow tags (key-value pairs)"
    )
    extra_metadata: Optional[dict[str, str]] = Field(
        default=None, description="Additional metadata for log_params"
    )


class WandbExportConfig(FlexibleModel):
    """Weights & Biases export configuration."""

    project: Optional[str] = Field(default=None, description="W&B project name")
    entity: Optional[str] = Field(default=None, description="W&B entity (team/user)")
    name: Optional[str] = Field(default=None, description="Run name")
    tags: Optional[list[str]] = Field(default=None, description="Run tags")
    notes: Optional[str] = Field(default=None, description="Run notes")


class ExportConfig(FlexibleModel):
    """Export configuration section."""

    mlflow: Optional[MLflowExportConfig] = Field(
        default=None, description="MLflow export configuration"
    )
    wandb: Optional[WandbExportConfig] = Field(
        default=None, description="W&B export configuration"
    )


# =============================================================================
# Root Config
# =============================================================================


class NELConfig(StrictModel):
    """
    Root NeMo Evaluator Launcher configuration.

    This model validates the full structure of an NEL configuration file.
    Uses extra="forbid" to catch unknown top-level fields.
    """

    model_config = ConfigDict(extra="forbid")

    # Hydra-specific fields (allowed but not required)
    defaults: Optional[list[Any]] = Field(
        default=None, description="Hydra defaults (processed before validation)"
    )
    user_config_path: Optional[str] = Field(
        default=None, description="Path to user config (added by RunConfig.from_hydra)"
    )

    # Optional name field
    name: Optional[str] = Field(default=None, description="Configuration name")

    # Main configuration sections
    execution: ExecutionConfig = Field(description="Execution configuration")
    deployment: DeploymentConfig = Field(description="Deployment configuration")
    target: Optional[TargetConfig] = Field(
        default=None, description="Target endpoint configuration"
    )
    evaluation: Optional[Union[EvaluationSectionConfig, list]] = Field(
        default=None, description="Evaluation tasks configuration"
    )
    export: Optional[ExportConfig] = Field(
        default=None, description="Export configuration"
    )

    @field_validator("evaluation", mode="before")
    @classmethod
    def handle_empty_evaluation(cls, v):
        """Handle empty evaluation list (default config)."""
        if v == [] or v is None:
            return None
        return v


# =============================================================================
# Config Resolution and Validation
# =============================================================================


def resolve_config(
    config_path: str, hydra_overrides: list[str] | None = None
) -> DictConfig:
    """Resolve config using Hydra, same as NEL launcher.

    Args:
        config_path: Path to the config file
        hydra_overrides: Optional list of Hydra overrides (e.g., ["execution.output_dir=/tmp"])

    Returns:
        Resolved DictConfig
    """
    GlobalHydra.instance().clear()

    try:
        # Reuse NEL's config resolution logic
        from nemo_evaluator_launcher.api.types import RunConfig

        cfg = RunConfig.from_hydra(
            config=config_path, hydra_overrides=hydra_overrides or []
        )
    finally:
        GlobalHydra.instance().clear()

    return cfg


def format_validation_error(err: dict) -> str:
    """Format a single validation error into a readable message."""
    path = ".".join(str(p) for p in err["loc"])
    msg = err["msg"]
    error_type = err["type"]

    if error_type == "extra_forbidden":
        return (
            f"Unexpected field '{path}': This field is not recognized. Check for typos."
        )
    elif error_type == "missing":
        return f"Missing required field '{path}': {msg}"
    elif error_type == "literal_error":
        return f"Invalid value at '{path}': {msg}"
    elif error_type == "union_tag_not_found":
        return f"Missing discriminator at '{path}': {msg}"
    else:
        return f"Invalid value at '{path}': {msg}"


def _strip_missing_values(obj: Any) -> Any:
    """Recursively strip Hydra's ??? placeholder values from config.

    This allows Pydantic to report 'missing required field' errors naturally
    instead of accepting '???' as a valid string value.
    """
    if isinstance(obj, dict):
        return {k: _strip_missing_values(v) for k, v in obj.items() if v != "???"}
    elif isinstance(obj, list):
        return [_strip_missing_values(item) for item in obj if item != "???"]
    else:
        return obj


def check_warnings(config_dict: dict) -> list[str]:
    """Check for potential misconfigurations that warrant warnings."""
    warnings = []

    # Check for interceptors list without 'endpoint' interceptor
    # This is valid but the eval won't make any API calls
    try:
        target = config_dict.get("target", {})
        api_endpoint = target.get("api_endpoint", {})
        adapter_config = api_endpoint.get("adapter_config", {})
        interceptors = adapter_config.get("interceptors")

        if interceptors is not None and isinstance(interceptors, list):
            interceptor_names = [
                i.get("name") for i in interceptors if isinstance(i, dict)
            ]
            if "endpoint" not in interceptor_names:
                warnings.append(
                    "Interceptors list is defined but missing 'endpoint' interceptor. "
                    "Without it, no API calls will be made during evaluation."
                )
    except (KeyError, TypeError, AttributeError):
        pass  # Malformed config will be caught by validation

    return warnings


def validate_config(cfg: DictConfig) -> tuple[bool, list[str], list[str]]:
    """Validate resolved config against Pydantic models.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []

    # Convert to dict for Pydantic
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Strip ??? values so Pydantic reports "missing required field" naturally
    config_dict = _strip_missing_values(config_dict)

    # Check for warnings before validation
    warnings = check_warnings(config_dict)

    try:
        NELConfig(**config_dict)
        return True, [], warnings
    except ValidationError as e:
        for err in e.errors():
            errors.append(format_validation_error(err))
        return False, errors, warnings


def main():
    """Main entry point for config validation."""
    if len(sys.argv) < 2:
        print("Usage: python verify_config.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Validating: {config_path}")

    try:
        cfg = resolve_config(config_path)
        valid, errors, warnings = validate_config(cfg)

        if valid:
            print("✓ Configuration is valid!")
            if warnings:
                for warn in warnings:
                    print(f"  ⚠ Warning: {warn}")
            sys.exit(0)
        else:
            print("✗ Configuration has errors:")
            for err in errors:
                print(f"  - {err}")
            if warnings:
                print("Warnings:")
                for warn in warnings:
                    print(f"  ⚠ {warn}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to process config: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
