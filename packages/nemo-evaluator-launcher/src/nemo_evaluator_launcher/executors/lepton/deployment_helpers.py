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
"""Lepton deployment helper functions for nemo-evaluator-launcher.

Handles Lepton endpoint creation, management, and health checks.
"""

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Import lepton dependencies
from omegaconf import DictConfig

from nemo_evaluator_launcher.common.helpers import _str_to_echo_command
from nemo_evaluator_launcher.common.logging_utils import logger


def deep_merge(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def replace_placeholders(data: Any, replacements: Dict[str, str]) -> Any:
    """Replace placeholders in the data structure."""

    def replace_in_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: replace_in_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_in_obj(item) for item in obj]
        elif isinstance(obj, str):
            result = obj
            for placeholder, value in replacements.items():
                result = result.replace(f"{{{{{placeholder}}}}}", value)
            return result
        else:
            return obj

    return replace_in_obj(data)


def generate_lepton_spec(cfg: DictConfig) -> Dict[str, Any]:
    """Generate a Lepton endpoint specification from nemo-evaluator-launcher configuration.

    This function creates a layered configuration by merging:
    1. Platform defaults (from execution.lepton_platform.platform_defaults)
    2. Environment settings (from execution.lepton_platform)
    3. Inference engine config (from deployment.* - vllm/sglang settings)
    4. Lepton platform config (from deployment.lepton_config - Lepton-specific settings)

    Args:
        cfg: The nemo-evaluator-launcher configuration object containing all settings.

    Returns:
        Dict containing the Lepton endpoint specification.
    """

    # Step 1: Start with platform defaults from execution config
    platform_defaults = {}
    if hasattr(cfg, "execution") and hasattr(cfg.execution, "lepton_platform"):
        deployment_config = cfg.execution.lepton_platform.get("deployment", {})
        platform_defaults = deployment_config.get("platform_defaults", {})

    base_config = deep_merge({}, platform_defaults)

    # Step 2: Apply deployment-specific settings from execution config
    if hasattr(cfg, "execution") and hasattr(cfg.execution, "lepton_platform"):
        lepton_platform = cfg.execution.lepton_platform
        deployment_config = lepton_platform.get("deployment", {})

        # Add deployment node group as affinity constraint
        deployment_node_group = deployment_config.get("node_group")
        if deployment_node_group:
            if not base_config.get("resource_requirement"):
                base_config["resource_requirement"] = {}
            base_config["resource_requirement"]["affinity"] = {
                "allowed_dedicated_node_groups": [deployment_node_group]
            }

        # Add queue config from platform defaults
        platform_defaults = deployment_config.get("platform_defaults", {})
        if platform_defaults.get("queue_config"):
            base_config["queue_config"] = platform_defaults.get("queue_config")

    # Step 3: Get Lepton-specific config from deployment.lepton_config
    if not hasattr(cfg.deployment, "lepton_config"):
        raise ValueError(
            "deployment.lepton_config is required when using Lepton executor"
        )

    lepton_config = cfg.deployment.lepton_config

    # Step 4: Convert inference engine config to container spec
    container_spec = _create_inference_container_spec(cfg.deployment)

    # Step 5: Apply Lepton platform deployment configurations
    deployment_config = {
        "resource_requirement": {
            **base_config.get("resource_requirement", {}),
            "resource_shape": lepton_config.resource_shape,
            "min_replicas": lepton_config.min_replicas,
            "max_replicas": lepton_config.max_replicas,
        },
        "auto_scaler": lepton_config.auto_scaler,
        "container": container_spec,
        "envs": [],
    }

    # Add health check configuration if provided
    if hasattr(lepton_config, "health") and lepton_config.health:
        deployment_config["health"] = lepton_config.health
    # Merge deployment config into base config
    final_config = deep_merge(base_config, deployment_config)

    # Step 6: Add environment variables from lepton_config
    if hasattr(lepton_config, "envs") and lepton_config.envs:
        from omegaconf import DictConfig

        for key, value in lepton_config.envs.items():
            env_var: Dict[str, Any] = {"name": key}

            # Support both direct values and secret references
            if isinstance(value, (dict, DictConfig)) and "value_from" in value:
                # Secret reference: {value_from: {secret_name_ref: "secret_name"}}
                env_var["value_from"] = dict(value["value_from"])
            else:
                # Direct value: "direct_value"
                env_var["value"] = str(value)

            final_config["envs"].append(env_var)

    # Step 6b: Auto-populate environment variables from deployment parameters
    _add_deployment_derived_envs(final_config["envs"], cfg.deployment)

    # Step 7: Add mounts with intelligent path construction
    if hasattr(lepton_config, "mounts") and lepton_config.mounts.enabled:
        # Get storage source from task config mounts (since mounts are shared between tasks and deployments)
        storage_source = "node-nfs:lepton-shared-fs"  # default
        if hasattr(cfg, "execution") and hasattr(cfg.execution, "lepton_platform"):
            task_config = cfg.execution.lepton_platform.get("tasks", {})
            task_mounts = task_config.get("mounts", [])
            if task_mounts:
                storage_source = task_mounts[0].get("from", storage_source)

        final_config["mounts"] = [
            {
                "path": lepton_config.mounts.cache_path,
                "from": storage_source,
                "mount_path": lepton_config.mounts.mount_path,
                "mount_options": {},
            }
        ]

    # Step 8: Extract image_pull_secrets to top level (required by Lepton API)
    if "image_pull_secrets" in final_config:
        image_pull_secrets = final_config["image_pull_secrets"]
        # Convert OmegaConf ListConfig to regular Python list
        from omegaconf import ListConfig

        if isinstance(image_pull_secrets, (list, ListConfig)):
            final_config["image_pull_secrets"] = list(image_pull_secrets)
        else:
            # Remove invalid image_pull_secrets
            final_config.pop("image_pull_secrets", None)

    # Step 9: Add API tokens if provided (supports both single and multiple tokens)
    if hasattr(lepton_config, "api_tokens") and lepton_config.api_tokens:
        from omegaconf import DictConfig

        api_tokens_list = []

        for token_config in lepton_config.api_tokens:
            token_var: Dict[str, Any] = {}

            # Support both direct values and secret references
            if isinstance(token_config, (dict, DictConfig)):
                if "value" in token_config:
                    # Direct value: {value: "token_string"}
                    token_var["value"] = str(token_config["value"])
                elif "value_from" in token_config:
                    # Secret reference: {value_from: {secret_name_ref: "secret_name"}}
                    token_var["value_from"] = dict(token_config["value_from"])
            else:
                # Simple string value
                token_var["value"] = str(token_config)

            api_tokens_list.append(token_var)

        final_config["api_tokens"] = api_tokens_list

    # Backward compatibility: support legacy single api_token
    elif hasattr(lepton_config, "api_token") and lepton_config.api_token:
        final_config["api_tokens"] = [{"value": lepton_config.api_token}]

    # Step 10: Replace placeholders
    replacements = {
        "MODEL_CACHE_NAME": _generate_model_cache_name(cfg.deployment.image)
    }
    final_config_with_replacements: Dict[str, Any] = replace_placeholders(
        final_config, replacements
    )

    return final_config_with_replacements


def _create_inference_container_spec(deployment_cfg: DictConfig) -> Dict[str, Any]:
    """Create container specification from inference engine config (vLLM/SGLang/NIM).

    Args:
        deployment_cfg: Deployment configuration containing vLLM/SGLang/NIM settings.

    Returns:
        Container specification for Lepton.
    """
    # Extract pre_cmd from deployment_cfg
    pre_cmd: str = deployment_cfg.get("pre_cmd") or ""
    container_spec = {
        "image": deployment_cfg.image,
        "ports": [{"container_port": deployment_cfg.port}],
    }

    # Generate command based on deployment type
    if deployment_cfg.type == "vllm":
        # Convert vLLM command template to actual command
        command_parts = [
            "vllm",
            "serve",
            deployment_cfg.checkpoint_path,
            f"--tensor-parallel-size={deployment_cfg.tensor_parallel_size}",
            f"--pipeline-parallel-size={deployment_cfg.pipeline_parallel_size}",
            f"--data-parallel-size={deployment_cfg.data_parallel_size}",
            f"--port={deployment_cfg.port}",
            f"--served-model-name={deployment_cfg.served_model_name}",
        ]

        # Add extra args if provided
        if hasattr(deployment_cfg, "extra_args") and deployment_cfg.extra_args:
            command_parts.extend(deployment_cfg.extra_args.split())

        # Wrap with pre_cmd if provided
        if pre_cmd:
            create_pre_script_cmd = _str_to_echo_command(
                pre_cmd, filename="deployment_pre_cmd.sh"
            )
            original_cmd = " ".join(shlex.quote(str(c)) for c in command_parts)
            command_parts = [
                "/bin/bash",
                "-c",
                f"{create_pre_script_cmd.cmd} && source deployment_pre_cmd.sh && exec {original_cmd}",
            ]

        container_spec["command"] = command_parts

    elif deployment_cfg.type == "sglang":
        # Convert SGLang command template to actual command
        command_parts = [
            "python3",
            "-m",
            "sglang.launch_server",
            f"--model-path={deployment_cfg.checkpoint_path}",
            "--host=0.0.0.0",
            f"--port={deployment_cfg.port}",
            f"--served-model-name={deployment_cfg.served_model_name}",
            f"--tp={deployment_cfg.tensor_parallel_size}",
            f"--dp={deployment_cfg.data_parallel_size}",
        ]

        # Add extra args if provided
        if hasattr(deployment_cfg, "extra_args") and deployment_cfg.extra_args:
            command_parts.extend(deployment_cfg.extra_args.split())

        # Wrap with pre_cmd if provided
        if pre_cmd:
            create_pre_script_cmd = _str_to_echo_command(
                pre_cmd, filename="deployment_pre_cmd.sh"
            )
            original_cmd = " ".join(shlex.quote(str(c)) for c in command_parts)
            command_parts = [
                "/bin/bash",
                "-c",
                f"{create_pre_script_cmd.cmd} && source deployment_pre_cmd.sh && exec {original_cmd}",
            ]

        container_spec["command"] = command_parts

    elif deployment_cfg.type == "nim":
        # NIM containers use their default entrypoint - no custom command needed
        # Configuration is handled via environment variables
        # pre_cmd is not supported for NIM deployments
        if pre_cmd:
            logger.error(
                "pre_cmd is not supported for NIM deployments",
                deployment_type="nim",
                pre_cmd=pre_cmd,
            )
            raise ValueError("pre_cmd is not supported for NIM deployments")

    return container_spec


def _add_deployment_derived_envs(envs_list: list, deployment_cfg: DictConfig) -> None:
    """Add environment variables derived from deployment configuration.

    Args:
        envs_list: List to append environment variables to.
        deployment_cfg: Deployment configuration to derive from.
    """
    deployment_type = deployment_cfg.type

    # Common environment variables for all deployment types
    if (
        hasattr(deployment_cfg, "served_model_name")
        and deployment_cfg.served_model_name
    ):
        envs_list.append(
            {"name": "SERVED_MODEL_NAME", "value": deployment_cfg.served_model_name}
        )

    if hasattr(deployment_cfg, "port") and deployment_cfg.port:
        envs_list.append({"name": "MODEL_PORT", "value": str(deployment_cfg.port)})

    # Deployment-specific environment variables
    if deployment_type == "vllm":
        if (
            hasattr(deployment_cfg, "checkpoint_path")
            and deployment_cfg.checkpoint_path
        ):
            envs_list.append(
                {"name": "MODEL_PATH", "value": deployment_cfg.checkpoint_path}
            )
        if (
            hasattr(deployment_cfg, "tensor_parallel_size")
            and deployment_cfg.tensor_parallel_size
        ):
            envs_list.append(
                {
                    "name": "TENSOR_PARALLEL_SIZE",
                    "value": str(deployment_cfg.tensor_parallel_size),
                }
            )

    elif deployment_type == "sglang":
        if (
            hasattr(deployment_cfg, "checkpoint_path")
            and deployment_cfg.checkpoint_path
        ):
            envs_list.append(
                {"name": "MODEL_PATH", "value": deployment_cfg.checkpoint_path}
            )
        if (
            hasattr(deployment_cfg, "tensor_parallel_size")
            and deployment_cfg.tensor_parallel_size
        ):
            envs_list.append(
                {
                    "name": "TENSOR_PARALLEL_SIZE",
                    "value": str(deployment_cfg.tensor_parallel_size),
                }
            )

    elif deployment_type == "nim":
        # NIM-specific derived environment variables
        if (
            hasattr(deployment_cfg, "served_model_name")
            and deployment_cfg.served_model_name
        ):
            envs_list.append(
                {"name": "NIM_MODEL_NAME", "value": deployment_cfg.served_model_name}
            )


def _generate_model_cache_name(image: str) -> str:
    """Generate a cache directory name from the container image.

    Args:
        image: Container image string like 'nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6'

    Returns:
        Clean cache name like 'llama-3-1-8b-instruct'
    """
    # Extract model name from image path
    if "/" in image:
        model_part = image.split("/")[-1]  # Get 'llama-3.1-8b-instruct:1.8.6'
    else:
        model_part = image

    # Remove version tag
    if ":" in model_part:
        model_part = model_part.split(":")[0]  # Get 'llama-3.1-8b-instruct'

    # Replace dots with dashes for filesystem compatibility
    return model_part.replace(".", "-")


def create_lepton_endpoint(cfg: DictConfig, endpoint_name: str) -> bool:
    """Create a Lepton endpoint using the lep CLI.

    Args:
        cfg: The nemo-evaluator-launcher configuration object.
        endpoint_name: Name for the endpoint.

    Returns:
        True if endpoint creation succeeded, False otherwise.
    """
    spec = generate_lepton_spec(cfg)

    # Convert OmegaConf objects to regular Python objects for JSON serialization
    from omegaconf import DictConfig, ListConfig

    def convert_to_json_serializable(obj: Any) -> Any:
        """Recursively convert OmegaConf objects to regular Python objects."""
        if isinstance(obj, (DictConfig, dict)):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (ListConfig, list)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    json_spec = convert_to_json_serializable(spec)

    # Write spec to temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(json_spec, f, indent=2)
        spec_file = f.name

    try:
        # Create endpoint using lep CLI
        result = subprocess.run(
            ["lep", "endpoint", "create", "--file", spec_file, "--name", endpoint_name],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print(f"✅ Successfully created Lepton endpoint: {endpoint_name}")
            return True
        else:
            error_msg = result.stderr.strip() if result.stderr else ""
            output_msg = result.stdout.strip() if result.stdout else ""
            print(
                f"✗ Failed to create Lepton endpoint | Endpoint: {endpoint_name} | Return code: {result.returncode}"
            )
            if error_msg:
                print(f"   stderr: {error_msg}")
            if output_msg:
                print(f"   stdout: {output_msg}")
            return False

    except subprocess.TimeoutExpired as e:
        print(
            f"✗ Timeout creating Lepton endpoint | Endpoint: {endpoint_name} | Timeout: 300s"
        )
        if hasattr(e, "stderr") and e.stderr:
            print(f"   stderr: {e.stderr}")
        if hasattr(e, "stdout") and e.stdout:
            print(f"   stdout: {e.stdout}")
        return False
    except subprocess.CalledProcessError as e:
        print(
            f"✗ Error creating Lepton endpoint | Endpoint: {endpoint_name} | Error: {e}"
        )
        if hasattr(e, "stderr") and e.stderr:
            print(f"   stderr: {e.stderr}")
        if hasattr(e, "stdout") and e.stdout:
            print(f"   stdout: {e.stdout}")
        return False
    finally:
        # Clean up temporary file
        Path(spec_file).unlink(missing_ok=True)


def delete_lepton_endpoint(endpoint_name: str) -> bool:
    """Delete a Lepton endpoint.

    Args:
        endpoint_name: Name of the endpoint to delete.

    Returns:
        True if deletion succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            ["lep", "endpoint", "remove", "--name", endpoint_name],
            capture_output=True,
            text=True,
            timeout=60,
        )

        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def get_lepton_endpoint_status(endpoint_name: str) -> Optional[dict[str, Any]]:
    """Get the status of a Lepton endpoint.

    Args:
        endpoint_name: Name of the endpoint.

    Returns:
        Status dict if endpoint exists, None otherwise. See
        https://github.com/leptonai/leptonai/blob/7de93b95357126da1e86fa99f54f9a769d5d2646/leptonai/api/v1/types/deployment.py#L338
        for the definition.
    """
    try:
        # TODO(agronskiy): why not use Python API?
        cmd = ["lep", "endpoint", "get", "--name", endpoint_name]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None

        endpoint_info = json.loads(result.stdout)
        status = endpoint_info.get("status", {})
        if isinstance(status, dict):
            return status
        logger.error(
            "Result of running lep command returne non-dict status",
            cmd=cmd,
            status=status,
        )
        return None

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        return None


def wait_for_lepton_endpoint_ready(endpoint_name: str, timeout: int = 600) -> bool:
    """Wait for a Lepton endpoint to become ready.

    Args:
        endpoint_name: Name of the endpoint.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if endpoint becomes ready, False if timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_lepton_endpoint_status(endpoint_name)

        # `get_lepton_endpoint_status` might return `None` if
        # e.g. there was a network error, see definition.
        if status is not None:
            state = status.get("state", "").lower()
            if state == "ready":
                logger.info(
                    "Lepton endpoint is ready",
                    endpoint_name=endpoint_name,
                )

                return True
            elif state in ["failed", "error"]:
                return False

        logger.debug(
            "Waiting for lepton endpoint",
            endpoint_name=endpoint_name,
            timeout=timeout,
            time_delta=time.time() - start_time,
            curr_status=status,
        )
        time.sleep(10)

    logger.error(
        "Timeout waiting for lepton endpoint",
        endpoint_name=endpoint_name,
        timeout=timeout,
    )
    return False


def get_lepton_endpoint_url(endpoint_name: str) -> Optional[str]:
    """Get the URL of a Lepton endpoint.

    Args:
        endpoint_name: Name of the endpoint.

    Returns:
        Endpoint URL if available, None otherwise.
    """
    try:
        result = subprocess.run(
            ["lep", "endpoint", "get", "--name", endpoint_name],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            endpoint_info = json.loads(result.stdout)
            status = endpoint_info.get("status", {})
            endpoint = status.get("endpoint", {})
            external_endpoint = endpoint.get("external_endpoint")
            # Ensure we return a proper string type or None
            if isinstance(external_endpoint, str):
                return external_endpoint
            else:
                return None
        else:
            return None
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        return None
