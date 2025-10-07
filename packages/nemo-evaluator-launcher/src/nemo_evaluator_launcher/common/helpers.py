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
import base64
import copy
import datetime
from typing import Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.logging_utils import logger


def _yaml_to_echo_command(yaml_str: str, filename: str = "config_ef.yaml") -> str:
    yaml_str_b64 = base64.b64encode(yaml_str.encode("utf-8")).decode("utf-8")
    return f'echo "{yaml_str_b64}" | base64 -d > {filename}'


def get_eval_factory_config(
    cfg: DictConfig, user_task_config: DictConfig, task_definition: dict
) -> dict:
    """Extract config fields for eval factory.

    This function extracts the config field similar to how overrides are handled.
    """
    # Extract config fields similar to overrides - convert to basic Python types first
    cfg_config = cfg.evaluation.get("config", {})
    user_config = user_task_config.get("config", {})

    # Convert OmegaConf objects to basic Python types
    if cfg_config:
        cfg_config = OmegaConf.to_container(cfg_config, resolve=True)
    if user_config:
        user_config = OmegaConf.to_container(user_config, resolve=True)

    # Merge the configs
    config_fields = copy.deepcopy(cfg_config or {})
    config_fields.update(user_config or {})

    return config_fields


def get_eval_factory_command(
    cfg: DictConfig, user_task_config: DictConfig, task_definition: dict
) -> str:
    config_fields = get_eval_factory_config(cfg, user_task_config, task_definition)

    overrides = copy.deepcopy(dict(cfg.evaluation.get("overrides", {})))
    overrides.update(dict(user_task_config.get("overrides", {})))
    # NOTE(dfridman): Temporary fix to make sure that the overrides arg is not split into multiple lines.
    # Consider passing a JSON object on Eval Factory side
    overrides = {
        k: (v.strip("\n") if isinstance(v, str) else v) for k, v in overrides.items()
    }
    overrides_str = ",".join([f"{k}={v}" for k, v in overrides.items()])
    model_url = get_endpoint_url(cfg, user_task_config, task_definition)

    model_id = get_served_model_name(cfg)
    model_type = task_definition["endpoint_type"]
    eval_type = task_definition["task"]

    create_file_cmd = _yaml_to_echo_command(
        yaml.safe_dump(config_fields), "config_ef.yaml"
    )
    eval_command = f"""cmd=$([[ $(command -v nemo-evaluator) ]] && echo 'nemo-evaluator' || echo 'eval-factory') && $cmd run_eval --model_id {model_id} --model_type {model_type} --eval_type {eval_type} --model_url {model_url} --api_key_name API_KEY --output_dir /results --run_config config_ef.yaml"""

    if overrides:
        eval_command = f"{eval_command} --overrides {overrides_str}"

    return create_file_cmd + " && " + "cat config_ef.yaml && " + eval_command


def get_endpoint_url(
    cfg: DictConfig, user_task_config: DictConfig, task_definition: dict
) -> str:
    def apply_url_override(url: str) -> str:
        """Apply user URL override if provided."""
        override_url = user_task_config.get("overrides", {}).get(
            "config.target.api_endpoint.url"
        )
        return override_url if override_url is not None else url

    if cfg.deployment.type == "none":
        # For deployment: none, use target URL regardless of executor type
        if OmegaConf.is_missing(cfg.target.api_endpoint, "url"):
            raise ValueError(
                "API endpoint URL is not set. Add `target.api_endpoint.url` to your config "
                "OR override via CLI"
            )
        return apply_url_override(cfg.target.api_endpoint.url)

    elif (
        hasattr(cfg, "target")
        and hasattr(cfg.target, "api_endpoint")
        and hasattr(cfg.target.api_endpoint, "url")
        and not OmegaConf.is_missing(cfg.target.api_endpoint, "url")
    ):
        # For Lepton executor with dynamically set target URL
        return apply_url_override(cfg.target.api_endpoint.url)

    else:
        # Local executor - use localhost
        task_endpoint_type = task_definition["endpoint_type"]
        endpoint_uri = cfg.deployment.endpoints[task_endpoint_type]
        endpoint_url = f"http://127.0.0.1:{cfg.deployment.port}{endpoint_uri}"
        return endpoint_url


def get_health_url(cfg: DictConfig, endpoint_url: str) -> str:
    if cfg.deployment.type == "none":
        logger.warning("Using endpoint URL as health URL", will_be_used=endpoint_url)
        return endpoint_url  # TODO(public release) is using model url as health url OK?
    health_uri = cfg.deployment.endpoints["health"]
    health_url = f"http://127.0.0.1:{cfg.deployment.port}{health_uri}"
    return health_url


def get_served_model_name(cfg: DictConfig) -> str:
    if cfg.deployment.type == "none":
        return str(cfg.target.api_endpoint.model_id)
    else:
        return str(cfg.deployment.served_model_name)


def get_api_key_name(cfg: DictConfig) -> str | None:
    res = cfg.get("target", {}).get("api_endpoint", {}).get("api_key_name", None)
    return str(res) if res else None


def get_timestamp_string(include_microseconds: bool = True) -> str:
    """Get timestamp in format YYYYmmdd_HHMMSS_ffffff."""
    dt = datetime.datetime.now()
    fmt = "%Y%m%d_%H%M%S"
    if include_microseconds:
        fmt += "_%f"
    dts = datetime.datetime.strftime(dt, fmt)
    return dts


def get_eval_factory_dataset_size_from_run_config(run_config: dict) -> Optional[int]:
    config = run_config["config"]
    limit_samples = config["params"].get("limit_samples", None)
    if limit_samples is not None:
        return int(limit_samples)

    # TODO(dfridman): Move `dataset_size` values to the corresponding `framework.yaml` in Eval Factory.
    dataset_sizes = {
        ("lm-evaluation-harness", "ifeval"): 541,
        ("simple_evals", "gpqa_diamond"): 198,
        ("simple_evals", "gpqa_diamond_nemo"): 198,
        ("simple_evals", "AA_math_test_500"): 500,
        ("simple_evals", "math_test_500_nemo"): 500,
        ("simple_evals", "aime_2024_nemo"): 30,
        ("simple_evals", "AA_AIME_2024"): 30,
        ("simple_evals", "aime_2025_nemo"): 30,
        ("simple_evals", "AIME_2025"): 30,
        ("simple_evals", "humaneval"): 164,
        ("simple_evals", "mmlu"): 14042,
        ("simple_evals", "mmlu_pro"): 12032,
        ("bigcode-evaluation-harness", "mbpp"): 500,
        ("bigcode-evaluation-harness", "humaneval"): 164,
        ("livecodebench", "livecodebench_0724_0125"): 315,
        ("livecodebench", "livecodebench_0824_0225"): 279,
        ("hle", "hle"): 2684,
        ("scicode", "aa_scicode"): 338,
    }
    dataset_size = dataset_sizes.get((run_config["framework_name"], config["type"]))
    if dataset_size is None:
        return None
    else:
        dataset_size = int(dataset_size)

    downsampling_ratio = (
        config["params"].get("extra", {}).get("downsampling_ratio", None)
    )
    if downsampling_ratio is not None:
        dataset_size = int(round(dataset_size * downsampling_ratio))

    n_samples = int(config["params"].get("extra", {}).get("n_samples", 1))
    dataset_size *= n_samples
    return dataset_size
