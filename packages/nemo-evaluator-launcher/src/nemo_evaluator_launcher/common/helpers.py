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
from dataclasses import dataclass
from typing import Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.cli.version import get_versions
from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass(frozen=True)
class CmdAndReadableComment:
    """See the comment to `_yaml_to_echo_command`."""

    # Actual command. Might include hard-to-debug elements such as base64-encoded
    # configs.
    cmd: str
    # A debuggale readable comment that can be passed along for accompanying
    # the actual command
    debug: str
    # Whether the content might be potentially unsafe. This is a flag useful for
    # downstream callers who want to raise exceptions e.g. when a script was
    # saved that would execute this command.
    is_potentially_unsafe: bool = False


def _str_to_echo_command(str_to_save: str, filename: str) -> CmdAndReadableComment:
    """Create a safe (see below) echo command saving a string to file.

    Safety in this context means the ability to pass such echo command through the
    `bash -c '...'` boundaries for example.

    Naturally, enconding with base64 creates debuggability issues. For that, the second
    output of the function is the string with bash comment signs prepended.
    """
    str_to_save_b64 = base64.b64encode(str_to_save.encode("utf-8")).decode("utf-8")
    debug_str = "\n".join(
        [f"# Contents of {filename}"] + ["# " + s for s in str_to_save.splitlines()]
    )
    return CmdAndReadableComment(
        cmd=f'echo "{str_to_save_b64}" | base64 -d > {filename}', debug=debug_str
    )


def _set_nested_optionally_overriding(
    d: dict, keys: list[str], val: object, *, override_if_exists: bool = False
):
    """Sets d[...keys....] = value, creating keys all the way"""
    temp = d
    for key in keys[:-1]:
        temp = temp.setdefault(key, {})
    if override_if_exists or keys[-1] not in temp:
        temp[keys[-1]] = val


def get_eval_factory_config(
    cfg: DictConfig,
    user_task_config: DictConfig,
) -> dict:
    """Extract config fields for eval factory.

    This function extracts the config field similar to how overrides are handled.

    Overrides will start to be deprecated (or not, but at least a warning will be logged).
    """

    if cfg.evaluation.get("overrides") or user_task_config.get("overrides"):
        # TODO(agronskiy): start removing overrides, test `test_start_deprecating_overrides`
        # will start failing soon.
        logger.warning(
            "We are deprecating using old-style dot-delimited overrides "
            "in favour of `nemo_evaluator_config` field. Please check "
            "the documentation."
        )

    logger.debug("Getting nemo evaluator merged config")
    # Extract config fields similar to overrides - convert to basic Python types first
    # Support both new and old format for backward compatibility
    cfg_config = cfg.evaluation.get("nemo_evaluator_config") or cfg.evaluation.get(
        "config", {}
    )
    user_config = user_task_config.get("nemo_evaluator_config") or user_task_config.get(
        "config", {}
    )

    # Convert OmegaConf objects to basic Python types
    if cfg_config:
        cfg_config = OmegaConf.to_container(cfg_config, resolve=True)
    if user_config:
        user_config = OmegaConf.to_container(user_config, resolve=True)

    # Merge the configs
    merged_nemo_evaluator_config: dict = OmegaConf.to_container(
        OmegaConf.merge(cfg_config, user_config)
    )

    logger.debug(
        "Merged nemo evaluator config, not final",
        source_global_cfg=cfg_config,
        source_task_config=user_config,
        result=merged_nemo_evaluator_config,
    )

    return merged_nemo_evaluator_config


def get_eval_factory_command(
    cfg: DictConfig,
    user_task_config: DictConfig,
    task_definition: dict,
) -> CmdAndReadableComment:
    # This gets the eval_factory_config merged from both top-level and task-level.
    merged_nemo_evaluator_config = get_eval_factory_config(
        cfg,
        user_task_config,
    )

    # We now prepare the config to be passed to `nemo-evaluator` command.
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["target", "api_endpoint", "url"],
        get_endpoint_url(
            cfg,
            merged_nemo_evaluator_config=merged_nemo_evaluator_config,
            endpoint_type=task_definition["endpoint_type"],
        ),
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["target", "api_endpoint", "model_id"],
        get_served_model_name(cfg),
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["target", "api_endpoint", "type"],
        task_definition["endpoint_type"],
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["config", "type"],
        task_definition["task"],
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["config", "output_dir"],
        "/results",
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["target", "api_endpoint", "api_key"],
        "API_KEY",
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        [
            "metadata",
            "launcher_resolved_config",
        ],
        OmegaConf.to_container(cfg, resolve=True),
    )
    _set_nested_optionally_overriding(
        merged_nemo_evaluator_config,
        ["metadata", "versioning"],
        get_versions(),
    )

    # Now get the pre_cmd either from `evaluation.pre_cmd` or task-level pre_cmd. Note the
    # order -- task level wins.
    pre_cmd: str = (
        user_task_config.get("pre_cmd") or cfg.evaluation.get("pre_cmd") or ""
    )

    is_potentially_unsafe = False
    if pre_cmd:
        logger.warning(
            "Found non-empty pre_cmd that might be a security risk if executed. "
            "Setting `is_potentially_unsafe` to `True`",
            pre_cmd=pre_cmd,
        )
        is_potentially_unsafe = True
        _set_nested_optionally_overriding(
            merged_nemo_evaluator_config,
            ["metadata", "pre_cmd"],
            pre_cmd,
        )

    create_pre_script_cmd = _str_to_echo_command(pre_cmd, filename="pre_cmd.sh")

    create_yaml_cmd = _str_to_echo_command(
        yaml.safe_dump(merged_nemo_evaluator_config), "config_ef.yaml"
    )

    # NOTE: we use `source` to allow tricks like exports etc (if needed) -- it runs in the same
    # shell as the command.
    eval_command = (
        "cmd=$(command -v nemo-evaluator >/dev/null 2>&1 && echo nemo-evaluator || echo eval-factory) "
        + "&& source pre_cmd.sh "
        + "&& $cmd run_eval --run_config config_ef.yaml"
    )

    # NOTE: see note and test about deprecating that.
    overrides = copy.deepcopy(dict(cfg.evaluation.get("overrides", {})))
    overrides.update(dict(user_task_config.get("overrides", {})))
    # NOTE(dfridman): Temporary fix to make sure that the overrides arg is not split into multiple lines.
    # Consider passing a JSON object on Eval Factory side
    overrides = {
        k: (v.strip("\n") if isinstance(v, str) else v) for k, v in overrides.items()
    }
    overrides_str = ",".join([f"{k}={v}" for k, v in overrides.items()])
    if overrides_str:
        eval_command = f"{eval_command} --overrides {overrides_str}"

    # We return both the command and the debugging base64-decoded strings, useful
    # for exposing when building scripts.
    return CmdAndReadableComment(
        cmd=create_pre_script_cmd.cmd
        + " && "
        + create_yaml_cmd.cmd
        + " && "
        + eval_command,
        debug=create_pre_script_cmd.debug + "\n\n" + create_yaml_cmd.debug,
        is_potentially_unsafe=is_potentially_unsafe,
    )


def get_endpoint_url(
    cfg: DictConfig,
    merged_nemo_evaluator_config: dict,
    endpoint_type: str,
) -> str:
    def apply_url_override(url: str) -> str:
        """Apply user URL override if provided."""
        nemo_evaluator_config_url = (
            merged_nemo_evaluator_config.get("target", {})
            .get("api_endpoint", {})
            .get("url", None)
        )

        if nemo_evaluator_config_url:
            return nemo_evaluator_config_url

        # Being deprecated, see `get_eval_factory_config` message.
        overrides_old_style_url = merged_nemo_evaluator_config.get("overrides", {}).get(
            "target.api_endpoint.url", None
        )
        if overrides_old_style_url:
            return overrides_old_style_url

        return url

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
        endpoint_uri = cfg.deployment.endpoints[endpoint_type]

        # Use HAProxy port if multiple_instances is enabled
        if cfg.deployment.get("multiple_instances", False):
            proxy_config = cfg.execution.get("proxy", {}).get("config", {})
            port = proxy_config.get("haproxy_port", 5009)
        else:
            port = cfg.deployment.port

        endpoint_url = f"http://127.0.0.1:{port}{endpoint_uri}"
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
