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
"""Unified environment variable handling for NeMo Evaluator Launcher.

Three value types are supported via explicit prefixes:
    "!lit:some_value"       — literal value, written directly
    "!host:HOST_VAR"        — resolved from host env at config-load time via os.getenv()
    "!runtime:RUNTIME_VAR"  — late-bound, resolved by the execution environment at runtime
"""

import copy
import os
import re
import secrets
import warnings
from dataclasses import dataclass, field

from omegaconf import DictConfig

from nemo_evaluator_launcher.common.logging_utils import logger

# --- Value types ---

PREFIX_LIT = "!lit:"
PREFIX_HOST = "!host:"
PREFIX_RUNTIME = "!runtime:"


@dataclass(frozen=True)
class EnvVarLiteral:
    """A literal env var value, written directly."""

    value: str


@dataclass(frozen=True)
class EnvVarFromHost:
    """An env var sourced from the host environment at config-load time."""

    host_var_name: str


@dataclass(frozen=True)
class EnvVarRuntime:
    """A late-bound env var, resolved by the execution environment at runtime."""

    runtime_var_name: str


EnvVarValue = EnvVarLiteral | EnvVarFromHost | EnvVarRuntime

# Pattern matching a valid env var name (used for backward-compat heuristic)
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_var_value(raw: str, default_type: str = "host") -> EnvVarValue:
    """Parse a raw env var value string into a typed EnvVarValue.

    Supports explicit prefixes (!lit:, !host:, !runtime:) and backward-compatible
    unprefixed values (with deprecation warnings).

    Args:
        raw: The raw string value from config.
        default_type: How to interpret unprefixed bare names. "host" (default) treats
            them as env var references (for evaluation.env_vars backward compat).
            "lit" treats them as literal values (for execution.env_vars backward compat).

    Raises ValueError if ${oc.env:...} Hydra resolver syntax is detected.
    """
    # Hard error for Hydra resolver syntax
    if "${oc.env:" in raw or "${oc.decode:" in raw:
        raise ValueError(
            f"Hydra resolver syntax '{raw}' is not allowed in env var fields. "
            "It resolves secrets into the config object, defeating secret isolation. "
            "Use '!host:VAR_NAME' instead."
        )

    # Explicit prefixes
    if raw.startswith(PREFIX_LIT):
        return EnvVarLiteral(value=raw[len(PREFIX_LIT) :])
    if raw.startswith(PREFIX_HOST):
        return EnvVarFromHost(host_var_name=raw[len(PREFIX_HOST) :])
    if raw.startswith(PREFIX_RUNTIME):
        return EnvVarRuntime(runtime_var_name=raw[len(PREFIX_RUNTIME) :])

    # Backward-compatible: $VAR_NAME → !host:VAR_NAME
    if raw.startswith("$") and _ENV_VAR_NAME_RE.match(raw[1:]):
        warnings.warn(
            f"Unprefixed env var value '{raw}' is deprecated. "
            f"Use '!host:{raw[1:]}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return EnvVarFromHost(host_var_name=raw[1:])

    # Backward-compatible: bare env var name
    if _ENV_VAR_NAME_RE.match(raw):
        if default_type == "host":
            warnings.warn(
                f"Unprefixed env var value '{raw}' is deprecated. "
                f"Use '!host:{raw}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return EnvVarFromHost(host_var_name=raw)
        else:
            warnings.warn(
                f"Unprefixed env var value '{raw}' is deprecated. "
                f"Use '!lit:{raw}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return EnvVarLiteral(value=raw)

    # Backward-compatible: anything else (paths, URLs, etc.) → !lit:VALUE
    warnings.warn(
        f"Unprefixed env var value '{raw}' is deprecated. Use '!lit:{raw}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return EnvVarLiteral(value=raw)


def resolve_env_var(target_name: str, val: EnvVarValue) -> tuple[str, str | None]:
    """Resolve an EnvVarValue to (target_name, resolved_value_or_None).

    - Literal  → value as-is
    - Host     → os.getenv(host_var_name), raises if missing
    - Runtime  → None (marker only, resolved at execution time)

    Returns:
        (target_name, resolved_value) where resolved_value is None for runtime vars.

    Raises:
        ValueError: If a host env var is not set in the current environment.
    """
    if isinstance(val, EnvVarLiteral):
        return (target_name, val.value)

    if isinstance(val, EnvVarFromHost):
        resolved = os.getenv(val.host_var_name)
        if resolved is None:
            raise ValueError(
                f"Environment variable '{val.host_var_name}' (referenced by '{target_name}') "
                "is not set. Set it in your environment or use --env-file to load from a file."
            )
        return (target_name, resolved)

    if isinstance(val, EnvVarRuntime):
        return (target_name, None)

    raise TypeError(f"Unknown EnvVarValue type: {type(val)}")


# --- Disambiguation + .secrets.env generation ---


@dataclass(frozen=True)
class VarRemapping:
    """Maps an original env var name to its disambiguated name."""

    original_name: str
    disambiguated_name: str


@dataclass
class SecretsEnvResult:
    """Result of generating a .secrets.env file."""

    secrets_content: str
    """Full .secrets.env file body (export VAR=value lines)."""

    group_remappings: dict[str, list[VarRemapping]] = field(default_factory=dict)
    """Per group: list of original→disambiguated name mappings."""

    runtime_vars: dict[str, list[VarRemapping]] = field(default_factory=dict)
    """Per group: runtime vars that have no value in the secrets file."""


def _make_disambiguated_name(original_name: str, group_name: str, token: str) -> str:
    """Create a disambiguated env var name: ORIGINAL_<token>_<SANITIZED_GROUP>."""
    sanitized_group = re.sub(r"[^a-zA-Z0-9]", "_", group_name).upper()
    return f"{original_name}_{token}_{sanitized_group}"


def generate_secrets_env(
    env_groups: dict[str, dict[str, EnvVarValue]],
) -> SecretsEnvResult:
    """Generate a .secrets.env file from grouped env var definitions.

    Each group (e.g. "deployment", "eval_task_a") gets its vars disambiguated
    with a suffix: VAR_<random_token>_<GROUP_NAME>.

    Args:
        env_groups: Mapping of group_name → {target_var_name: EnvVarValue}.

    Returns:
        SecretsEnvResult with the file content, remappings, and runtime vars.
    """
    lines: list[str] = []
    group_remappings: dict[str, list[VarRemapping]] = {}
    runtime_vars: dict[str, list[VarRemapping]] = {}

    # One random token per generate call (shared across groups for this invocation)
    token = secrets.token_hex(3)  # 6 hex chars, e.g. "a3f1b2"

    for group_name, env_vars in env_groups.items():
        group_remappings[group_name] = []
        runtime_vars[group_name] = []

        for target_name, val in env_vars.items():
            disambiguated = _make_disambiguated_name(target_name, group_name, token)
            _, resolved_value = resolve_env_var(target_name, val)

            remapping = VarRemapping(
                original_name=target_name,
                disambiguated_name=disambiguated,
            )

            if resolved_value is None:
                # Runtime var — no value to write, executor handles it
                runtime_vars[group_name].append(remapping)
                logger.debug(
                    "Runtime env var (no value in secrets file)",
                    target=target_name,
                    disambiguated=disambiguated,
                    group=group_name,
                )
            else:
                group_remappings[group_name].append(remapping)
                lines.append(f"export {disambiguated}={resolved_value}")
                logger.debug(
                    "Resolved env var for secrets file",
                    target=target_name,
                    disambiguated=disambiguated,
                    group=group_name,
                )

    secrets_content = "\n".join(lines) + ("\n" if lines else "")

    return SecretsEnvResult(
        secrets_content=secrets_content,
        group_remappings=group_remappings,
        runtime_vars=runtime_vars,
    )


def build_reexport_commands(group_name: str, result: SecretsEnvResult) -> str:
    """Build shell commands to re-export disambiguated vars back to original names.

    For a group, generates commands like:
        export HF_TOKEN=$HF_TOKEN_a3f1b2_TASK_A

    Args:
        group_name: The group to generate re-export commands for.
        result: The SecretsEnvResult from generate_secrets_env().

    Returns:
        Shell command string (semicolon-separated exports), or empty string.
    """
    commands = []
    for remapping in result.group_remappings.get(group_name, []):
        commands.append(
            f"export {remapping.original_name}=${remapping.disambiguated_name}"
        )
    for remapping in result.runtime_vars.get(group_name, []):
        commands.append(
            f"export {remapping.original_name}=${remapping.disambiguated_name}"
        )
    return " ; ".join(commands)


# --- Config collection helpers ---


def _collect_top_level_env_vars(cfg: DictConfig) -> dict[str, str]:
    """Collect top-level env_vars from cfg.env_vars (new unified config)."""
    return dict(cfg.get("env_vars", {}) or {})


def collect_eval_env_vars(
    cfg: DictConfig,
    task: DictConfig,
    task_definition: dict,
    api_key_name: str | None = None,
) -> dict[str, EnvVarValue]:
    """Collect and parse evaluation env vars from config for a single task.

    Merges (last wins):
        cfg.env_vars → cfg.evaluation.env_vars → task.env_vars
        → cfg.execution.env_vars.evaluation (deprecated) → api_key

    Validates required_env_vars from task_definition are present.

    Args:
        cfg: Full run config.
        task: The specific evaluation task config.
        task_definition: Task definition from tasks mapping (has required_env_vars).
        api_key_name: API key env var name (from get_api_key_name(cfg)), or None.

    Returns:
        dict mapping target_name → EnvVarValue.

    Raises:
        ValueError: If required env vars are missing from config.
    """
    # Collect raw env vars (target_name → raw_value_string)
    # 1. Top-level env_vars (new unified config)
    raw_env_vars: dict[str, str] = _collect_top_level_env_vars(cfg)

    # 2. evaluation.env_vars (global eval-level)
    raw_env_vars.update(copy.deepcopy(dict(cfg.evaluation.get("env_vars", {}))))

    # 3. task.env_vars (task-level overrides)
    raw_env_vars.update(task.get("env_vars", {}))

    # 5. API key
    if api_key_name:
        if "API_KEY" in raw_env_vars:
            raise ValueError(
                "API_KEY is already defined in env_vars. "
                "Remove it or remove target.api_endpoint.api_key_name."
            )
        raw_env_vars["API_KEY"] = api_key_name

    # Check required env vars (excluding NEMO_EVALUATOR_DATASET_DIR)
    # Also check the deprecated exec eval vars for required var coverage
    exec_eval_vars = dict(cfg.execution.get("env_vars", {}).get("evaluation", {}))
    all_var_names = set(raw_env_vars.keys()) | set(exec_eval_vars.keys())
    for required_env_var in task_definition.get("required_env_vars", []):
        if required_env_var == "NEMO_EVALUATOR_DATASET_DIR":
            continue
        if required_env_var not in all_var_names:
            raise ValueError(
                f"{task.name} task requires environment variable {required_env_var}. "
                "Specify it in the task subconfig in the 'env_vars' dict as the following "
                f"pair {required_env_var}: !host:YOUR_ENV_VAR_NAME"
            )

    # Parse main env vars (evaluation context: bare names default to host)
    parsed: dict[str, EnvVarValue] = {}
    for target_name, raw_value in raw_env_vars.items():
        parsed[target_name] = parse_env_var_value(str(raw_value))

    # 4. Deprecated: execution.env_vars.evaluation (parsed with literal default)
    if exec_eval_vars:
        warnings.warn(
            "cfg.execution.env_vars.evaluation is deprecated. "
            "Move these variables to top-level env_vars or evaluation.env_vars instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        for target_name, raw_value in exec_eval_vars.items():
            parsed[target_name] = parse_env_var_value(
                str(raw_value), default_type="lit"
            )

    return parsed


def collect_deployment_env_vars(cfg: DictConfig) -> dict[str, EnvVarValue]:
    """Collect and parse deployment env vars from config.

    Merges (last wins):
        cfg.env_vars → cfg.execution.env_vars.deployment (deprecated)
        → cfg.deployment.env_vars (deprecated)

    Args:
        cfg: Full run config.

    Returns:
        dict mapping target_name → EnvVarValue.
    """
    # 1. Top-level env_vars (new unified config) — uses host default
    top_level_vars = _collect_top_level_env_vars(cfg)
    parsed: dict[str, EnvVarValue] = {}
    for target_name, raw_value in top_level_vars.items():
        parsed[target_name] = parse_env_var_value(str(raw_value))

    # 2. Deprecated: execution.env_vars.deployment — uses literal default
    exec_deploy_vars = dict(cfg.execution.get("env_vars", {}).get("deployment", {}))
    if exec_deploy_vars:
        warnings.warn(
            "cfg.execution.env_vars.deployment is deprecated. "
            "Move these variables to top-level env_vars or deployment.env_vars instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        for target_name, raw_value in exec_deploy_vars.items():
            parsed[target_name] = parse_env_var_value(
                str(raw_value), default_type="lit"
            )

    # 3. Deprecated: cfg.deployment.env_vars — uses literal default
    if cfg.deployment.get("env_vars"):
        warnings.warn(
            "cfg.deployment.env_vars will be deprecated in future versions. "
            "Use top-level env_vars instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        for target_name, raw_value in cfg.deployment["env_vars"].items():
            parsed[target_name] = parse_env_var_value(
                str(raw_value), default_type="lit"
            )

    return parsed


def collect_execution_eval_env_vars(cfg: DictConfig) -> dict[str, EnvVarValue]:
    """Collect and parse execution-level evaluation env vars.

    These come from cfg.execution.env_vars.evaluation (deprecated path).
    Now folded into collect_eval_env_vars() hierarchy; this function remains
    for backward compatibility with executors that call it separately.

    Args:
        cfg: Full run config.

    Returns:
        dict mapping target_name → EnvVarValue.
    """
    raw_env_vars: dict[str, str] = dict(
        cfg.execution.get("env_vars", {}).get("evaluation", {})
    )

    # Parse — execution.env_vars values default to literal
    parsed: dict[str, EnvVarValue] = {}
    for target_name, raw_value in raw_env_vars.items():
        parsed[target_name] = parse_env_var_value(str(raw_value), default_type="lit")

    return parsed
