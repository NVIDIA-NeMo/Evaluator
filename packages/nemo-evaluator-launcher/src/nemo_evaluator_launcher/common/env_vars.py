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
    "lit:some_value"       — literal value, written directly
    "host:HOST_VAR"        — resolved from host env at config-load time via os.getenv()
    "runtime:RUNTIME_VAR"  — late-bound, resolved by the execution environment at runtime
"""

import copy
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import ClassVar

from omegaconf import DictConfig

from nemo_evaluator_launcher.common.logging_utils import logger

# --- Value types ---


@dataclass(frozen=True)
class EnvVarLiteral:
    """A literal env var value, written directly."""

    PREFIX: ClassVar[str] = "lit:"
    value: str


@dataclass(frozen=True)
class EnvVarFromHost:
    """An env var sourced from the host environment at config-load time."""

    PREFIX: ClassVar[str] = "host:"
    host_var_name: str


@dataclass(frozen=True)
class EnvVarRuntime:
    """A late-bound env var, resolved by the execution environment at runtime."""

    PREFIX: ClassVar[str] = "runtime:"
    runtime_var_name: str


EnvVarValue = EnvVarLiteral | EnvVarFromHost | EnvVarRuntime

# Pattern matching a valid env var name (used for backward-compat heuristic)
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_env_var_value(raw: str) -> EnvVarValue:
    """Parse a raw env var value string into a typed EnvVarValue.

    Every value must carry an explicit prefix (host:, lit:, runtime:).
    Unprefixed values raise ValueError with a suggested fix.

    Args:
        raw: The raw string value from config.

    Raises:
        ValueError: If the value has no recognized prefix or uses Hydra resolver syntax.
    """
    # Hard error for Hydra resolver syntax
    if "${oc.env:" in raw or "${oc.decode:" in raw:
        raise ValueError(
            f"Hydra resolver syntax '{raw}' is not allowed in env var fields. "
            "It resolves secrets into the config object, defeating secret isolation. "
            f"Use '{EnvVarFromHost.PREFIX}VAR_NAME' instead."
        )

    # Explicit prefixes
    if raw.startswith(EnvVarLiteral.PREFIX):
        return EnvVarLiteral(value=raw[len(EnvVarLiteral.PREFIX) :])
    if raw.startswith(EnvVarFromHost.PREFIX):
        return EnvVarFromHost(host_var_name=raw[len(EnvVarFromHost.PREFIX) :])
    if raw.startswith(EnvVarRuntime.PREFIX):
        return EnvVarRuntime(runtime_var_name=raw[len(EnvVarRuntime.PREFIX) :])

    # No recognized prefix — build a helpful suggestion
    if raw.startswith("$") and _ENV_VAR_NAME_RE.match(raw[1:]):
        suggestion = f"{EnvVarFromHost.PREFIX}{raw[1:]}"
    elif _ENV_VAR_NAME_RE.match(raw):
        suggestion = f"{EnvVarFromHost.PREFIX}{raw}"
    else:
        suggestion = f"{EnvVarLiteral.PREFIX}{raw}"

    raise ValueError(
        f"Env var value '{raw}' must have an explicit prefix. "
        f"Use '{suggestion}' instead. "
        f"Run the migration tool: nel-migrate-config your_config.yaml"
    )


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
    """Result of generating a .secrets.env file for a set of env var groups.

    The .secrets.env file contains ``export DISAMBIGUATED=value`` lines.
    Each original env var name is disambiguated with a per-group suffix
    (e.g. ``HF_TOKEN_a3f1_TASK_A``) so that different groups (tasks,
    deployment) can define the same var name with different values
    without collisions.

    At runtime the script sources .secrets.env and then re-exports each
    disambiguated name back to the original name right before the
    container that needs it (see ``build_reexport_commands``).
    """

    secrets_content: str
    """Full .secrets.env file body — newline-separated ``export KEY="value"`` lines."""

    group_remappings: dict[str, list[VarRemapping]] = field(default_factory=dict)
    """Per-group list of (original_name → disambiguated_name) mappings for
    resolved vars (literal and host). Used by ``build_reexport_commands``
    to emit ``export ORIGINAL="${DISAMBIGUATED}"`` shell commands."""

    runtime_vars: dict[str, list[VarRemapping]] = field(default_factory=dict)
    """Per-group runtime vars (runtime:). These have no value in .secrets.env;
    their reexport references the runtime variable name directly
    (e.g. ``export API_KEY="${NGC_API_TOKEN}"``)."""

    literal_disambiguated_names: set[str] = field(default_factory=set)
    """Set of disambiguated names whose values came from EnvVarLiteral.
    Used by ``redact_secrets_env_content`` to show literal values in
    clear text while masking actual secrets in dry-run output."""


def _make_disambiguated_name(original_name: str, group_name: str, token: str) -> str:
    """Create a disambiguated env var name: ORIGINAL_<token>_<SANITIZED_GROUP>."""
    sanitized_group = re.sub(r"[^a-zA-Z0-9]", "_", group_name).upper()
    return f"{original_name}_{token}_{sanitized_group}"


def generate_secrets_env(
    env_groups: dict[str, dict[str, EnvVarValue]],
) -> SecretsEnvResult:
    """Generate a .secrets.env file from grouped env var definitions.

    Each group (e.g. "deployment", "task_a") gets its vars disambiguated
    with a per-group suffix: VAR_<token>_<GROUP_NAME> (unique token per group).

    Args:
        env_groups: Mapping of group_name → {target_var_name: EnvVarValue}.

    Returns:
        SecretsEnvResult with the file content, remappings, and runtime vars.
    """
    lines: list[str] = []
    group_remappings: dict[str, list[VarRemapping]] = {}
    runtime_vars: dict[str, list[VarRemapping]] = {}
    literal_disambiguated_names: set[str] = set()

    for group_name, env_vars in env_groups.items():
        # Unique token per group so disambiguated names differ across groups
        token = secrets.token_hex(2)  # 4 hex chars, e.g. "a3f1"
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
                # Runtime var — no value to write; reexport references the
                # runtime var name directly (not the disambiguated suffix).
                runtime_remapping = VarRemapping(
                    original_name=target_name,
                    disambiguated_name=val.runtime_var_name,
                )
                runtime_vars[group_name].append(runtime_remapping)
                logger.debug(
                    "Runtime env var (no value in secrets file)",
                    target=target_name,
                    runtime_var=val.runtime_var_name,
                    group=group_name,
                )
            else:
                group_remappings[group_name].append(remapping)
                lines.append(f'export {disambiguated}="{resolved_value}"')
                if isinstance(val, EnvVarLiteral):
                    literal_disambiguated_names.add(disambiguated)
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
        literal_disambiguated_names=literal_disambiguated_names,
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
    # Resolved vars (literal + host) first — they source from .secrets.env
    for remapping in result.group_remappings.get(group_name, []):
        commands.append(
            'export %s="${%s}"'
            % (remapping.original_name, remapping.disambiguated_name)
        )
    # Runtime vars second — they may reference vars re-exported above
    # (e.g. API_KEY=$NGC_API_TOKEN where NGC_API_TOKEN was just re-exported)
    for remapping in result.runtime_vars.get(group_name, []):
        commands.append(
            'export %s="${%s}"'
            % (remapping.original_name, remapping.disambiguated_name)
        )
    return " ; ".join(commands)


def redact_secrets_env_content(
    secrets_content: str,
    literal_disambiguated_names: set[str] | None = None,
) -> str:
    """Redact resolved values in .secrets.env content for dry-run display.

    Lines like ``export KEY=secret_value`` become ``export KEY=***alue``.
    Values with 4 or fewer characters are fully masked as ``***``.
    Literal values (keys in *literal_disambiguated_names*) are shown as-is.
    Non-export lines are passed through unchanged.
    """
    literal_names = literal_disambiguated_names or set()
    redacted_lines: list[str] = []
    for line in secrets_content.splitlines():
        if line.startswith("export ") and "=" in line[len("export ") :]:
            key, _, value = line[len("export ") :].partition("=")
            # Strip surrounding quotes added for shell safety
            unquoted = value.strip('"')
            if key in literal_names:
                redacted_lines.append(line)
            elif len(unquoted) > 4:
                redacted_lines.append(f'export {key}="***{unquoted[-4:]}"')
            else:
                redacted_lines.append(f'export {key}="***"')
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines) + ("\n" if secrets_content.endswith("\n") else "")


# --- Config collection helpers ---


def _collect_top_level_env_vars(cfg: DictConfig) -> dict[str, str]:
    """Collect top-level env_vars from cfg.env_vars (new unified config)."""
    return dict(cfg.get("env_vars", {}) or {})


def collect_eval_env_vars(
    cfg: DictConfig,
    task: DictConfig,
    api_key_name: str | None = None,
) -> dict[str, EnvVarValue]:
    """Collect and parse evaluation env vars from config for a single task.

    Merges (last wins):
        cfg.env_vars → cfg.evaluation.env_vars → task.env_vars → api_key

    Args:
        cfg: Full run config.
        task: The specific evaluation task config.
        api_key_name: API key env var name (from get_api_key_name(cfg)), or None.

    Returns:
        dict mapping target_name → EnvVarValue.
    """
    # Collect raw env vars (target_name → raw_value_string)
    # 1. Top-level env_vars (new unified config)
    raw_env_vars: dict[str, str] = _collect_top_level_env_vars(cfg)

    # 2. evaluation.env_vars (global eval-level)
    raw_env_vars.update(copy.deepcopy(dict(cfg.evaluation.get("env_vars", {}))))

    # 3. task.env_vars (task-level overrides)
    raw_env_vars.update(task.get("env_vars", {}))

    # 4. API key — ensure the named env var is present in env_vars.
    # If the user already declared it (e.g. NGC_API_TOKEN: host:NGC_API_TOKEN),
    # do nothing. Otherwise, add it as a host ref so it gets resolved from the host env.
    if api_key_name and api_key_name not in raw_env_vars:
        raw_env_vars[api_key_name] = f"{EnvVarFromHost.PREFIX}{api_key_name}"

    # Parse main env vars
    parsed: dict[str, EnvVarValue] = {}
    for target_name, raw_value in raw_env_vars.items():
        parsed[target_name] = parse_env_var_value(str(raw_value))

    return parsed


def collect_deployment_env_vars(cfg: DictConfig) -> dict[str, EnvVarValue]:
    """Collect and parse deployment env vars from config.

    Merges (last wins):
        cfg.env_vars → cfg.deployment.env_vars

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

    # 2. cfg.deployment.env_vars to override the top ones
    if cfg.deployment.get("env_vars"):
        for target_name, raw_value in cfg.deployment["env_vars"].items():
            parsed[target_name] = parse_env_var_value(str(raw_value))

    return parsed


def collect_exporters_env_vars(cfg: DictConfig) -> dict[str, EnvVarValue]:
    """Collect and parse exporters env vars from config.

    Merges (last wins):
        cfg.env_vars → cfg.execution.env_vars.export (deprecated)
        → cfg.deployment.env_vars

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

    # 2. export.env_vars
    export_vars = cfg.get("export", {}).get("env_vars", {})
    if export_vars:
        for target_name, raw_value in export_vars.items():
            parsed[target_name] = parse_env_var_value(str(raw_value))

    return parsed
