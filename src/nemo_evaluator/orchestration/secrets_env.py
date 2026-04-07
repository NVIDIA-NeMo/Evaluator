# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-group env var disambiguation for secure multi-container execution.

Ported from the old nemo-evaluator-launcher's env_vars.py pattern.  Each
"group" (e.g. a model service, an eval task) can define env vars with
potentially conflicting names (e.g. different ``HF_TOKEN`` values).  All
values are stored in a single ``.secrets.env`` file under disambiguated
names, and a per-group re-export command restores the original name just
before the container that needs it.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VarRemapping:
    """Maps an original env var name to its disambiguated name."""

    original_name: str
    disambiguated_name: str


@dataclass
class SecretsEnvResult:
    """Result of generating a .secrets.env file for grouped env var definitions.

    The .secrets.env file contains ``export DISAMBIGUATED="value"`` lines.
    Each original var is disambiguated with a per-group suffix so that
    different groups can define the same var name with different values.

    At runtime, ``source .secrets.env`` loads all values, then
    ``build_reexport_commands(group)`` emits the correct ``export ORIG=``
    lines for each container.
    """

    secrets_content: str
    """Full .secrets.env body — newline-separated ``export KEY="value"`` lines."""

    group_remappings: dict[str, list[VarRemapping]] = field(default_factory=dict)
    """Per-group list of (original → disambiguated) mappings."""


def _make_disambiguated_name(original_name: str, group_name: str, token: str) -> str:
    sanitized_group = re.sub(r"[^a-zA-Z0-9]", "_", group_name).upper()
    return f"{original_name}_{token}_{sanitized_group}"


def generate_secrets_env(
    env_groups: dict[str, dict[str, str]],
) -> SecretsEnvResult:
    """Generate a .secrets.env file from grouped env var definitions.

    Each group gets a unique random 4-hex token so that identical var
    names in different groups map to different disambiguated names.

    Args:
        env_groups: Mapping of group_name → {target_var_name: value}.

    Returns:
        SecretsEnvResult with the file content and per-group remappings.
    """
    lines: list[str] = []
    group_remappings: dict[str, list[VarRemapping]] = {}

    for group_name, env_vars in env_groups.items():
        token = secrets.token_hex(2)
        group_remappings[group_name] = []

        for target_name, value in env_vars.items():
            disambiguated = _make_disambiguated_name(target_name, group_name, token)
            group_remappings[group_name].append(
                VarRemapping(original_name=target_name, disambiguated_name=disambiguated)
            )
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'export {disambiguated}="{escaped}"')

    secrets_content = "\n".join(lines) + ("\n" if lines else "")

    return SecretsEnvResult(
        secrets_content=secrets_content,
        group_remappings=group_remappings,
    )


def build_reexport_commands(group_name: str, result: SecretsEnvResult) -> str:
    """Build shell commands to re-export disambiguated vars back to original names.

    For a group, generates commands like::

        export HF_TOKEN="${HF_TOKEN_a3f1_SVC_NEMOTRON}"

    Called right before each ``srun`` so Pyxis ``--container-env=HF_TOKEN``
    picks up the correct value for that group.

    Returns:
        Newline-separated export commands, or empty string if no vars.
    """
    commands = []
    for remapping in result.group_remappings.get(group_name, []):
        commands.append(f'export {remapping.original_name}="${{{remapping.disambiguated_name}}}"')
    return "\n".join(commands)


def reexport_keys(group_name: str, result: SecretsEnvResult) -> list[str]:
    """Return the list of original env var names that a group needs re-exported.

    These are the keys that should appear in ``--container-env=`` flags.
    """
    return [r.original_name for r in result.group_remappings.get(group_name, [])]


def redact_secrets_env_content(secrets_content: str) -> str:
    """Redact resolved values in .secrets.env content for dry-run display.

    Lines like ``export KEY="secret_value"`` become ``export KEY="***alue"``.
    Values with 4 or fewer characters are fully masked as ``"***"``.
    Non-export lines are passed through unchanged.
    """
    redacted_lines: list[str] = []
    for line in secrets_content.splitlines():
        if line.startswith("export ") and "=" in line[len("export ") :]:
            key, _, value = line[len("export ") :].partition("=")
            unquoted = value.strip('"')
            if len(unquoted) > 4:
                redacted_lines.append(f'export {key}="***{unquoted[-4:]}"')
            else:
                redacted_lines.append(f'export {key}="***"')
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines) + ("\n" if secrets_content.endswith("\n") else "")
