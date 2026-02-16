#!/usr/bin/env python3
#
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
"""Migrate launcher config YAML to use explicit $host:/$lit: env var prefixes.

Usage:
    python scripts/migrate_config.py config.yaml              # preview (dry-run)
    python scripts/migrate_config.py config.yaml --write      # overwrite in place
    python scripts/migrate_config.py config.yaml -o new.yaml  # write to new file

Heuristics for classifying bare (unprefixed) values:
  - Already prefixed ($host:, $lit:, $runtime:) → kept as-is
  - Deprecated !host:/!lit:/!runtime: → converted to $host:/$lit:/$runtime:
  - $VAR_NAME (dollar-prefixed) → $host:VAR_NAME
  - Bare UPPER_CASE name matching [A-Z_][A-Z0-9_]* → $host:NAME
  - Numbers (0, 1, etc.) → $lit:value
  - Everything else (paths, URLs, mixed-case, spaces) → $lit:value
"""

import argparse
import re
import sys

# Matches a valid env-var-shaped name: starts with letter/underscore, rest alphanumeric/underscore
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches an UPPER_CASE env var name (the typical host-env pattern)
_UPPER_CASE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Prefixes that mean the value is already migrated (new $ syntax)
_KNOWN_PREFIXES = ("$host:", "$lit:", "$runtime:")

# Deprecated prefixes that need conversion from ! to $
_DEPRECATED_PREFIX_MAP = {
    "!host:": "$host:",
    "!lit:": "$lit:",
    "!runtime:": "$runtime:",
}

# Sections whose env_vars values default to literal (execution context)
_LITERAL_DEFAULT_SECTIONS = {"execution"}


def classify_value(raw: str, parent_section: str | None) -> str:
    """Return the migrated value with an explicit prefix.

    Args:
        raw: The raw string value from the YAML.
        parent_section: The top-level config section this env_var lives under
                        (e.g. "execution", "evaluation", "deployment", "env_vars").
    """
    if not isinstance(raw, str):
        # Numbers, bools — treat as literal
        return f"$lit:{raw}"

    stripped = raw.strip()

    # Already has the new $ prefix — no change needed
    if stripped.startswith(_KNOWN_PREFIXES):
        return raw

    # Deprecated ! prefix — convert to $
    for old_prefix, new_prefix in _DEPRECATED_PREFIX_MAP.items():
        if stripped.startswith(old_prefix):
            return new_prefix + stripped[len(old_prefix):]

    # Hydra resolver — flag but don't auto-migrate (needs manual attention)
    if "${oc.env:" in stripped or "${oc.decode:" in stripped:
        return raw  # leave as-is, user must handle manually

    # $VAR_NAME → host ref (strip the leading $)
    if stripped.startswith("$") and _ENV_VAR_NAME_RE.match(stripped[1:]):
        return f"$host:{stripped[1:]}"

    # Pure numeric string → literal
    if stripped.isdigit():
        return f"$lit:{stripped}"

    # Bare valid env var name
    if _ENV_VAR_NAME_RE.match(stripped):
        # In execution context, bare names were treated as literals
        if parent_section in _LITERAL_DEFAULT_SECTIONS:
            return f"$lit:{stripped}"
        # UPPER_CASE → almost certainly a host env var reference
        if _UPPER_CASE_RE.match(stripped):
            return f"$host:{stripped}"
        # Mixed-case valid identifier — ambiguous, default to literal
        return f"$lit:{stripped}"

    # Everything else: paths, URLs, strings with spaces/slashes → literal
    return f"$lit:{stripped}"


def _find_env_vars_sections(data, path=None):
    """Yield (path, dict) for every env_vars mapping in the config tree."""
    if path is None:
        path = []
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = path + [key]
            if key == "env_vars" and isinstance(value, dict):
                yield current_path, value
            else:
                yield from _find_env_vars_sections(value, current_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            yield from _find_env_vars_sections(item, path + [str(idx)])


def _get_parent_section(path: list[str]) -> str | None:
    """Return the top-level section name for an env_vars path.

    For paths like ["execution", "env_vars"] → "execution"
    For paths like ["env_vars"] → "env_vars"
    For paths like ["evaluation", "tasks", "0", "env_vars"] → "evaluation"
    """
    if len(path) >= 2:
        return path[0]
    if path:
        return path[0]
    return None


def _is_nested_env_vars(env_dict: dict) -> bool:
    """Check if this env_vars dict has sub-keys like deployment/evaluation (execution-style)."""
    return any(isinstance(v, dict) for v in env_dict.values())


def migrate_env_vars_dict(
    env_dict: dict, parent_section: str | None
) -> dict[str, tuple[str, str]]:
    """Migrate a flat env_vars dict. Returns {key: (old_value, new_value)}."""
    changes = {}
    for key, value in env_dict.items():
        if isinstance(value, dict):
            # Nested (execution.env_vars.deployment / .evaluation) — recurse
            continue
        old = str(value)
        new = classify_value(value, parent_section)
        if old != new:
            changes[key] = (old, new)
    return changes


def process_yaml_text(text: str, changes_by_key: dict[str, tuple[str, str]]) -> str:
    """Apply value replacements directly to YAML text, preserving formatting and comments.

    For each env var key with a changed value, find `key: old_value` lines and
    replace the value portion.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip()
        replaced = False
        for key, (old_val, new_val) in changes_by_key.items():
            # Match "KEY: value" or "KEY: value # comment" patterns
            # old_val may or may not be quoted in the YAML
            patterns = [
                f"{key}: {old_val}",
                f'{key}: "{old_val}"',
                f"{key}: '{old_val}'",
            ]
            for pattern in patterns:
                if stripped.startswith(pattern):
                    indent = line[: len(line) - len(stripped)]
                    rest = stripped[len(pattern) :]
                    result.append(f"{indent}{key}: {new_val}{rest}")
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            result.append(line)
    return "\n".join(result)


def migrate(text: str) -> tuple[str, list[str]]:
    """Migrate a YAML config string. Returns (new_text, list of change descriptions)."""
    # Use yaml only for understanding the structure, apply changes to raw text
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return text, []

    all_changes: dict[str, tuple[str, str]] = {}
    descriptions: list[str] = []

    for path, env_dict in _find_env_vars_sections(data):
        parent = _get_parent_section(path)
        section_label = ".".join(path)

        if _is_nested_env_vars(env_dict):
            # execution.env_vars with deployment/evaluation sub-dicts
            for sub_key, sub_dict in env_dict.items():
                if isinstance(sub_dict, dict):
                    sub_parent = "execution" if parent == "execution" else parent
                    changes = migrate_env_vars_dict(sub_dict, sub_parent)
                    for k, (old, new) in changes.items():
                        descriptions.append(
                            f"  {section_label}.{sub_key}.{k}: {old} -> {new}"
                        )
                    all_changes.update(changes)
        else:
            changes = migrate_env_vars_dict(env_dict, parent)
            for k, (old, new) in changes.items():
                descriptions.append(f"  {section_label}.{k}: {old} -> {new}")
            all_changes.update(changes)

    if not all_changes:
        return text, ["No changes needed."]

    new_text = process_yaml_text(text, all_changes)
    return new_text, descriptions


def main():
    parser = argparse.ArgumentParser(
        description="Migrate launcher config YAML to use explicit $host:/$lit: env var prefixes."
    )
    parser.add_argument("config", help="Path to the config YAML file")
    parser.add_argument(
        "--write", "-w", action="store_true", help="Overwrite the input file in place"
    )
    parser.add_argument(
        "--output", "-o", help="Write migrated config to this path (instead of stdout)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        text = f.read()

    new_text, descriptions = migrate(text)

    print(f"Migration: {args.config}", file=sys.stderr)
    for desc in descriptions:
        print(desc, file=sys.stderr)

    if args.write:
        with open(args.config, "w") as f:
            f.write(new_text)
        print(f"\nWritten to {args.config}", file=sys.stderr)
    elif args.output:
        with open(args.output, "w") as f:
            f.write(new_text)
        print(f"\nWritten to {args.output}", file=sys.stderr)
    else:
        print(new_text)


if __name__ == "__main__":
    main()
