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
  - ${oc.env:VAR} Hydra resolvers → $host:VAR
  - ${oc.decode:...} Hydra resolvers → listed as non-migrated (needs manual review)
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

# Prefixes that mean the value is already migrated
_KNOWN_PREFIXES = ("$host:", "$lit:", "$runtime:")

# Regex to extract the variable name from ${oc.env:VAR_NAME} or ${oc.env:VAR_NAME,default}
_OC_ENV_RE = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,[^}]*)?\}")


def classify_value(raw: str) -> str | None:
    """Return the migrated value with an explicit prefix, or None if non-migrable.

    Returns:
        The migrated string, or None if the value cannot be auto-migrated
        (e.g. ${oc.decode:...}) and needs manual review.

    Args:
        raw: The raw string value from the YAML.
    """
    if not isinstance(raw, str):
        # Numbers, bools — treat as literal
        return f"$lit:{raw}"

    stripped = raw.strip()

    # Already has the new $ prefix — no change needed
    if stripped.startswith(_KNOWN_PREFIXES):
        return raw

    # ${oc.decode:...} — cannot auto-migrate, needs manual review
    if "${oc.decode:" in stripped:
        return None

    # ${oc.env:VAR_NAME} → $host:VAR_NAME
    m = _OC_ENV_RE.fullmatch(stripped)
    if m:
        return f"$host:{m.group(1)}"

    # $VAR_NAME → host ref (strip the leading $)
    if stripped.startswith("$") and _ENV_VAR_NAME_RE.match(stripped[1:]):
        return f"$host:{stripped[1:]}"

    # Pure numeric string → literal
    if stripped.isdigit():
        return f"$lit:{stripped}"

    # Bare valid env var name
    if _ENV_VAR_NAME_RE.match(stripped):
        # UPPER_CASE → almost certainly a host env var reference
        if _UPPER_CASE_RE.match(stripped):
            return f"$host:{stripped}"
        # Mixed-case valid identifier — ambiguous, default to host
        return f"$host:{stripped}"

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


def _is_nested_env_vars(env_dict: dict) -> bool:
    """Check if this env_vars dict has sub-keys like deployment/evaluation (execution-style)."""
    return any(isinstance(v, dict) for v in env_dict.values())


def migrate_env_vars_dict(
    env_dict: dict,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Migrate a flat env_vars dict.

    Returns:
        (changes, non_migrable) where changes is {key: (old_value, new_value)}
        and non_migrable is a list of "key: value" strings that need manual review.
    """
    changes = {}
    non_migrable = []
    for key, value in env_dict.items():
        if isinstance(value, dict):
            # Nested (execution.env_vars.deployment / .evaluation) — recurse
            continue
        old = str(value)
        new = classify_value(value)
        if new is None:
            non_migrable.append(f"{key}: {old}")
        elif old != new:
            changes[key] = (old, new)
    return changes, non_migrable


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


def migrate(text: str) -> tuple[str, list[str], list[str]]:
    """Migrate a YAML config string.

    Returns:
        (new_text, change_descriptions, non_migrable_descriptions)
    """
    # Use yaml only for understanding the structure, apply changes to raw text
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return text, [], []

    all_changes: dict[str, tuple[str, str]] = {}
    descriptions: list[str] = []
    all_non_migrable: list[str] = []

    for path, env_dict in _find_env_vars_sections(data):
        section_label = ".".join(path)

        if _is_nested_env_vars(env_dict):
            # execution.env_vars with deployment/evaluation sub-dicts
            for sub_key, sub_dict in env_dict.items():
                if isinstance(sub_dict, dict):
                    changes, non_migrable = migrate_env_vars_dict(sub_dict)
                    for k, (old, new) in changes.items():
                        descriptions.append(
                            f"  {section_label}.{sub_key}.{k}: {old} -> {new}"
                        )
                    for item in non_migrable:
                        all_non_migrable.append(f"  {section_label}.{sub_key}.{item}")
                    all_changes.update(changes)
        else:
            changes, non_migrable = migrate_env_vars_dict(env_dict)
            for k, (old, new) in changes.items():
                descriptions.append(f"  {section_label}.{k}: {old} -> {new}")
            for item in non_migrable:
                all_non_migrable.append(f"  {section_label}.{item}")
            all_changes.update(changes)

    if not all_changes and not all_non_migrable:
        return text, ["No changes needed."], []

    new_text = process_yaml_text(text, all_changes) if all_changes else text
    return new_text, descriptions, all_non_migrable


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

    new_text, descriptions, non_migrable = migrate(text)

    print(f"Migration: {args.config}", file=sys.stderr)
    for desc in descriptions:
        print(desc, file=sys.stderr)

    if non_migrable:
        print(
            "\n\033[31mNon-migrable (needs manual review):\033[0m",
            file=sys.stderr,
        )
        for item in non_migrable:
            print(f"\033[31m{item}\033[0m", file=sys.stderr)

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
