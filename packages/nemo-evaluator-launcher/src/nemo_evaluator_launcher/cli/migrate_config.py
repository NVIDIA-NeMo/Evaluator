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
# TODO(Apr 2026): Remove this entrypoint once the migration period is over.
"""Migrate launcher config YAML to use explicit host:/lit: env var prefixes.

Usage:
    nel-migrate-config config.yaml              # preview (dry-run)
    nel-migrate-config config.yaml --write      # overwrite in place
    nel-migrate-config config.yaml -o new.yaml  # write to new file

Heuristics for classifying bare (unprefixed) values:
  - Already prefixed (host:, lit:, runtime:) → kept as-is
  - ${oc.env:VAR} Hydra resolvers → host:VAR
  - ${oc.decode:...} Hydra resolvers → listed as non-migrated (needs manual review)
  - $VAR_NAME (dollar-prefixed) → host:VAR_NAME
  - Bare UPPER_CASE name matching [A-Z_][A-Z0-9_]* → host:NAME
  - Numbers (0, 1, etc.) → lit:value
  - Everything else (paths, URLs, mixed-case, spaces) → lit:value

Structural migration:
  - execution.env_vars.deployment → deployment.env_vars
  - execution.env_vars.evaluation → top-level env_vars
  - Existing (new-location) keys take precedence over execution.env_vars keys
"""

import argparse
import re
import sys

# Matches a valid env-var-shaped name: starts with letter/underscore, rest alphanumeric/underscore
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matches an UPPER_CASE env var name (the typical host-env pattern)
_UPPER_CASE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Prefixes that mean the value is already migrated
_KNOWN_PREFIXES = ("host:", "lit:", "runtime:")

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
        return f"lit:{raw}"

    stripped = raw.strip()

    # Already has the new $ prefix — no change needed
    if stripped.startswith(_KNOWN_PREFIXES):
        return raw

    # ${oc.decode:...} — cannot auto-migrate, needs manual review
    if "${oc.decode:" in stripped:
        return None

    # ${oc.env:VAR_NAME} → host:VAR_NAME
    m = _OC_ENV_RE.fullmatch(stripped)
    if m:
        return f"host:{m.group(1)}"

    # $VAR_NAME → host ref (strip the leading $)
    if stripped.startswith("$") and _ENV_VAR_NAME_RE.match(stripped[1:]):
        return f"host:{stripped[1:]}"

    # Pure numeric string → literal
    if stripped.isdigit():
        return f"lit:{stripped}"

    # Bare valid env var name
    if _ENV_VAR_NAME_RE.match(stripped):
        # UPPER_CASE → almost certainly a host env var reference
        if _UPPER_CASE_RE.match(stripped):
            return f"host:{stripped}"
        # Mixed-case valid identifier — ambiguous, default to host
        return f"host:{stripped}"

    # Everything else: paths, URLs, strings with spaces/slashes → literal
    return f"lit:{stripped}"


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
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Migrate a flat env_vars dict.

    Returns:
        (changes, non_migrable) where changes is [(key, old_value, new_value), ...]
        and non_migrable is a list of "key: value" strings that need manual review.
    """
    changes = []
    non_migrable = []
    for key, value in env_dict.items():
        if isinstance(value, dict):
            continue
        old = str(value)
        new = classify_value(value)
        if new is None:
            non_migrable.append(f"{key}: {old}")
        elif old != new:
            changes.append((key, old, new))
    return changes, non_migrable


def process_yaml_text(text: str, changes: list[tuple[str, str, str]]) -> str:
    """Apply value replacements directly to YAML text, preserving formatting and comments.

    For each env var key with a changed value, find `key: old_value` lines and
    replace the value portion.  Changes is a list of (key, old_val, new_val).
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip()
        replaced = False
        for key, old_val, new_val in changes:
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


# ---------------------------------------------------------------------------
# Structural migration: execution.env_vars → deployment.env_vars / env_vars
# ---------------------------------------------------------------------------


def _get_indent(line: str) -> int:
    """Return the number of leading spaces."""
    return len(line) - len(line.lstrip())


def _is_content_line(line: str) -> bool:
    """Return True if the line is not blank and not a pure comment."""
    stripped = line.lstrip()
    return bool(stripped) and not stripped.startswith("#")


def _find_top_level_key(lines: list[str], key: str) -> int | None:
    """Find the line index of a top-level YAML key (indent 0)."""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if _get_indent(line) == 0 and (
            stripped == f"{key}:" or stripped.startswith(f"{key}: ")
        ):
            return i
    return None


def _find_child_key(
    lines: list[str], key: str, parent_line: int, parent_indent: int
) -> int | None:
    """Find a child key directly under parent_line at parent_indent + 2."""
    target_indent = parent_indent + 2
    for i in range(parent_line + 1, len(lines)):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _get_indent(lines[i])
        if indent <= parent_indent:
            break  # Out of parent scope
        if indent == target_indent and (
            stripped == f"{key}:" or stripped.startswith(f"{key}: ")
        ):
            return i
    return None


def _find_block_end(lines: list[str], start_line: int) -> int:
    """Find the end (exclusive) of the indented block starting at start_line.

    The block includes the key line and all subsequent lines that are either
    blank/comment or indented deeper than the key line.
    """
    start_indent = _get_indent(lines[start_line])
    end = start_line + 1
    while end < len(lines):
        stripped = lines[end].lstrip()
        if not stripped or stripped.startswith("#"):
            end += 1
            continue
        if _get_indent(lines[end]) <= start_indent:
            break
        end += 1
    return end


def _parse_flat_entries(
    lines: list[str], key_line: int | None, key_indent: int
) -> dict[str, str]:
    """Parse key: value entries from a YAML block.

    Returns {name: raw_value_str} for entries directly under key_line.
    """
    if key_line is None:
        return {}
    entries: dict[str, str] = {}
    entry_indent = key_indent + 2
    for i in range(key_line + 1, len(lines)):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _get_indent(lines[i])
        if indent <= key_indent:
            break
        if indent == entry_indent:
            if ": " in stripped:
                k, v = stripped.split(": ", 1)
                entries[k] = v
            elif stripped.endswith(": {}"):
                # empty dict like "evaluation: {}"
                pass
            elif stripped.endswith(":"):
                # Sub-block (e.g. nested dict) — skip
                pass
    return entries


def _find_insert_point(lines: list[str], block_key_line: int) -> int:
    """Find where to append a new entry in a block: after the last content line.

    Unlike _find_block_end, this skips trailing blank lines so that new entries
    are placed immediately after existing content, not after a gap.
    """
    start_indent = _get_indent(lines[block_key_line])
    last_content = block_key_line
    for i in range(block_key_line + 1, len(lines)):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if _get_indent(lines[i]) <= start_indent:
            break
        last_content = i
    return last_content + 1


def _remove_lines(lines: list[str], start: int, end: int) -> list[str]:
    """Remove lines[start:end], also removing consecutive blank lines at the cut point."""
    result = lines[:start] + lines[end:]
    # Clean up consecutive blank lines at the cut point
    while (
        start < len(result)
        and start > 0
        and result[start].strip() == ""
        and result[start - 1].strip() == ""
    ):
        result.pop(start)
    return result


def _relocate_execution_env_vars(text: str) -> tuple[str, list[str]]:
    """Relocate execution.env_vars.{deployment,evaluation} to proper locations.

    - execution.env_vars.deployment entries → deployment.env_vars
    - execution.env_vars.evaluation entries → top-level env_vars
    - Existing (new-location) keys take precedence
    - The execution.env_vars block is removed from the text

    Returns (modified_text, descriptions).
    """
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return text, []

    exec_cfg = data.get("execution", {})
    if not isinstance(exec_cfg, dict):
        return text, []

    exec_env = exec_cfg.get("env_vars")
    if not isinstance(exec_env, dict):
        return text, []

    # Only handle nested style (has deployment/evaluation sub-dicts)
    deploy_vars = exec_env.get("deployment", {})
    eval_vars = exec_env.get("evaluation", {})
    if not isinstance(deploy_vars, dict) or not isinstance(eval_vars, dict):
        return text, []

    if not deploy_vars and not eval_vars:
        # Both empty — just remove the block, nothing to relocate
        lines = text.split("\n")
        exec_line = _find_top_level_key(lines, "execution")
        if exec_line is not None:
            env_line = _find_child_key(lines, "env_vars", exec_line, 0)
            if env_line is not None:
                env_end = _find_block_end(lines, env_line)
                lines = _remove_lines(lines, env_line, env_end)
                return "\n".join(lines), ["  Removed empty execution.env_vars"]
        return text, []

    # Determine what already exists at the target locations
    existing_top_env = data.get("env_vars", {}) or {}
    existing_deploy_env = (data.get("deployment") or {}).get("env_vars", {}) or {}

    descriptions = []

    # Build entries to insert (apply value migration too)
    deploy_to_insert: dict[str, str] = {}
    for k, v in deploy_vars.items():
        if k in existing_deploy_env:
            descriptions.append(
                f"  execution.env_vars.deployment.{k}: skipped (already in deployment.env_vars)"
            )
            continue
        migrated = classify_value(v)
        if migrated is not None:
            deploy_to_insert[k] = migrated
            descriptions.append(
                f"  execution.env_vars.deployment.{k} → deployment.env_vars.{k}: {migrated}"
            )

    eval_to_insert: dict[str, str] = {}
    for k, v in eval_vars.items():
        if k in existing_top_env:
            descriptions.append(
                f"  execution.env_vars.evaluation.{k}: skipped (already in env_vars)"
            )
            continue
        migrated = classify_value(v)
        if migrated is not None:
            eval_to_insert[k] = migrated
            descriptions.append(
                f"  execution.env_vars.evaluation.{k} → env_vars.{k}: {migrated}"
            )

    lines = text.split("\n")

    # Step 1: Find and remove the execution.env_vars block
    exec_line = _find_top_level_key(lines, "execution")
    if exec_line is None:
        return text, []

    env_line = _find_child_key(lines, "env_vars", exec_line, 0)
    if env_line is None:
        return text, []

    env_end = _find_block_end(lines, env_line)
    lines = _remove_lines(lines, env_line, env_end)

    # Step 2: Insert deployment entries into deployment.env_vars
    if deploy_to_insert:
        deploy_line = _find_top_level_key(lines, "deployment")
        if deploy_line is not None:
            # Check if deployment.env_vars already exists
            deploy_env_line = _find_child_key(lines, "env_vars", deploy_line, 0)
            if deploy_env_line is not None:
                # Append entries after the last existing entry
                insert_at = _find_insert_point(lines, deploy_env_line)
                new_lines = [f"    {k}: {v}" for k, v in deploy_to_insert.items()]
                lines = lines[:insert_at] + new_lines + lines[insert_at:]
            else:
                # Create env_vars sub-key under deployment
                insert_at = _find_insert_point(lines, deploy_line)
                new_lines = ["  env_vars:"] + [
                    f"    {k}: {v}" for k, v in deploy_to_insert.items()
                ]
                lines = lines[:insert_at] + new_lines + lines[insert_at:]
        else:
            # Create deployment section (insert before evaluation if exists, else at end)
            eval_line = _find_top_level_key(lines, "evaluation")
            insert_at = eval_line if eval_line is not None else len(lines)
            new_lines = [
                "",
                "deployment:",
                "  env_vars:",
            ] + [f"    {k}: {v}" for k, v in deploy_to_insert.items()]
            lines = lines[:insert_at] + new_lines + lines[insert_at:]

    # Step 3: Insert evaluation entries into top-level env_vars
    if eval_to_insert:
        top_env_line = _find_top_level_key(lines, "env_vars")
        if top_env_line is not None:
            # Append entries after the last existing entry
            insert_at = _find_insert_point(lines, top_env_line)
            new_lines = [f"  {k}: {v}" for k, v in eval_to_insert.items()]
            lines = lines[:insert_at] + new_lines + lines[insert_at:]
        else:
            # Create top-level env_vars section (insert before execution)
            exec_line = _find_top_level_key(lines, "execution")
            insert_at = exec_line if exec_line is not None else 0
            # Find the right spot — after defaults/comments, before execution
            new_lines = [
                "",
                "env_vars:",
            ] + [f"  {k}: {v}" for k, v in eval_to_insert.items()]
            lines = lines[:insert_at] + new_lines + [""] + lines[insert_at:]

    # Clean up: collapse consecutive blank lines
    cleaned: list[str] = []
    for line in lines:
        if line.strip() == "" and cleaned and cleaned[-1].strip() == "":
            continue
        cleaned.append(line)

    return "\n".join(cleaned), descriptions


# ---------------------------------------------------------------------------
# Main migration pipeline
# ---------------------------------------------------------------------------


def migrate(text: str) -> tuple[str, list[str], list[str]]:
    """Migrate a YAML config string.

    Phase 1: Structural migration (execution.env_vars → proper locations).
    Phase 2: Value prefix migration (host:, lit:, etc.).

    Returns:
        (new_text, change_descriptions, non_migrable_descriptions)
    """
    import yaml

    # Phase 1: Structural relocation of execution.env_vars
    text, structural_descs = _relocate_execution_env_vars(text)

    # Phase 2: Value prefix migration
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        descs = structural_descs or ["No changes needed."]
        return text, descs, []

    all_changes: list[tuple[str, str, str]] = []
    descriptions: list[str] = []
    all_non_migrable: list[str] = []

    for path, env_dict in _find_env_vars_sections(data):
        section_label = ".".join(path)

        if _is_nested_env_vars(env_dict):
            # execution.env_vars with deployment/evaluation sub-dicts
            for sub_key, sub_dict in env_dict.items():
                if isinstance(sub_dict, dict):
                    changes, non_migrable = migrate_env_vars_dict(sub_dict)
                    for k, old, new in changes:
                        descriptions.append(
                            f"  {section_label}.{sub_key}.{k}: {old} -> {new}"
                        )
                    for item in non_migrable:
                        all_non_migrable.append(f"  {section_label}.{sub_key}.{item}")
                    all_changes.extend(changes)
        else:
            changes, non_migrable = migrate_env_vars_dict(env_dict)
            for k, old, new in changes:
                descriptions.append(f"  {section_label}.{k}: {old} -> {new}")
            for item in non_migrable:
                all_non_migrable.append(f"  {section_label}.{item}")
            all_changes.extend(changes)

    all_descs = structural_descs + descriptions
    if not all_descs and not all_non_migrable:
        return text, ["No changes needed."], []

    new_text = process_yaml_text(text, all_changes) if all_changes else text
    return new_text, all_descs, all_non_migrable


def main():
    parser = argparse.ArgumentParser(
        description="Migrate launcher config YAML to use explicit host:/lit: env var prefixes."
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
