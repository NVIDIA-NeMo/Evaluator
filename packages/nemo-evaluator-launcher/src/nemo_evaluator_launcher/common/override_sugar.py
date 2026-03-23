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
"""Syntactic sugar for task-name-based CLI overrides.

Allows users to write:
    -o 'evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=64'
instead of fragile index-based overrides:
    -o '++evaluation.tasks.1.nemo_evaluator_config.config.params.parallelism=64'
"""

from __future__ import annotations

import re
from collections import defaultdict

import yaml
from omegaconf import DictConfig, OmegaConf

# Regex to strip Hydra prefixes (++, +, ~) from the beginning of an override key.
_HYDRA_PREFIX_RE = re.compile(r"^(\+\+|\+|~)?(.*)$")

_TASKS_PREFIX = "evaluation.tasks."


def partition_overrides(
    overrides: list[str],
) -> tuple[list[str], list[str]]:
    """Split CLI overrides into Hydra-native and task-name sugar overrides.

    An override is considered "sugar" when:
    - It starts with ``[++|+|~]evaluation.tasks.<segment>.`` (after stripping the
      Hydra prefix)
    - ``<segment>`` is **not** parseable as an ``int`` (i.e. it is a task name,
      not a positional index).

    Everything else is passed through to Hydra unchanged.

    Returns:
        (hydra_overrides, task_name_overrides)
    """
    hydra_overrides: list[str] = []
    task_name_overrides: list[str] = []

    for override in overrides:
        if _is_task_name_override(override):
            task_name_overrides.append(override)
        else:
            hydra_overrides.append(override)

    return hydra_overrides, task_name_overrides


def _is_task_name_override(override: str) -> bool:
    """Return True if *override* uses the task-name sugar pattern."""
    # Strip the Hydra prefix
    m = _HYDRA_PREFIX_RE.match(override)
    if m is None:
        return False
    key_value = m.group(2)  # everything after the prefix

    # Must start with "evaluation.tasks."
    if not key_value.startswith(_TASKS_PREFIX):
        return False

    rest = key_value[len(_TASKS_PREFIX) :]

    # The very next dot-segment must be non-integer to qualify as sugar.
    # For a bare "evaluation.tasks.mmlu" (no more dots, no =), we still consider it
    # sugar (it would be an error later, but it's not a valid Hydra index override either).
    first_segment = rest.split(".", 1)[0].split("=", 1)[0]
    if not first_segment:
        raise ValueError("invalid override={override!r}")

    try:
        int(first_segment)
    except ValueError:
        return True  # non-integer → sugar

    return False  # integer → pass through to Hydra


def apply_task_name_overrides(
    cfg: DictConfig,
    task_overrides: list[str],
) -> DictConfig:
    """Resolve task-name sugar overrides and apply them to *cfg* in-place.

    Steps:
    1. Build a ``name → [index, …]`` mapping from ``cfg.evaluation.tasks``.
    2. For each override, find the matching task name (exact then suffix match),
       rewrite to an index-based path, and apply via ``OmegaConf.update``.

    Raises:
        ValueError: If a task name is not found, or matches ambiguously.
    """
    if not task_overrides:
        return cfg

    task_name_to_index_mapping = _build_task_name_to_index_mapping(cfg)

    for override in task_overrides:
        prefix, after_prefix = _strip_hydra_prefix(override)
        # after_prefix looks like "evaluation.tasks.<task_name_and_key>=<value>"
        task_name_and_key_value = after_prefix[
            len(_TASKS_PREFIX) :
        ]  # "<task_name_and_key>=<value>"
        task_name, key, value_str = _resolve_task_segment(
            task_name_and_key_value, task_name_to_index_mapping
        )
        indices = _lookup_task(task_name, task_name_to_index_mapping)

        for idx in indices:
            dotted_path = f"evaluation.tasks.{idx}"
            if key:
                dotted_path += f".{key}"

            if prefix == "~":
                _delete_key(cfg, dotted_path)
            else:
                value = _parse_value(value_str) if value_str is not None else None
                force_add = prefix == "++"
                OmegaConf.update(cfg, dotted_path, value, force_add=force_add)

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_task_name_to_index_mapping(cfg: DictConfig) -> dict[str, list[int]]:
    """Return ``{task_name: [idx, …]}`` from ``cfg.evaluation.tasks``."""
    mapping = defaultdict(list)
    tasks = cfg.evaluation.tasks
    for idx, task in enumerate(tasks):
        mapping[task.name].append(idx)
        if len(mapping[task.name]) > 1:
            raise NotImplementedError("duplicate task.names not supported yet")
    return dict(mapping)


def _strip_hydra_prefix(override: str) -> tuple[str, str]:
    """Return ``(prefix, key_value)`` where *prefix* is ``''``, ``'++'``, ``'+'``, or ``'~'``."""
    m = _HYDRA_PREFIX_RE.match(override)
    return (m.group(1) or ""), m.group(2)


def _resolve_task_segment(
    task_name_and_key_value: str,
    task_name_to_index_mapping: dict[str, list[int]],
) -> tuple[str, str, str | None]:
    """Split `task_name_and_key_value` into task name, remaining key and value.

    Returns:
        (task_name, remaining_key, value_str)
        *value_str* is ``None`` when there is no ``=`` in the override.
    """
    # Split off the value first (everything after the first =)
    all_task_names = list(task_name_to_index_mapping.keys())
    value_str = None
    task_name_and_key = task_name_and_key_value
    eq_idx = task_name_and_key_value.find("=")
    if eq_idx != -1:
        value_str = task_name_and_key_value[eq_idx + 1 :]
        task_name_and_key = task_name_and_key_value[:eq_idx]

    # task_name_and_key is now "<task_name_segments>.<remaining_key>".
    # Try all splits from shortest candidate name to longest.
    segments = task_name_and_key.split(".")
    best_task_name = None
    best_remaining_key = None

    for i in range(1, len(segments) + 1):
        candidate_task_name = ".".join(segments[:i])
        remaining_key = ".".join(segments[i:])
        # Check exact match
        if candidate_task_name in task_name_to_index_mapping:
            best_task_name = candidate_task_name
            best_remaining_key = remaining_key
            break
        # Check suffix match
        suffix_matches = [
            task_name
            for task_name in all_task_names
            if task_name.endswith(f".{candidate_task_name}")
            or task_name == candidate_task_name
        ]
        if suffix_matches:
            best_task_name = candidate_task_name
            best_remaining_key = remaining_key
            break

    if best_task_name is None:
        # Fall back to shortest split (first segment as task name)
        best_task_name = segments[0]
        best_remaining_key = ".".join(segments[1:])

    return best_task_name, best_remaining_key, value_str


def _lookup_task(
    task_name: str,
    task_name_to_index_mapping: dict[str, list[int]],
) -> list[int]:
    """Resolve *task_name* to list of indices.

    Strategy: exact match first, then suffix match.  Errors on 0 or >1 matches.
    """

    # Exact match
    if task_name in task_name_to_index_mapping:
        indices = task_name_to_index_mapping[task_name]
        if len(indices) > 1:
            _raise_ambiguous(task_name, indices)
        return indices

    # Suffix match: e.g. "mmlu" matches "lm-evaluation-harness.mmlu"
    suffix_hits = [
        (name, idxs)
        for name, idxs in task_name_to_index_mapping.items()
        if name.endswith(f".{task_name}")
    ]

    if len(suffix_hits) == 0:
        raise ValueError(
            f"Task '{task_name}' not found in config. "
            f"Available tasks: {sorted(task_name_to_index_mapping.keys())}"
        )

    if len(suffix_hits) == 1:
        name, idxs = suffix_hits[0]
        if len(idxs) > 1:
            _raise_ambiguous(name, idxs)
        return idxs

    # Multiple suffix matches
    matched_names = [name for name, _ in suffix_hits]
    raise ValueError(
        f"Task '{task_name}' matches multiple tasks: {sorted(matched_names)}. "
        f"Use the full task name to disambiguate."
    )


def _raise_ambiguous(task_name: str, indices: list[int]) -> None:
    hint_lines = [f"  -o '++evaluation.tasks.{idx}.<rest>=<value>'" for idx in indices]
    raise ValueError(
        f"Task '{task_name}' appears at indices {indices} — cannot resolve unambiguously.\n"
        f"Use index-based override instead:\n" + "\n".join(hint_lines)
    )


def _delete_key(cfg: DictConfig, dotted_path: str) -> None:
    """Delete a key from *cfg* at *dotted_path*, mirroring Hydra's ``~`` semantics."""

    # rsplit(".", 1) splits on the last dot, giving at most 2 parts
    parts = dotted_path.rsplit(".", 1)
    # len == 2: path has at least one dot, e.g. "evaluation.tasks.0.xxx" → parent = "evaluation.tasks.0", key = "xxx"
    # len == 1: no dots at all, e.g. "evaluation_tasks_0" (very unlikely this function will be called with dotted_path without dots)
    if len(parts) == 2:
        parent_path, key = parts
        parent = OmegaConf.select(cfg, parent_path)
    else:
        parent = cfg
        key = parts[0]
    if parent is None:
        raise ValueError(f"Cannot delete '{dotted_path}': parent path does not exist.")
    if key not in parent:
        raise ValueError(f"Cannot delete '{dotted_path}': key '{key}' does not exist.")
    del parent[key]


def _parse_value(raw: str) -> object:
    """Parse a Hydra-style override value string into a Python object.

    Uses YAML parsing to handle strings, numbers, booleans, lists, and dicts.
    """
    if not raw:
        return ""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return parsed
