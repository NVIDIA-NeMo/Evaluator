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
from typing import TYPE_CHECKING

import yaml
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

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
        return False

    try:
        int(first_segment)
    except ValueError:
        return True  # non-integer → sugar

    return False  # integer → pass through to Hydra


def apply_task_name_overrides(
    cfg: "DictConfig",
    task_overrides: list[str],
) -> "DictConfig":
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

    name_to_indices = _build_task_name_index(cfg)
    all_names = list(name_to_indices.keys())

    for override in task_overrides:
        prefix, key_value = _strip_hydra_prefix(override)
        # key_value looks like "evaluation.tasks.<task_name_and_rest>=<value>"
        rest = key_value[len(_TASKS_PREFIX) :]  # "<task_name_and_rest>=<value>"

        task_name, remaining_path, value_str = _resolve_task_segment(
            rest, name_to_indices, all_names
        )

        indices = _lookup_task(task_name, name_to_indices, all_names)

        for idx in indices:
            dotted_path = f"evaluation.tasks.{idx}"
            if remaining_path:
                dotted_path += f".{remaining_path}"
            value = _parse_value(value_str) if value_str is not None else None
            force_add = prefix == "++"
            OmegaConf.update(cfg, dotted_path, value, force_add=force_add)

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_task_name_index(cfg: "DictConfig") -> dict[str, list[int]]:
    """Return ``{task_name: [idx, …]}`` from ``cfg.evaluation.tasks``."""
    mapping: dict[str, list[int]] = defaultdict(list)
    tasks = cfg.evaluation.tasks
    for idx, task in enumerate(tasks):
        name = task.name if hasattr(task, "name") else str(task)
        mapping[name].append(idx)
    return dict(mapping)


def _strip_hydra_prefix(override: str) -> tuple[str, str]:
    """Return ``(prefix, key_value)`` where *prefix* is ``''``, ``'++'``, ``'+'``, or ``'~'``."""
    m = _HYDRA_PREFIX_RE.match(override)
    assert m is not None
    return (m.group(1) or ""), m.group(2)


def _resolve_task_segment(
    rest: str,
    name_to_indices: dict[str, list[int]],
    all_names: list[str],
) -> tuple[str, str, str | None]:
    """Given the part after ``evaluation.tasks.``, find the task name, remaining path, and value.

    For task names that contain dots (e.g. ``lm-evaluation-harness.mmlu``), we try
    progressively longer candidate names and pick the longest match.

    Returns:
        (task_name, remaining_path, value_str)
        *value_str* is ``None`` when there is no ``=`` in the override.
    """
    # Split off the value first (everything after the first unquoted '=')
    value_str: str | None = None
    eq_idx = rest.find("=")
    if eq_idx != -1:
        value_str = rest[eq_idx + 1 :]
        rest = rest[:eq_idx]

    # rest is now "<task_name_segments>.<remaining_path>" (all dots).
    # Try all splits from longest candidate name to shortest.
    segments = rest.split(".")
    best_name: str | None = None
    best_remaining: str | None = None

    for i in range(len(segments), 0, -1):
        candidate = ".".join(segments[:i])
        remaining = ".".join(segments[i:])
        # Check exact match
        if candidate in name_to_indices:
            best_name = candidate
            best_remaining = remaining
            break
        # Check suffix match
        suffix_matches = [n for n in all_names if n.endswith(f".{candidate}") or n == candidate]
        if suffix_matches:
            best_name = candidate
            best_remaining = remaining
            break

    if best_name is None:
        # Fall back to shortest split (first segment as task name)
        best_name = segments[0]
        best_remaining = ".".join(segments[1:])

    return best_name, best_remaining or "", value_str


def _lookup_task(
    task_name: str,
    name_to_indices: dict[str, list[int]],
    all_names: list[str],
) -> list[int]:
    """Resolve *task_name* to list of indices.

    Strategy: exact match first, then suffix match.  Errors on 0 or >1 matches.
    """
    # Exact match
    if task_name in name_to_indices:
        indices = name_to_indices[task_name]
        if len(indices) > 1:
            _raise_ambiguous(task_name, indices, all_names)
        return indices

    # Suffix match: e.g. "mmlu" matches "lm-evaluation-harness.mmlu"
    suffix_hits = [
        (name, idxs)
        for name, idxs in name_to_indices.items()
        if name.endswith(f".{task_name}")
    ]

    if len(suffix_hits) == 0:
        available = sorted(all_names)
        raise ValueError(
            f"Task '{task_name}' not found in config. "
            f"Available tasks: {available}"
        )

    if len(suffix_hits) == 1:
        name, idxs = suffix_hits[0]
        if len(idxs) > 1:
            _raise_ambiguous(name, idxs, all_names)
        return idxs

    # Multiple suffix matches
    matched_names = [name for name, _ in suffix_hits]
    raise ValueError(
        f"Task '{task_name}' matches multiple tasks: {sorted(matched_names)}. "
        f"Use the full task name to disambiguate."
    )


def _raise_ambiguous(
    task_name: str, indices: list[int], all_names: list[str]
) -> None:
    hint_lines = [
        f"  -o '++evaluation.tasks.{idx}.<rest>=<value>'"
        for idx in indices
    ]
    raise ValueError(
        f"Task '{task_name}' appears at indices {indices} — cannot resolve unambiguously.\n"
        f"Use index-based override instead:\n" + "\n".join(hint_lines)
    )


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
