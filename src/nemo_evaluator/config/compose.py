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
"""YAML config composition: defaults lists, config groups, base inheritance.

Provides Hydra-like config composition without external dependencies:
  - ``defaults:`` lists reference reusable fragments or base configs
  - Config groups: directory name maps to top-level key (e.g. clusters/ -> cluster:)
  - ``_base_:`` inheritance for variant configs (e.g. 24a extends 24)
  - ``${.path.to.val}`` self-referencing interpolation (resolved after merge)
  - ``null`` values delete keys inherited from a base
  - Underscore-prefixed top-level keys (e.g. ``_common``, ``_model``) are
    user-defined variables: available as ``${._common.field}`` self-references,
    then stripped before Pydantic validation.

All existing flat YAML configs continue to work unchanged — ``defaults:`` is optional.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_GROUP_KEY_MAP = {"clusters": "cluster", "cluster": "cluster"}

_SELF_REF_RE = re.compile(r"\$\{\.([a-zA-Z0-9_.]+)\}")


def compose_config(config_path: str | Path) -> dict[str, Any]:
    """Load, compose, and return a merged raw dict (before Pydantic validation).

    Callers should pass the result to ``parse_eval_config()`` afterwards.
    """
    config_path = Path(config_path).resolve()
    search_paths = _build_search_paths(config_path)
    raw = _load_and_compose(config_path, search_paths, _seen=set())
    raw = _resolve_self_refs(raw, raw)
    _strip_private_keys(raw)
    return raw


# ── Core composition ─────────────────────────────────────────────────────


def _load_and_compose(
    config_path: Path,
    search_paths: list[Path],
    *,
    _seen: set[Path],
) -> dict[str, Any]:
    """Load a YAML file and recursively resolve its ``defaults:`` list."""
    resolved = config_path.resolve()
    if resolved in _seen:
        raise ValueError(f"Circular config reference: {config_path}")
    _seen.add(resolved)

    raw = yaml.safe_load(config_path.read_text()) or {}
    defaults = raw.pop("defaults", None)
    if defaults is None:
        return raw

    self_idx: int | None = None
    for i, entry in enumerate(defaults):
        if entry == "_self_":
            self_idx = i
            break
    if self_idx is not None:
        defaults.pop(self_idx)
    else:
        self_idx = len(defaults)

    layers: list[dict[str, Any]] = []
    for entry in defaults:
        if isinstance(entry, dict) and "_base_" in entry:
            base_path = _find_base(entry["_base_"], config_path, search_paths)
            layers.append(_load_and_compose(base_path, search_paths, _seen=_seen))
        elif isinstance(entry, str):
            layers.append(_load_fragment(entry, search_paths))
        else:
            raise ValueError(f"Invalid defaults entry: {entry!r}")

    layers.insert(self_idx, raw)

    result: dict[str, Any] = {}
    for layer in layers:
        result = _deep_merge(result, layer)

    _prune_nulls(result)
    return result


# ── Merge ─────────────────────────────────────────────────────────────────


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge. Lists and scalars: overlay wins. Dicts: recurse."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _prune_nulls(data: dict) -> None:
    """Remove keys explicitly set to ``null`` (deletion from base)."""
    to_delete = [k for k, v in data.items() if v is None]
    for k in to_delete:
        del data[k]
    for v in data.values():
        if isinstance(v, dict):
            _prune_nulls(v)


def _strip_private_keys(data: dict) -> None:
    """Remove underscore-prefixed top-level keys (user-defined variables).

    These keys serve as reusable anchors for ``${.path}`` self-references
    and are resolved before this function runs.  Only top-level keys are
    stripped; nested underscore keys are left intact as they may be
    legitimate Pydantic model fields.
    """
    for key in [k for k in data if isinstance(k, str) and k.startswith("_")]:
        del data[key]


# ── Fragment loading ──────────────────────────────────────────────────────


def _load_fragment(name: str, search_paths: list[Path]) -> dict[str, Any]:
    """Load a config-group fragment and wrap it under the correct top-level key."""
    parts = name.split("/", 1)
    if len(parts) == 2:
        group = parts[0]
        path = _find_file(name, search_paths)
        content = yaml.safe_load(path.read_text()) or {}
        top_key = _GROUP_KEY_MAP.get(group, group)
        return {top_key: content}
    path = _find_file(name, search_paths)
    return yaml.safe_load(path.read_text()) or {}


def _find_file(name: str, search_paths: list[Path]) -> Path:
    """Locate a fragment file across search paths, trying .yaml/.yml extensions."""
    for base in search_paths:
        for ext in ("", ".yaml", ".yml"):
            candidate = base / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Config fragment '{name}' not found in: {[str(p) for p in search_paths]}")


def _find_base(name: str, referrer: Path, search_paths: list[Path]) -> Path:
    """Locate a base config file (for ``_base_:`` references)."""
    return _find_file(name, [referrer.parent] + search_paths)


def _build_search_paths(config_path: Path) -> list[Path]:
    """Build ordered search paths for fragment resolution."""
    paths: list[Path] = []
    conf_dir = config_path.parent / "conf"
    if conf_dir.is_dir():
        paths.append(conf_dir)
    user_conf = Path.home() / ".config" / "nemo-evaluator" / "conf"
    if user_conf.is_dir():
        paths.append(user_conf)
    pkg_defaults = Path(__file__).parent / "defaults"
    if pkg_defaults.is_dir():
        paths.append(pkg_defaults)
    return paths


# ── Self-referencing interpolation ────────────────────────────────────────


def _resolve_self_refs(data: Any, root: dict) -> Any:
    """Replace ``${.dotted.path}`` references with values from the merged config."""
    if isinstance(data, str):

        def _replace(m: re.Match) -> str:
            val: Any = root
            for key in m.group(1).split("."):
                if isinstance(val, dict) and key in val:
                    val = val[key]
                else:
                    return m.group(0)
            return str(val) if not isinstance(val, str) else val

        return _SELF_REF_RE.sub(_replace, data)
    if isinstance(data, dict):
        return {k: _resolve_self_refs(v, root) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_self_refs(v, root) for v in data]
    return data
