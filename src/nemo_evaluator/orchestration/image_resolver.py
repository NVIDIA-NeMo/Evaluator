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
"""Resolve container images from ``containers.toml`` for SLURM and Docker execution."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import tomllib


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    from nemo_evaluator.resources import containers_toml_path

    return tomllib.loads(containers_toml_path().read_text(encoding="utf-8"))


def _default_base() -> str:
    """Return the base image from containers.toml."""
    mapping = _load_mapping()
    defaults = mapping.get("defaults", {})
    return f"{defaults['registry']}:{defaults['tag']}"


def _variant_image(base: str, variant: str) -> str:
    """Append variant suffix to base image tag (e.g. ``nel:v2`` → ``nel:v2-lm-eval``)."""
    if variant == "base" or not variant:
        return base
    if base.endswith(f"-{variant}"):
        return base
    if ":" in base:
        return f"{base}-{variant}"
    return f"{base}:{variant}"


def scheme_to_variant(bench_name: str) -> str:
    """Map a benchmark URI to its container variant."""
    mapping = _load_mapping()
    schemes = mapping.get("schemes", {})
    for scheme, var in schemes.items():
        if bench_name.startswith(scheme):
            return var
    return "base"


def resolve_eval_image(bench_name: str, base_override: str | None = None) -> str:
    """Return the container image for a benchmark URI. ``base_override`` bypasses lookup."""
    if base_override:
        return base_override
    variant = scheme_to_variant(bench_name)
    if not variant:
        return ""
    return _variant_image(_default_base(), variant)


def resolve_deployment_image(deploy_type: str) -> str:
    """Return the default deployment container for a server type."""
    mapping = _load_mapping()
    return mapping.get("deployment", {}).get(deploy_type, "")


def default_base_image(base_override: str | None = None) -> str:
    """Return the base eval image."""
    return base_override or _default_base()
