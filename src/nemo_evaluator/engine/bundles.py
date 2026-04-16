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
"""Eval bundle discovery and matching utilities.

Used by both ``nel compare`` (single- and multi-benchmark) and
``nel gate`` (multi-benchmark quality gate).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_bundles(directory: Path) -> dict[str, Path]:
    """Find eval-*.json bundles, mapping benchmark_name to bundle_path.

    Supports flat (bundles in root) and nested (one subdir per benchmark)
    layouts.  Raises ValueError on duplicate benchmark names.
    """
    bundles: dict[str, Path] = {}

    def _register(p: Path) -> None:
        name = extract_benchmark_name(p)
        if name is None:
            return
        if name in bundles:
            raise ValueError(f"Duplicate benchmark {name!r} in {directory}: found in {bundles[name]} and {p}")
        bundles[name] = p

    for p in sorted(directory.glob("eval-*.json")):
        _register(p)

    for sub in sorted(directory.iterdir()):
        if not sub.is_dir():
            continue
        sub_bundles = sorted(sub.glob("eval-*.json"))
        if len(sub_bundles) == 1:
            _register(sub_bundles[0])
        elif len(sub_bundles) > 1:
            logger.warning(
                "Directory %s contains %d eval bundles; skipping.",
                sub,
                len(sub_bundles),
            )

    return bundles


def extract_benchmark_name(bundle_path: Path) -> str | None:
    """Read the benchmark name from an eval bundle JSON file."""
    try:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        return data.get("benchmark", {}).get("name")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read bundle %s: %s", bundle_path, e)
        return None


def match_bundles(
    base: dict[str, Path], cand: dict[str, Path]
) -> tuple[dict[str, tuple[Path, Path]], set[str], set[str]]:
    """Match baseline and candidate bundles by benchmark name.

    Returns (matched, baseline_only, candidate_only) where matched maps
    benchmark_name to (base_path, cand_path).
    """
    matched: dict[str, tuple[Path, Path]] = {}
    base_only = set(base) - set(cand)
    cand_only = set(cand) - set(base)
    for name in sorted(set(base) & set(cand)):
        matched[name] = (base[name], cand[name])
    return matched, base_only, cand_only
