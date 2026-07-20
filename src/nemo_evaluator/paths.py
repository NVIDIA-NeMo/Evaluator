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
"""Repo-relative path constants and lookups.

Resolves locations that live outside the installed package (e.g. vendored
registry overrides at the repo root) without depending on the process
working directory.
"""

from __future__ import annotations

from pathlib import Path

# Three parents up from src/nemo_evaluator/paths.py reaches the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def local_registry_override_dir() -> Path:
    """Harbor registry overrides directory.

    In built wheels the override shards are force-included alongside
    the package as ``nemo_evaluator/_registry_overrides``; in editable
    installs the data still lives at the repo root.  The wheel path
    wins when present so containerised orchestrators (Dockerfile.harbor
    `pip install .`) resolve overrides without depending on
    ``Path(__file__).parents[2]``, which doesn't point at the repo
    root outside of editable installs.
    """
    in_package = Path(__file__).parent / "_registry_overrides"
    if in_package.is_dir():
        return in_package
    return REPO_ROOT / "harbor_datasets" / "registry_overrides"


def local_task_overlay_dir() -> Path:
    """Harbor task-overlays directory.

    Same wheel-vs-editable resolution as
    :func:`local_registry_override_dir`: built wheels force-include the
    overlay tree as ``nemo_evaluator/_task_overlays``; editable installs
    read the repo-root copy.
    """
    in_package = Path(__file__).parent / "_task_overlays"
    if in_package.is_dir():
        return in_package
    return REPO_ROOT / "harbor_datasets" / "task_overlays"
