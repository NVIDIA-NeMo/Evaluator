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
"""Artifact Access helpers."""

from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def chmod_a_rx(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.chmod(path.stat().st_mode | 0o555)
    except OSError as exc:
        logger.warning("Failed to apply artifact access to %s: %s", path, exc)


def _default_parent_stop(root_dir: Path) -> Path | None:
    home = Path.home()
    try:
        resolved_root = root_dir.resolve()
        resolved_home = home.resolve()
    except OSError:
        return None

    if resolved_root == resolved_home or resolved_home in resolved_root.parents:
        return resolved_home
    return None


def _parent_chain(root_dir: Path, *, stop_at: Path | None = None) -> Iterator[Path]:
    try:
        root = root_dir.resolve()
    except OSError:
        root = root_dir.absolute()

    stop = stop_at or _default_parent_stop(root)
    if stop is not None:
        try:
            stop = stop.resolve()
        except OSError:
            stop = stop.absolute()
        if root == stop:
            return

    current = root.parent
    while True:
        yield current
        if current == stop or current == current.parent:
            return
        current = current.parent


def warn_if_parent_chain_lacks_execute(root_dir: Path, *, stop_at: Path | None = None) -> None:
    blocked: list[Path] = []
    for path in _parent_chain(root_dir, stop_at=stop_at):
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            logger.warning("Failed to inspect artifact access parent %s: %s", path, exc)
            continue

        if mode & 0o111 != 0o111:
            blocked.append(path)

    if blocked:
        logger.warning(
            "Artifact access may be blocked because parent directories lack a+x: %s. "
            "Grant execute permission on these directories so other cluster users can traverse to %s; "
            "read permission is not required for traversal when artifact paths are known.",
            ", ".join(str(path) for path in blocked),
            root_dir,
        )


def apply_artifact_access(
    *,
    root_dir: Path,
    artifact_dirs: set[Path],
    artifact_files: set[Path],
) -> None:
    chmod_a_rx(root_dir)
    warn_if_parent_chain_lacks_execute(root_dir)
    for path in sorted(artifact_dirs):
        chmod_a_rx(path)
    for path in sorted(artifact_files):
        chmod_a_rx(path)
