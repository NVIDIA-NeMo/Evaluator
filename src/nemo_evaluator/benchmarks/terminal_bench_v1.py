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
"""Terminal-Bench v1 — auto-download + map to Harbor format on first use.

Registers ``terminal-bench-v1`` as a built-in benchmark.  On first run the
upstream repo (``laude-institute/terminal-bench``) is cloned, tasks are
converted from v1 layout to Harbor format via ``TerminalBenchMapper``, and the
result is cached in ``harbor_datasets/terminal-bench@1.0/``.

Config usage::

    benchmarks:
      - name: terminal-bench-v1
        solver:
          type: harbor
          service: nemotron
          agent: terminus-2
        sandbox:
          type: ecs_fargate
          ...
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from nemo_evaluator.environments.harbor import HarborEnvironment
from nemo_evaluator.environments.registry import register

logger = logging.getLogger(__name__)

_GIT_URL = "https://github.com/laude-institute/terminal-bench.git"
_DATASET_DIR_NAME = "terminal-bench@1.0"
_XP_TASK = "install-windows-xp"
_XP_ENTRYPOINT = 'ENTRYPOINT [ "supervisord", "-c", "/etc/supervisor/supervisord.conf" ]'
_XP_WRAPPER_ENTRYPOINT = 'ENTRYPOINT ["/nel-entrypoint.sh"]'
_XP_DOCKERFILE_PATCH = f"""\
# NEL patch: upstream compose passes no command, but NEL sandboxes do.
# The wrapper starts supervisord, then execs "$@".
COPY nel-entrypoint.sh /nel-entrypoint.sh
RUN chmod +x /nel-entrypoint.sh
{_XP_WRAPPER_ENTRYPOINT}"""
_XP_WRAPPER = """\
#!/bin/sh
set -e

supervisord -c /etc/supervisor/supervisord.conf &
pid=$!

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid"
    exit $?
fi

exec "$@"
"""


def _git(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            check=True,
            capture_output=True,
            **kwargs,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(
            "git failed: %s\nstderr: %s",
            " ".join(cmd),
            exc.stderr,
        )
        raise


def _find_tasks_dir(repo: Path) -> Path | None:
    """Locate the directory containing v1 tasks (subdirs with ``task.yaml``)."""
    for candidate in [repo, repo / "tasks", repo / "benchmarks", repo / "original-tasks"]:
        if candidate.is_dir() and any((d / "task.yaml").exists() for d in candidate.iterdir() if d.is_dir()):
            return candidate
    return None


def _patch_install_windows_xp_entrypoint(output_dir: Path) -> None:
    """Patch upstream's service ENTRYPOINT into a command-friendly wrapper."""
    env_dir = output_dir / _XP_TASK / "environment"
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        return

    text = dockerfile.read_text(encoding="utf-8")
    already_patched = _XP_WRAPPER_ENTRYPOINT in text
    needs_patch = _XP_ENTRYPOINT in text
    if not (already_patched or needs_patch):
        logger.warning("%s Dockerfile did not match expected supervisord ENTRYPOINT; skipping patch", _XP_TASK)
        return

    wrapper = env_dir / "nel-entrypoint.sh"
    if not wrapper.exists() or wrapper.read_text(encoding="utf-8") != _XP_WRAPPER:
        wrapper.write_text(_XP_WRAPPER, encoding="utf-8")
    if needs_patch:
        dockerfile.write_text(text.replace(_XP_ENTRYPOINT, _XP_DOCKERFILE_PATCH, 1), encoding="utf-8")


def _ensure_dataset(datasets_dir: str | None = None) -> Path:
    """Return path to the mapped dataset, downloading + mapping if needed."""
    from harbor.mappers.terminal_bench import TerminalBenchMapper

    base = Path(datasets_dir or os.environ.get("HARBOR_DATASETS_DIR", "./harbor_datasets"))
    output_dir = base / _DATASET_DIR_NAME
    marker = output_dir / ".tbv1_mapped"

    if marker.exists():
        _patch_install_windows_xp_entrypoint(output_dir)
        n = sum(1 for d in output_dir.iterdir() if d.is_dir() and (d / "instruction.md").exists())
        logger.info("terminal-bench-v1: %d cached tasks", n)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "repo"
        logger.info("Cloning Terminal-Bench v1 from %s …", _GIT_URL)
        _git(["git", "clone", "--depth", "1", _GIT_URL, str(clone_dir)])

        tasks_root = _find_tasks_dir(clone_dir)
        if tasks_root is None:
            raise RuntimeError(f"No terminal-bench v1 task directories found in {clone_dir}")

        all_tasks = sorted(
            (d for d in tasks_root.iterdir() if d.is_dir() and (d / "task.yaml").exists()),
            key=lambda d: d.name,
        )

        logger.info("terminal-bench-v1: mapping %d tasks …", len(all_tasks))

        mapper = TerminalBenchMapper()
        mapped, failed = 0, 0
        for src in all_tasks:
            dst = output_dir / src.name
            if dst.exists():
                mapped += 1
                continue
            try:
                mapper._map_task(src, dst)
                mapped += 1
            except Exception:
                logger.warning("Failed to map task %s", src.name, exc_info=True)
                failed += 1

        _patch_install_windows_xp_entrypoint(output_dir)
        logger.info(
            "terminal-bench-v1: mapped %d tasks (%d failed)",
            mapped,
            failed,
        )
        marker.touch()

    return output_dir


@register("terminal-bench-v1")
class TerminalBenchV1(HarborEnvironment):
    """Terminal-Bench 1.0 — auto-downloaded and mapped to Harbor format."""

    def __init__(self, num_examples: int | None = None) -> None:
        dataset_path = _ensure_dataset()
        super().__init__(dataset_path=dataset_path, num_examples=num_examples)
