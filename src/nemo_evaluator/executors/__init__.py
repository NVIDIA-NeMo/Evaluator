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
"""Executor abstraction: run/status/stop/logs/resume for each execution backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.config import EvalConfig
    from nemo_evaluator.run_store import RunMeta


@dataclass
class ProcessState:
    executor: str
    running: bool
    details: dict


class Executor(ABC):
    name: str = ""

    @abstractmethod
    def run(
        self,
        config: "EvalConfig",
        *,
        dry_run: bool = False,
        resume: bool = False,
        background: bool = False,
        submit: bool = False,
    ) -> None: ...

    @abstractmethod
    def status(self, output_dir: str | Path) -> ProcessState: ...

    @abstractmethod
    def stop(self, output_dir: str | Path) -> bool: ...

    @staticmethod
    @abstractmethod
    def detect(output_dir: str | Path) -> bool: ...

    def logs(self, output_dir: str | Path, *, follow: bool = False, tail: int | None = None) -> str | None:
        """Return log content. Default: read nel_eval.log."""
        log = Path(output_dir) / "nel_eval.log"
        if not log.exists():
            return None
        text = log.read_text(encoding="utf-8")
        if tail:
            lines = text.splitlines()
            return "\n".join(lines[-tail:])
        return text

    def resume_run(self, run_meta: "RunMeta", **kwargs) -> None:
        """Resume a stopped/failed run. Override per executor."""
        raise NotImplementedError(f"{self.name} executor does not support resume")


_REGISTRY: dict[str, Executor] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    from nemo_evaluator.executors.local_executor import LocalExecutor

    _REGISTRY["local"] = LocalExecutor()

    from nemo_evaluator.executors.docker_executor import DockerExecutor

    _REGISTRY["docker"] = DockerExecutor()

    from nemo_evaluator.executors.slurm_executor import SlurmExecutor

    _REGISTRY["slurm"] = SlurmExecutor()


def get_executor(name: str) -> Executor:
    """Get an executor by name (matches ``cluster.type``)."""
    _ensure_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown executor {name!r}. Available: {available}")
    return _REGISTRY[name]


def detect_executor(output_dir: str | Path) -> Executor | None:
    """Detect which executor produced the given output directory."""
    _ensure_loaded()
    for ex in _REGISTRY.values():
        if ex.detect(output_dir):
            return ex
    return None
