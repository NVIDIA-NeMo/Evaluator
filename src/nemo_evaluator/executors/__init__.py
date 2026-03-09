"""Executor abstraction: run/status/stop for each execution backend."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nemo_evaluator.eval.config import EvalConfig


@dataclass
class ProcessState:
    executor: str
    running: bool
    details: dict


class Executor(Protocol):
    name: str

    def run(
        self,
        config: "EvalConfig",
        *,
        dry_run: bool = False,
        resume: bool = False,
        background: bool = False,
        submit: bool = False,
    ) -> None: ...

    def status(self, output_dir: str | Path) -> ProcessState: ...

    def stop(self, output_dir: str | Path) -> bool: ...

    @staticmethod
    def detect(output_dir: str | Path) -> bool: ...


_REGISTRY: dict[str, Executor] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

    from nemo_evaluator.executors.local import LocalExecutor
    _REGISTRY["local"] = LocalExecutor()

    from nemo_evaluator.executors.docker import DockerExecutor
    _REGISTRY["docker"] = DockerExecutor()

    from nemo_evaluator.executors.slurm import SlurmExecutor
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
