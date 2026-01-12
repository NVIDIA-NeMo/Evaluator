from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class NemoSandboxCommand:
    """
    TB-independent command model for driving an interactive terminal.

    Mirrors the fields terminal-bench agents commonly use (but does not depend on TB).
    """

    command: str
    min_timeout_sec: float = 0.0
    max_timeout_sec: float = 180.0
    block: bool = False
    append_enter: bool = True


@runtime_checkable
class NemoSandboxSession(Protocol):
    """
    Minimal session API used by agents/harnesses (tmux-like).
    """

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None: ...

    def send_command(self, command: NemoSandboxCommand) -> None: ...

    def capture_pane(self, capture_entire: bool = False) -> str: ...

    def is_session_alive(self) -> bool: ...

    def get_incremental_output(self) -> str: ...

    def get_asciinema_timestamp(self) -> float: ...

    def copy_to_sandbox(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None: ...


class NemoEvaluatorSandbox(ABC):
    """
    Abstract factory for evaluator sandboxes.

    Implementations are responsible for provisioning an isolated environment and exposing
    a tmux-like session API for agents to interact with it.
    """

    @classmethod
    @abstractmethod
    def spin_up(
        cls,
        *,
        task_id: str,
        trial_name: str,
        run_id: str,
        pre_upload_paths: Iterable[Path] | None = None,
        upload_dest_dir: str | None = None,
        **kwargs,
    ) -> ContextManager["NemoEvaluatorSandbox"]:
        raise NotImplementedError

    @abstractmethod
    def create_session(
        self,
        session_name: str,
        is_active_stream: bool = False,
        as_configured_user: bool = True,
    ) -> NemoSandboxSession:
        raise NotImplementedError

    @abstractmethod
    def copy_to_sandbox(
        self,
        *,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
