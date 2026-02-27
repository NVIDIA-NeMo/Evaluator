# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Sandbox protocol and shared types.

Any sandbox backend (ECS Fargate, local Docker, Modal, etc.) implements
the :class:`Sandbox` protocol so harnesses can be backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self


@dataclass
class ExecResult:
    """Result of a command executed inside a sandbox."""

    stdout: str
    stderr: str
    return_code: int


class Sandbox(Protocol):
    """Minimal contract every sandbox backend must satisfy."""

    def start(self, *, force_build: bool = False) -> None: ...

    def stop(self) -> None: ...

    def exec(self, command: str, timeout_sec: float = 180) -> ExecResult: ...

    def upload(self, local_path: Path, remote_path: str) -> None: ...

    def download(self, remote_path: str, local_path: Path) -> None: ...

    @property
    def is_running(self) -> bool: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...
