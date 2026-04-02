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
"""Base classes for evaluation environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import ImageBuildRequest, Sandbox, SandboxSpec


@dataclass
class SeedResult:
    prompt: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] | None = None
    system: str | None = None
    images: list[str] | None = None
    sandbox_spec: SandboxSpec | None = None
    verify_sandbox_spec: SandboxSpec | None = None
    capture_cmd: str | None = None
    apply_cmd: str | None = None


@dataclass
class VerifyResult:
    reward: float
    extracted_answer: str | None = None
    scoring_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalEnvironment(ABC):
    """Base for all evaluation environments -- BYOB, harness wrappers, remote, etc."""

    name: str = "unnamed"

    def __init__(self) -> None:
        self._dataset: list[dict[str, Any]] = []

    @property
    def dataset(self) -> list[dict[str, Any]]:
        return self._dataset

    @dataset.setter
    def dataset(self, value: list[dict[str, Any]]) -> None:
        self._dataset = value

    def __len__(self) -> int:
        return len(self._dataset)

    async def dataset_size(self) -> int:
        return len(self._dataset)

    @abstractmethod
    async def seed(self, idx: int) -> SeedResult: ...

    @abstractmethod
    async def verify(
        self,
        response: str,
        expected: str,
        sandbox: Sandbox | None = None,
        **metadata: Any,
    ) -> VerifyResult: ...

    async def sandbox_specs(self) -> list[SandboxSpec] | None:
        """Return sandbox specs for all problems (for pre-pull).

        Override to enable pre-pull. Iterates the dataset and computes
        image names without full prompt construction or API calls.
        Returns None (lazy pull) by default.
        """
        return None

    async def run_batch(self, solver: Any = None, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Optional batch execution. Override for environments that own the full loop
        (e.g. ContainerEnvironment). Return an artifact bundle dict,
        or None to fall through to the standard seed/solve/verify loop."""
        return None

    async def prepare(self) -> None:
        """Lifecycle hook for non-image provisioning before evaluation starts.

        Override for: downloading datasets, refreshing auth tokens, warming
        caches, etc.  Image provisioning is handled separately via
        :meth:`image_build_requests`.  Default is no-op.
        """

    async def image_build_requests(self) -> list[ImageBuildRequest] | None:
        """Declare images that must be built before evaluation starts.

        Returns a list of :class:`ImageBuildRequest` objects (each carrying
        the image names and a Docker build function), or ``None`` when no
        custom builds are needed.  The :class:`SandboxManager` consumes
        these and handles backend-specific conversion (SIF, ECR push, etc.).
        """
        return None

    async def close(self) -> None:
        """Clean up resources. Override for environments that hold connections."""
