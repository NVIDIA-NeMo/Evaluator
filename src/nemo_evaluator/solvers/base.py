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
"""Solver protocol: pluggable inference strategy for the eval loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ErrorKind(Enum):
    NONE = "none"
    GRACEFUL = "graceful"
    INFRA = "infra_error"
    TOOL_INFRA = "tool_infra"
    SYSTEM = "system"


@dataclass
class SolveResult:
    response: str
    model_response: ModelResponse | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # Pre-computed reward from solvers that bundle verification (e.g. gym /run).
    # When set, the eval loop skips the separate verify phase.
    reward: float | None = None
    scoring_details: dict[str, Any] = field(default_factory=dict)
    # Non-None when the solver caught an internal error and returned a
    # degraded result instead of raising.  The eval loop uses this to
    # decide whether to abort (skip_failed=False) or continue.
    error: str | None = None
    error_kind: ErrorKind = ErrorKind.NONE


class Solver(Protocol):
    """Solvers MAY also accept ``sandbox: Sandbox | None = None`` in solve().

    The eval loop detects this via ``_solver_accepts_sandbox()`` introspection
    and passes the sandbox when available.
    """

    async def solve(self, task: SeedResult) -> SolveResult: ...
