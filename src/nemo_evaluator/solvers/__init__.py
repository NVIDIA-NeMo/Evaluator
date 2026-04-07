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
"""Solver implementations: pluggable inference strategies for the eval loop."""

from nemo_evaluator.solvers.base import Solver, SolveResult
from nemo_evaluator.solvers.chat import ChatSolver
from nemo_evaluator.solvers.completion import CompletionSolver
from nemo_evaluator.solvers.nat import NatSolver
from nemo_evaluator.solvers.openclaw import OpenClawSolver
from nemo_evaluator.solvers.vlm import VLMSolver

__all__ = [
    "ChatSolver",
    "CompletionSolver",
    "HarborSolver",
    "NatSolver",
    "OpenClawSolver",
    "ReActSolver",
    "Solver",
    "SolveResult",
    "VLMSolver",
]


def __getattr__(name: str):
    if name == "HarborSolver":
        from nemo_evaluator.solvers.harbor import HarborSolver

        return HarborSolver
    if name == "ReActSolver":
        from nemo_evaluator.solvers.react import ReActSolver

        return ReActSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
