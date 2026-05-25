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
"""Output configuration schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrajectoriesConfig(BaseModel):
    """Configuration for the trajectories audit/enrichment report.

    The audit report (``trajectories_report.json``) is emitted unconditionally
    by the orchestrator finalize step when ``trajectories.jsonl`` exists. The
    optional ``enrich`` flag is reserved for splicing ``model_traffic.jsonl``
    captures into per-trial trajectories; it is wired in but the writer is
    not yet implemented (audit-only release).
    """

    model_config = ConfigDict(extra="forbid")

    enrich: bool = False


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "./eval_results"
    timestamped: bool = True
    progress_interval: float = 60.0
    report: list[str] = Field(default_factory=lambda: ["markdown"])
    export: list[str] = Field(default_factory=list)
    export_config: dict[str, Any] = Field(default_factory=dict)
    trajectories: TrajectoriesConfig = Field(default_factory=TrajectoriesConfig)
