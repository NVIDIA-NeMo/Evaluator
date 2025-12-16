# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Pydantic models for evaluation configurations."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EvaluationTaskConfig(BaseModel):
    """Configuration for a single evaluation task/benchmark."""

    name: str = Field(
        description="Task/benchmark name (e.g., 'ifeval', 'gpqa_diamond')"
    )
    container: Optional[str] = Field(
        default=None, description="Docker container override"
    )
    env_vars: Optional[Dict[str, str]] = Field(
        default=None, description="Task environment variables"
    )
    dataset_dir: Optional[str] = Field(
        default=None, description="Dataset directory path"
    )
    dataset_mount_path: Optional[str] = Field(
        default="/datasets", description="Dataset mount path in container"
    )
    nemo_evaluator_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Task-specific nemo-evaluator config"
    )


class EvaluationConfig(BaseModel):
    """Top-level evaluation configuration."""

    tasks: list[EvaluationTaskConfig] = Field(description="List of evaluation tasks")
    nemo_evaluator_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Global nemo-evaluator config"
    )
    env_vars: Optional[Dict[str, str]] = Field(
        default=None, description="Global environment variables"
    )
