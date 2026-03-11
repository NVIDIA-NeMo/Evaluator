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
"""Pydantic validation models for NEL config sections.

These models mirror the OmegaConf dataclass schemas but add runtime validation
with ``extra="forbid"``, turning unknown keys into actionable errors.

``Field(description=...)`` annotations populate the JSON schema
(``model_json_schema()``) and documentation tools automatically.

Sections covered (more can be added as needed):
- ``execution.mounts``  — :class:`MountsModel`
- ``evaluation``        — :class:`EvaluationModel`
- ``evaluation.tasks``  — :class:`TaskModel`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MountsModel(BaseModel):
    """Validation model for the ``execution.mounts`` section.

    Extra fields are forbidden — unknown keys indicate a typo or stale config.
    """

    model_config = ConfigDict(extra="forbid")

    deployment: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra volume mounts for the deployment (server) container. "
            "Mapping of host_path → container_path."
        ),
    )
    evaluation: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra volume mounts for the evaluation (client) container. "
            "Mapping of host_path → container_path."
        ),
    )
    mount_home: bool = Field(
        default=True,
        description=(
            "Whether to bind-mount the user's home directory into containers. "
            "Only respected by the SLURM executor — has no effect on local execution. "
            "Recommended to set false on shared clusters."
        ),
    )


class TaskModel(BaseModel):
    """Validation model for a single entry in ``evaluation.tasks``.

    Extra fields are forbidden — unknown keys indicate a typo or stale config.
    ``nemo_evaluator_config`` is a free-form pass-through and accepts any content.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="",
        description=(
            "Fully-qualified task name in ``harness.task`` format, "
            "e.g. ``lm-evaluation-harness.ifeval``. "
            "Run ``nel ls tasks`` to list all available tasks."
        ),
    )
    container: Optional[str] = Field(
        default=None,
        description=(
            "Docker image override for this task. "
            "Defaults to the container registered in the task registry."
        ),
    )
    endpoint_type: Optional[str] = Field(
        default=None,
        description="API endpoint type: ``chat`` or ``completions``. Auto-detected when omitted.",
    )
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Task-level environment variables. Overrides global and evaluation-level vars. "
            "Values must use ``host:``, ``lit:``, or ``runtime:`` prefixes."
        ),
    )
    nemo_evaluator_config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task-level config passed through to ``nemo-evaluator``. "
            "Arbitrary nesting is allowed — this is a free-form pass-through."
        ),
    )
    dataset_dir: Optional[str] = Field(
        default=None,
        description=(
            "Host path to a local dataset directory. "
            "Bind-mounted into the container at ``dataset_mount_path``."
        ),
    )
    dataset_mount_path: str = Field(
        default="/datasets",
        description="Container path where ``dataset_dir`` is mounted. Defaults to ``/datasets``.",
    )
    pre_cmd: str = Field(
        default="",
        description=(
            "Shell command executed inside the container before the evaluator starts. "
            "Requires ``NEMO_EVALUATOR_TRUST_PRE_CMD=1``."
        ),
    )
    post_cmd: str = Field(
        default="",
        description=(
            "Shell command executed inside the container after the evaluator finishes. "
            "Runs even if evaluation fails (EXIT trap)."
        ),
    )


class EvaluationModel(BaseModel):
    """Validation model for the top-level ``evaluation`` section.

    Extra fields are forbidden — unknown keys indicate a typo or stale config.
    ``nemo_evaluator_config`` is a free-form pass-through and accepts any content.
    """

    model_config = ConfigDict(extra="forbid")

    tasks: List[TaskModel] = Field(
        default_factory=list,
        description="Ordered list of evaluation tasks to run. Each entry must have at least a ``name``.",
    )
    nemo_evaluator_config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Global ``nemo-evaluator`` config applied to all tasks. "
            "Task-level ``nemo_evaluator_config`` is merged on top (task wins)."
        ),
    )
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Evaluation-level environment variables applied to all tasks. "
            "Overrides global ``env_vars``; overridden by task-level ``env_vars``. "
            "Values must use ``host:``, ``lit:``, or ``runtime:`` prefixes."
        ),
    )
    pre_cmd: str = Field(
        default="",
        description=(
            "Global pre-command executed in all task containers. "
            "Task-level ``pre_cmd`` overrides this per task."
        ),
    )
    post_cmd: str = Field(
        default="",
        description=(
            "Global post-command executed in all task containers after the evaluator. "
            "Task-level ``post_cmd`` overrides this per task. Runs even if evaluation fails."
        ),
    )
