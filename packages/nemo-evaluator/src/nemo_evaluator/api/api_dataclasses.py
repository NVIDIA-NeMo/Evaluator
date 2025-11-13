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


from enum import Enum
from typing import Any, Dict, Optional

import jinja2
from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator.adapters.adapter_config import AdapterConfig

# NOTE: For ApiEndpoint, EvaluationTarget, ConfigParams, and EvaluationConfig all fields
#       are Optional and default=None, because depending on the command run (run_eval or
#       ls) we either require them or don't. We also don't require user to provide all
#       of them. The framework.yml often provides the defaults.


class EndpointType(str, Enum):
    """EndpointType is used to determine appropriate URL, payload structure or native harness inference class"""

    UNDEFINED = "undefined"
    CHAT = "chat"
    COMPLETIONS = "completions"
    VLM = "vlm"
    EMBEDDING = "embedding"


class ApiEndpoint(BaseModel):
    """API endpoint configuration containing information on endpoint placement, targeted model name and adapter used before prompting endpoint."""

    model_config = ConfigDict(use_enum_values=True)

    api_key: Optional[str] = Field(
        description="Name of the env variable that stores API key for the model",
        default=None,
    )
    model_id: Optional[str] = Field(description="Name of the model", default=None)
    stream: Optional[bool] = Field(
        description="Whether responses should be streamed", default=None
    )
    type: Optional[EndpointType] = Field(
        description="The type of the target", default=None
    )
    url: Optional[str] = Field(description="Url of the model", default=None)

    adapter_config: Optional[AdapterConfig] = Field(
        description="Adapter configuration", default=None
    )


class EvaluationTarget(BaseModel):
    """Target configuration for API endpoints."""

    api_endpoint: Optional[ApiEndpoint] = Field(
        description="API endpoint to be used for evaluation", default=None
    )


class ConfigParams(BaseModel):
    """Parameters for evaluation execution."""

    limit_samples: Optional[int | float] = Field(
        description="Limit number of evaluation samples", default=None
    )
    max_new_tokens: Optional[int] = Field(
        description="Max tokens to generate", default=None
    )
    max_retries: Optional[int] = Field(
        description="Number of REST request retries", default=None
    )
    parallelism: Optional[int] = Field(
        description="Parallelism to be used", default=None
    )
    task: Optional[str] = Field(description="Name of the task", default=None)
    temperature: Optional[float] = Field(
        description="Float value between 0 and 1. temp of 0 indicates greedy decoding, where the token with highest prob is chosen. Temperature can't be set to 0.0 currently",
        default=None,
    )
    request_timeout: Optional[int] = Field(
        description="REST response timeout", default=None
    )
    top_p: Optional[float] = Field(
        description="Float value between 0 and 1; limits to the top tokens within a certain probability. top_p=0 means the model will only consider the single most likely token for the next prediction",
        default=None,
    )
    extra: Optional[Dict[str, Any]] = Field(
        description="Framework specific parameters to be used for evaluation",
        default_factory=dict,
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""

    output_dir: Optional[str] = Field(
        description="Directory to output the results", default=None
    )
    params: Optional[ConfigParams] = Field(
        description="Parameters to be used for evaluation", default=None
    )
    supported_endpoint_types: Optional[list[str]] = Field(
        description="Supported endpoint types like chat or completions", default=None
    )
    type: Optional[str] = Field(description="Type of the task", default=None)


class EvaluationMetadata(dict):
    """We put here various evaluation metadata that does not influence the evaluation."""

    pass


class Evaluation(BaseModel):
    command: str = Field(description="jinja template of the command to be executed")
    framework_name: str = Field(description="Name of the framework")
    pkg_name: str = Field(description="Name of the package")
    config: EvaluationConfig
    target: EvaluationTarget

    def render_command(self):
        values = self.model_dump()

        def recursive_render(tpl):
            prev = tpl
            while True:
                try:
                    curr = jinja2.Template(
                        prev, undefined=jinja2.StrictUndefined
                    ).render(values)
                    if curr != prev:
                        prev = curr
                    else:
                        return curr
                except jinja2.exceptions.UndefinedError as e:
                    raise ValueError(f"Missing required configuration field: {e}")

        return recursive_render(self.command)


class ScoreStats(BaseModel):
    """Stats for a score."""

    count: Optional[int] = Field(
        default=None,
        description="The number of values used for computing the score.",
    )
    sum: Optional[float] = Field(
        default=None,
        description="The sum of all values used for computing the score.",
    )
    sum_squared: Optional[float] = Field(
        default=None,
        description="The sum of the square of all values used for computing the score.",
    )
    min: Optional[float] = Field(
        default=None,
        description="The minimum of all values used for computing the score.",
    )
    max: Optional[float] = Field(
        default=None,
        description="The maximum of all values used for computing the score.",
    )
    mean: Optional[float] = Field(
        default=None,
        description="The mean of all values used for computing the score.",
    )
    variance: Optional[float] = Field(
        default=None,
        description="""This is the population variance, not the sample variance.""",
    )
    stddev: Optional[float] = Field(
        default=None,
        description="""This is the population standard deviation, not the sample standard deviation.""",
    )
    stderr: Optional[float] = Field(default=None, description="The standard error.")


class Score(BaseModel):
    """Atomic class that contains the value of particular metric and corresponding stats"""

    value: float = Field(description="The value/score produced on this metric")
    stats: ScoreStats = Field(description="Statistics associated with this metric")


class MetricResult(BaseModel):
    """Defines mapping from metric name to its scores."""

    scores: Dict[str, Score] = Field(
        default_factory=dict, description="Mapping from metric name to scores."
    )


class TaskResult(BaseModel):
    """Defines set of metrics that were calculated for particular task."""

    metrics: Dict[str, MetricResult] = Field(
        default_factory=dict,
        description="The value for all the metrics computed for the task",
    )


class GroupResult(BaseModel):
    """Some tasks can be grouped or logically split. This class defines result on grouping level."""

    groups: Optional[Dict[str, "GroupResult"]] = Field(
        default=None, description="The results for the subgroups."
    )
    metrics: Dict[str, MetricResult] = Field(
        default_factory=dict,
        description="The value for all the metrics computed for the group.",
    )


class EvaluationResult(BaseModel):
    """EvaluationResults bundles per-tasks and per-group results."""

    tasks: Optional[Dict[str, TaskResult]] = Field(
        default_factory=dict, description="The results at the task-level"
    )
    groups: Optional[Dict[str, GroupResult]] = Field(
        default_factory=dict, description="The results at the group-level"
    )
