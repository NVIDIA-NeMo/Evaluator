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
"""Predefined function-style scorers exposed as Metric-compatible scorers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator.environments.custom import scorer
from nemo_evaluator.scoring import MetricOutputSpec
from nemo_evaluator.scoring import ScorerInput
from nemo_evaluator.scoring.text import exact_match


class ParameterizedExactMatchConfig(BaseModel):
    """Configuration for the parameterized exact-match scorer."""

    model_config = ConfigDict(extra="forbid")

    case_sensitive: bool = Field(default=False, description="Whether comparison should preserve letter casing.")
    strip: bool = Field(default=True, description="Whether leading and trailing whitespace should be ignored.")


def parameterized_exact_match(sample: ScorerInput[ParameterizedExactMatchConfig]) -> dict[str, object]:
    response = sample.response
    target = str(sample.target)
    if sample.config.strip:
        response = response.strip()
        target = target.strip()
    if not sample.config.case_sensitive:
        response = response.casefold()
        target = target.casefold()
    return {"correct": response == target}


ExactMatchScorer = scorer(
    exact_match,
    metric_type="exact_match",
    outputs=[MetricOutputSpec.continuous_score("correct")],
)

ParameterizedExactMatchScorer = scorer(
    parameterized_exact_match,
    metric_type="parameterized_exact_match",
    outputs=[MetricOutputSpec.continuous_score("correct")],
    config_schema=ParameterizedExactMatchConfig,
)

__all__ = [
    "ExactMatchScorer",
    "ParameterizedExactMatchConfig",
    "ParameterizedExactMatchScorer",
    "parameterized_exact_match",
]
