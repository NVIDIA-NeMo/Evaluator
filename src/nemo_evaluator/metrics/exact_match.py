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
"""Class-based exact-match metric implementation."""

from __future__ import annotations

import re
import string
from typing import Literal

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator.scoring import MetricInput, MetricOutput, MetricOutputSpec, MetricResult

_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)

__all__ = ["ExactMatchMetric"]


class ExactMatchMetric(BaseModel):
    """Exact-match metric using the shared MetricInput -> MetricResult contract."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(description="Jinja template for the expected reference answer.")
    candidate: str | None = Field(
        default=None,
        description="Optional Jinja template for the candidate. Defaults to sample.output_text.",
    )
    type: Literal["exact-match"] = Field(default="exact-match", description="Stable metric type identifier.")

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("correct")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        item, sample = _template_payload_from_metric_input(input)
        context = _build_template_context(item, sample)
        reference = _render_template(self.reference, context)
        candidate = _render_template(self.candidate, context) if self.candidate is not None else sample.get("output_text")
        if not isinstance(reference, str):
            raise TypeError("ExactMatchMetric reference must render to a string.")
        if not isinstance(candidate, str):
            raise TypeError("ExactMatchMetric candidate must render to a string.")
        correct = 1.0 if _normalize(candidate) == _normalize(reference) else 0.0
        return MetricResult(outputs=[MetricOutput(name="correct", value=correct)])


def _template_payload_from_metric_input(input: MetricInput) -> tuple[dict[str, object], dict[str, object]]:
    item = dict(input.row.data)
    sample = dict(input.candidate.metadata)
    if input.candidate.output_text is not None:
        sample["output_text"] = input.candidate.output_text
    if input.candidate.response is not None:
        sample["response"] = input.candidate.response
    if input.candidate.trajectory is not None:
        sample["trajectory"] = input.candidate.trajectory
    return item, sample


def _build_template_context(item: dict[str, object], sample: dict[str, object]) -> dict[str, object]:
    return {**item, **sample, "item": item, "sample": sample}


def _render_template(template: str, context: dict[str, object]) -> str:
    return _JINJA_ENV.from_string(template).render(context)


def _normalize(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    return " ".join(value.split())
