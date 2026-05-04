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
"""Shared MetricInput -> MetricResult runtime contract."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

from nemo_evaluator.scoring.types import ScorerInput

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox


ConfigT = TypeVar("ConfigT", bound=Mapping[str, object] | BaseModel)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DatasetRow(BaseModel):
    """Original benchmark dataset row plus optional stable row identity."""

    model_config = ConfigDict(extra="forbid")

    row_index: int | None = None
    data: dict[str, object]


class CandidateOutput(BaseModel):
    """Candidate output being scored for one dataset row."""

    model_config = ConfigDict(extra="forbid")

    output_text: str | None = None
    response: object | None = None
    trajectory: object | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class MetricInput(BaseModel):
    """Complete per-row scoring input passed to a metric."""

    model_config = ConfigDict(extra="forbid")

    row: DatasetRow
    candidate: CandidateOutput


class ContinuousScore(RootModel[float]):
    """Continuous numeric metric value."""


class DiscreteScore(RootModel[int]):
    """Discrete numeric metric value."""


class Label(RootModel[str]):
    """String label metric value."""


class BooleanValue(RootModel[bool]):
    """Boolean metric value."""


class MetricOutputSpec(BaseModel, Generic[SchemaT]):
    """Schema for one named value emitted by a metric."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str | None = None
    value_schema: type[SchemaT]

    @field_validator("name")
    @classmethod
    def _name_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("metric output name must not be empty")
        return value

    @staticmethod
    def continuous_score(name: str, description: str | None = None) -> MetricOutputSpec[ContinuousScore]:
        return MetricOutputSpec[ContinuousScore](name=name, description=description, value_schema=ContinuousScore)

    @staticmethod
    def discrete_score(name: str, description: str | None = None) -> MetricOutputSpec[DiscreteScore]:
        return MetricOutputSpec[DiscreteScore](name=name, description=description, value_schema=DiscreteScore)

    @staticmethod
    def label(name: str, description: str | None = None) -> MetricOutputSpec[Label]:
        return MetricOutputSpec[Label](name=name, description=description, value_schema=Label)

    @staticmethod
    def boolean(name: str, description: str | None = None) -> MetricOutputSpec[BooleanValue]:
        return MetricOutputSpec[BooleanValue](name=name, description=description, value_schema=BooleanValue)

    @staticmethod
    def model(
        name: str,
        value_schema: type[SchemaT],
        description: str | None = None,
    ) -> MetricOutputSpec[SchemaT]:
        return MetricOutputSpec[SchemaT](name=name, description=description, value_schema=value_schema)

    def coerce_value(self, value: object) -> SchemaT:
        """Validate and coerce a raw output value to this spec's declared schema."""
        return self.value_schema.model_validate(value)

    def coerce_output(self, output: MetricOutput) -> SchemaT:
        """Validate and coerce a named metric output against this spec."""
        if output.name != self.name:
            raise ValueError(f"Expected metric output {self.name!r}, got {output.name!r}")
        return self.coerce_value(output.value)

    def value_json_schema(self) -> dict[str, object]:
        return self.value_schema.model_json_schema()


class MetricDescriptor(BaseModel):
    """Metadata needed to materialize a decorated scorer as a Metric."""

    model_config = ConfigDict(extra="forbid")

    type: str
    outputs: list[MetricOutputSpec] = Field(min_length=1)
    config_schema: type[BaseModel] | None = None

    @field_validator("type")
    @classmethod
    def _type_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("metric type must not be empty")
        return value


    @field_validator("outputs")
    @classmethod
    def _output_names_must_be_unique(
        cls, value: list[MetricOutputSpec]
    ) -> list[MetricOutputSpec]:
        names = [output.name for output in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate metric output names: {duplicates}")
        return value


class MetricOutput(BaseModel):
    """One named value emitted by a metric."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: object


class MetricResult(BaseModel):
    """Structured row-level metric result."""

    model_config = ConfigDict(extra="forbid")

    outputs: list[MetricOutput]


@runtime_checkable
class Metric(Protocol):
    """Shared row-scoring primitive."""

    @property
    def type(self) -> str: ...

    def output_spec(self) -> list[MetricOutputSpec]: ...

    async def compute_scores(self, input: MetricInput) -> MetricResult: ...


ScorerReturn = Mapping[str, object] | Awaitable[Mapping[str, object]]
ScorerCallable = Callable[[ScorerInput[ConfigT]], ScorerReturn]
ScorerConfig = Mapping[str, object] | BaseModel


class MetricScorerFunction(Protocol[ConfigT]):
    """Decorated scorer function that can be materialized as a metric."""

    @property
    def descriptor(self) -> MetricDescriptor: ...

    def __call__(self, sample: ScorerInput[ConfigT]) -> ScorerReturn: ...

    def to_metric(self) -> ScorerFunctionMetric[ConfigT]: ...


class ScorerFunctionMetric(Generic[ConfigT]):
    """Metric adapter for decorator-authored OSS ScorerInput -> dict scorers."""

    def __init__(
        self,
        *,
        descriptor: MetricDescriptor,
        scorer_fn: ScorerCallable[ConfigT],
        config: ConfigT | None = None,
        sandbox: "Sandbox | None" = None,
        target: object | None = None,
        target_field: str = "target",
    ) -> None:
        self._descriptor = descriptor
        self._scorer_fn = scorer_fn
        self._config: ConfigT | Mapping[str, object] | None = (
            self._validate_config(config) if config is not None else None
        )
        self._sandbox = sandbox
        self._target = target
        self._target_field = target_field

    @property
    def type(self) -> str:
        return self._descriptor.type

    @property
    def config(self) -> dict[str, object]:
        config = self._resolve_config()
        if isinstance(config, BaseModel):
            return config.model_dump()
        return dict(config)

    @property
    def sandbox(self) -> "Sandbox | None":
        return self._sandbox

    @property
    def target(self) -> object | None:
        return self._target

    @property
    def target_field(self) -> str:
        return self._target_field

    def bind(
        self,
        *,
        config: ConfigT | None = None,
        sandbox: "Sandbox | None" = None,
        target: object | None = None,
        target_field: str | None = None,
    ) -> ScorerFunctionMetric[ConfigT]:
        validated_config = self._config if config is None else self._validate_config(config)
        return ScorerFunctionMetric(
            descriptor=self._descriptor,
            scorer_fn=self._scorer_fn,
            config=cast(ConfigT, validated_config),
            sandbox=self._sandbox if sandbox is None else sandbox,
            target=self._target if target is None else target,
            target_field=self._target_field if target_field is None else target_field,
        )

    def bind_raw_config(
        self,
        *,
        config: ScorerConfig | None = None,
        sandbox: "Sandbox | None" = None,
        target: object | None = None,
        target_field: str | None = None,
    ) -> ScorerFunctionMetric[ConfigT]:
        """Bind dict-like runtime config, validating it against ``config_schema`` when present."""
        validated_config = self._config if config is None else self._validate_config(config, coerce=True)
        return ScorerFunctionMetric(
            descriptor=self._descriptor,
            scorer_fn=self._scorer_fn,
            config=cast(ConfigT, validated_config),
            sandbox=self._sandbox if sandbox is None else sandbox,
            target=self._target if target is None else target,
            target_field=self._target_field if target_field is None else target_field,
        )

    def output_spec(self) -> list[MetricOutputSpec]:
        return self._descriptor.outputs

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        # Merge row-level dataset data with per-row candidate metadata so legacy
        # ``ScorerInput`` scorers see solver-produced payloads (logprobs,
        # trajectories, etc.) alongside dataset fields. Candidate metadata
        # wins on key collisions — it is the more specific source.
        merged_metadata: dict[str, object] = {**dict(input.row.data), **dict(input.candidate.metadata)}
        sample: ScorerInput[ConfigT] = ScorerInput(
            response=input.candidate.output_text or "",
            target=self._target if self._target is not None else input.row.data.get(self._target_field),
            metadata=merged_metadata,
            config=cast(ConfigT, self._resolve_config()),
            sandbox=self._sandbox,
        )
        result = self._scorer_fn(sample)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError(f"scorer_fn must return a mapping, got {type(result).__name__}")
        metric_result = MetricResult(
            outputs=[MetricOutput(name=name, value=value) for name, value in cast(Mapping[str, object], result).items()]
        )
        return validate_metric_result(metric_result, self._descriptor.outputs)

    def _validate_config(
        self, config: ConfigT | ScorerConfig, *, coerce: bool = False
    ) -> ConfigT | Mapping[str, object]:
        schema = self._descriptor.config_schema
        if schema is None:
            if isinstance(config, BaseModel):
                return config.model_dump()
            return dict(config)
        if isinstance(config, schema):
            return cast(ConfigT, config)
        if not coerce:
            raise TypeError(
                f"config must be an instance of {schema.__name__}; "
                "use bind_raw_config(...) to validate dict-like runtime config"
            )
        payload = config.model_dump() if isinstance(config, BaseModel) else config
        return cast(ConfigT, schema.model_validate(payload))

    def _resolve_config(self) -> ConfigT | Mapping[str, object]:
        if self._config is not None:
            return self._config
        schema = self._descriptor.config_schema
        if schema is None:
            return {}
        return cast(ConfigT, schema.model_validate({}))


def validate_metric_result(result: MetricResult, outputs: list[MetricOutputSpec]) -> MetricResult:
    """Validate a metric result against its declared outputs."""
    returned_names = [output.name for output in result.outputs]
    duplicates = sorted({name for name in returned_names if returned_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate metric output names: {duplicates}")

    outputs_by_name = {output.name: output for output in outputs}
    declared_names = [output.name for output in outputs]
    declared = set(declared_names)
    returned = set(returned_names)
    missing = [name for name in declared_names if name not in returned]
    undeclared = [name for name in returned_names if name not in declared]

    if missing:
        raise ValueError(f"Missing declared metric outputs: {missing}")
    if undeclared:
        raise ValueError(f"Undeclared metric outputs: {undeclared}")
    for output in result.outputs:
        outputs_by_name[output.name].coerce_output(output)
    return result


def score_names_from_output_spec(outputs: list[MetricOutputSpec]) -> list[str]:
    """Return declared numeric score names from metric output specs."""
    return [
        output.name
        for output in outputs
        if issubclass(output.value_schema, ContinuousScore | DiscreteScore | BooleanValue)
    ]


__all__ = [
    "BooleanValue",
    "CandidateOutput",
    "ContinuousScore",
    "DatasetRow",
    "DiscreteScore",
    "Label",
    "Metric",
    "MetricDescriptor",
    "MetricInput",
    "MetricOutput",
    "MetricOutputSpec",
    "MetricResult",
    "MetricScorerFunction",
    "ScorerCallable",
    "ScorerConfig",
    "ScorerFunctionMetric",
    "ScorerReturn",
    "score_names_from_output_spec",
    "validate_metric_result",
]
