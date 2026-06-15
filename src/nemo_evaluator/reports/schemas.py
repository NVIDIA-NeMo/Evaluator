# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for trajectories.jsonl and model_traffic.jsonl rows.

Required fields (no default) mark the minimum a writer must emit.
Optional fields (``= None``) are expected but may be absent in older logs
or when the relevant feature is disabled.

``fields_as_coverage_paths`` flattens a schema into the
``{label: path_tuple}`` dict consumed by ``_missing_fields`` in the report.
"""

import types
import typing
from typing import Any

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Agent step
# ---------------------------------------------------------------------------


class StepMetricsExtra(BaseModel):
    model_config = ConfigDict(extra="allow")

    latency_ms: float | None = None
    finish_reason: str | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class StepMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    extra: StepMetricsExtra | None = None


class AgentStep(BaseModel):
    """One agent step inside ``trajectory[0].steps``."""

    model_config = ConfigDict(extra="allow")

    step_id: str
    source: str
    timestamp: str | None = None
    model_name: str | None = None
    status_code: int | None = None
    message: str | None = None
    reasoning_content: str | None = None
    tool_calls: list | None = None
    metrics: StepMetrics | None = None


# ---------------------------------------------------------------------------
# Trajectory row (trajectories.jsonl)
# ---------------------------------------------------------------------------


class FinalMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_steps: int | None = None


class TrajectoryAgent(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    version: str | None = None
    model_name: str | None = None
    parent_agent: str | None = None


class TrajectoryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str | None = None
    session_id: str | None = None
    agent: TrajectoryAgent | None = None
    extra: dict | None = None
    steps: list | None = None
    final_metrics: FinalMetrics | None = None


class TrajectoryRow(BaseModel):
    """One row in ``trajectories.jsonl``."""

    model_config = ConfigDict(extra="allow")

    problem_idx: int
    repeat: int
    reward: float
    model: str | None = None
    trajectory: list[TrajectoryEntry] | None = None


# ---------------------------------------------------------------------------
# Model-traffic row (model_traffic.jsonl)
# ---------------------------------------------------------------------------


class ModelTrafficUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


class ModelTrafficRow(BaseModel):
    """One row in ``model_traffic.jsonl``."""

    model_config = ConfigDict(extra="allow")

    model_traffic_request_id: str
    timestamp: str
    benchmark: str
    problem_idx: int
    repeat: int
    model: str = ""
    finish_reason: str = ""
    session_id: str | None = None
    adapter_request_id: str | None = None
    service: str | None = None
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    latency_ms: float | None = None
    usage: ModelTrafficUsage | None = None
    token_provenance: str | None = None
    error_type: str | None = None
    request_hash: str | None = None
    tool_calls_full: list | None = None
    reasoning_content: str | None = None
    message_content: str | None = None


# ---------------------------------------------------------------------------
# Coverage-path helper
# ---------------------------------------------------------------------------


def _inner_model(annotation: Any) -> type[BaseModel] | None:
    """BaseModel class if annotation is ``BaseModel`` or ``BaseModel | None``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _list_item_model(annotation: Any) -> type[BaseModel] | None:
    """BaseModel class if annotation is ``list[BaseModel]`` or ``list[BaseModel] | None``."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is list and args:
        return _inner_model(args[0])
    # Optional[list[BaseModel]] — one level of Union unwrap
    is_union = origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(annotation, types.UnionType)
    )
    if is_union:
        for arg in args:
            result = _list_item_model(arg)
            if result is not None:
                return result
    return None


def fields_as_coverage_paths(
    model_cls: type[BaseModel],
    *,
    _prefix: tuple = (),
    _label: str = "",
) -> dict[str, tuple]:
    """Flatten *model_cls* fields into ``{label: path_tuple}`` for coverage checks.

    Rules:
    - ``list[BaseModel]`` fields: add the list itself, then recurse into index 0.
    - ``BaseModel`` fields: recurse (the parent key is not added as a leaf).
    - Everything else: add as a leaf.
    """
    result: dict[str, tuple] = {}
    for name, _info in model_cls.model_fields.items():
        path = _prefix + (name,)
        label = f"{_label}.{name}" if _label else name
        annotation = _info.annotation

        list_model = _list_item_model(annotation)
        inner = _inner_model(annotation) if list_model is None else None

        if list_model is not None:
            result[label] = path
            result.update(
                fields_as_coverage_paths(list_model, _prefix=path + (0,), _label=f"{label}[0]")
            )
        elif inner is not None:
            result.update(fields_as_coverage_paths(inner, _prefix=path, _label=label))
        else:
            result[label] = path
    return result
