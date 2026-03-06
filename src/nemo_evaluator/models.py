"""Pydantic models for config validation and serialized output.

Runtime types (ModelResponse, StepRecord, RuntimeStats, FailureReport) live
in observability.types as lightweight dataclasses. This module handles:
- Input config validation (EvalConfig, RetryConfig, ShardConfig)
- Serialized bundle schema (EvalBundle, BenchmarkResult)
- Regression comparison (RegressionReport, RegressionDelta)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ScoreEntry(BaseModel):
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class CategoryScore(BaseModel):
    category: str
    n_samples: int
    mean_reward: float


class BenchmarkResult(BaseModel):
    name: str
    samples: int
    repeats: int = 1
    scores: dict[str, ScoreEntry] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    failures: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    categories: dict[str, CategoryScore] | None = None


class EvalConfig(BaseModel):
    benchmark: str = ""
    model: str = ""
    base_url: str = ""
    repeats: int = 1
    max_problems: int | None = None
    adapter: str | None = None
    harness: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float | None = None
    seed: int | None = None
    shard: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalBundle(BaseModel):
    """Central data structure produced by every evaluation run."""

    run_id: str
    config_hash: str
    sdk_version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: EvalConfig
    benchmark: BenchmarkResult
    n_results: int = 0

    @field_validator("run_id")
    @classmethod
    def run_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("run_id cannot be empty")
        return v


class ShardConfig(BaseModel):
    shard_idx: int
    total_shards: int
    problem_range: tuple[int, int]

    @field_validator("total_shards")
    @classmethod
    def positive_shards(cls, v: int) -> int:
        if v < 1:
            raise ValueError("total_shards must be >= 1")
        return v

    @field_validator("shard_idx")
    @classmethod
    def valid_idx(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError("shard_idx must be >= 0")
        return v


class RegressionDelta(BaseModel):
    metric: str
    baseline: float
    candidate: float
    delta: float
    pct_change: float | None = None
    ci_overlap: bool | None = None


class RegressionReport(BaseModel):
    baseline_id: str
    candidate_id: str
    score_deltas: list[RegressionDelta] = Field(default_factory=list)
    runtime_deltas: dict[str, Any] = Field(default_factory=dict)
    category_deltas: list[dict[str, Any]] = Field(default_factory=list)
    has_regression: bool = False
    threshold: float = 0.05


class RetryConfig(BaseModel):
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    jitter: bool = True
