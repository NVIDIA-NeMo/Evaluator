"""Gate policy schema for multi-benchmark accuracy quality gating.

Defines per-benchmark thresholds, tiers, and metric selection for
automated release qualification.  The gate engine (``engine.gate``)
applies these thresholds to paired evaluation data and produces a
GO / NO-GO / INCONCLUSIVE aggregate verdict.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tier(str, Enum):
    critical = "critical"
    supporting = "supporting"
    advisory = "advisory"


class Direction(str, Enum):
    higher_is_better = "higher_is_better"
    lower_is_better = "lower_is_better"


class BenchmarkGateDefaults(BaseModel):
    """Default thresholds applied to every benchmark unless overridden."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier = Tier.supporting
    max_drop: float = 0.015
    max_relative_drop: float | None = None
    relative_guard_below: float | None = None
    metric: str | None = None
    direction: Direction = Direction.higher_is_better

    @field_validator("max_drop")
    @classmethod
    def _max_drop_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"max_drop must be >= 0, got {v}")
        return v

    @field_validator("max_relative_drop")
    @classmethod
    def _max_relative_drop_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"max_relative_drop must be >= 0, got {v}")
        return v

    @field_validator("relative_guard_below")
    @classmethod
    def _relative_guard_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError(f"relative_guard_below must be in [0, 1], got {v}")
        return v


class BenchmarkGateEntry(BaseModel):
    """Per-benchmark overrides.  ``None`` fields inherit from defaults."""

    model_config = ConfigDict(extra="forbid")

    tier: Tier | None = None
    max_drop: float | None = None
    max_relative_drop: float | None = Field(default=None)
    relative_guard_below: float | None = Field(default=None)
    metric: str | None = None
    direction: Direction | None = None

    @field_validator("max_drop")
    @classmethod
    def _max_drop_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"max_drop must be >= 0, got {v}")
        return v

    @field_validator("max_relative_drop")
    @classmethod
    def _max_relative_drop_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"max_relative_drop must be >= 0, got {v}")
        return v

    @field_validator("relative_guard_below")
    @classmethod
    def _relative_guard_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError(f"relative_guard_below must be in [0, 1], got {v}")
        return v


class ResolvedBenchmarkPolicy(BaseModel):
    """Fully resolved policy for a single benchmark (no unset fields)."""

    model_config = ConfigDict(frozen=True)

    tier: Tier
    max_drop: float
    max_relative_drop: float | None
    relative_guard_below: float | None
    metric: str | None
    direction: Direction


class GatePolicy(BaseModel):
    """Top-level gate policy loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    defaults: BenchmarkGateDefaults = Field(default_factory=BenchmarkGateDefaults)
    benchmarks: dict[str, BenchmarkGateEntry] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported gate policy version: {v}")
        return v

    def resolve(self, benchmark_name: str) -> ResolvedBenchmarkPolicy:
        """Merge per-benchmark overrides onto defaults."""
        d = self.defaults
        entry = self.benchmarks.get(benchmark_name)
        if entry is None:
            return ResolvedBenchmarkPolicy(
                tier=d.tier,
                max_drop=d.max_drop,
                max_relative_drop=d.max_relative_drop,
                relative_guard_below=d.relative_guard_below,
                metric=d.metric,
                direction=d.direction,
            )
        return ResolvedBenchmarkPolicy(
            tier=entry.tier if entry.tier is not None else d.tier,
            max_drop=entry.max_drop if entry.max_drop is not None else d.max_drop,
            max_relative_drop=(entry.max_relative_drop if entry.max_relative_drop is not None else d.max_relative_drop),
            relative_guard_below=(
                entry.relative_guard_below if entry.relative_guard_below is not None else d.relative_guard_below
            ),
            metric=entry.metric if entry.metric is not None else d.metric,
            direction=entry.direction if entry.direction is not None else d.direction,
        )

    def required_benchmarks(self) -> set[str]:
        """Benchmark names that must be present for the gate to pass.

        Advisory benchmarks are excluded — they are tracked but not required.
        """
        required: set[str] = set()
        for name, entry in self.benchmarks.items():
            tier = entry.tier if entry.tier is not None else self.defaults.tier
            if tier != Tier.advisory:
                required.add(name)
        return required

    def validate_for_gate(self, supported_metrics: set[str]) -> None:
        """Validate that required benchmarks can be gated deterministically."""
        errors: list[str] = []
        for name in sorted(self.required_benchmarks()):
            resolved = self.resolve(name)
            if not resolved.metric:
                errors.append(f"{name}: required benchmark must resolve to an explicit metric")
                continue
            if resolved.metric not in supported_metrics:
                supported = ", ".join(sorted(supported_metrics))
                errors.append(f"{name}: unsupported metric {resolved.metric!r}; supported metrics: {supported}")
        if errors:
            raise ValueError("Invalid gate policy:\n" + "\n".join(errors))


def load_gate_policy(path: str | Path) -> GatePolicy:
    """Load and validate a gate policy YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gate policy not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Gate policy must be a YAML mapping, got {type(raw).__name__}")
    return GatePolicy.model_validate(raw)


def default_policy() -> GatePolicy:
    """Return a policy with all defaults (no per-benchmark overrides)."""
    return GatePolicy()
