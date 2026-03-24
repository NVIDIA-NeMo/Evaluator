"""Output configuration schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "./eval_results"
    timestamped: bool = True
    progress_interval: float = 60.0
    report: list[str] = Field(default_factory=lambda: ["markdown"])
    export: list[str] = Field(default_factory=list)
    export_config: dict[str, Any] = Field(default_factory=dict)
