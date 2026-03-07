"""YAML config schema with Pydantic validation."""
from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


_ENV_RE = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")
        return _ENV_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class TaskConfig(BaseModel):

    benchmark: str | None = None
    adapter: str | None = None
    harness: Literal["lm-eval"] | None = None

    resource_server: str | None = None
    endpoint: str | None = None
    protocol: str | None = None

    eval: str | None = None
    tasks: list[str] | None = None
    fewshot: int | None = None
    examples: int | None = None

    repeats: int | None = None
    max_problems: int | None = None
    system_prompt: str | None = None
    scoring: str | None = None

    model_url: str | None = None
    model_id: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def at_least_one_target(self) -> "TaskConfig":
        has_target = any([
            self.benchmark, self.adapter, self.harness, self.resource_server,
        ])
        if not has_target:
            raise ValueError(
                "Task must specify at least one of: benchmark, adapter, harness, resource_server"
            )
        return self


class EvaluationConfig(BaseModel):

    model_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    cache_dir: str | None = None
    output_dir: str = "./eval_results"
    resume: bool = False
    tasks: list[TaskConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_tasks(self) -> "EvaluationConfig":
        if not self.tasks:
            raise ValueError("Config must contain at least one task")
        return self


class ConfigFile(BaseModel):
    evaluation: EvaluationConfig | None = None

    tasks: list[TaskConfig] | None = None
    model_url: str | None = None
    model_id: str | None = None

    def resolve(self) -> EvaluationConfig:
        """Return a unified EvaluationConfig regardless of nesting."""
        if self.evaluation:
            return self.evaluation
        return EvaluationConfig(
            model_url=self.model_url,
            model_id=self.model_id,
            tasks=self.tasks or [],
        )


def parse_config(raw: dict[str, Any]) -> EvaluationConfig:
    """Parse and validate a raw YAML dict, expanding ${VAR} env vars."""
    expanded = _expand_env(raw)
    root = ConfigFile.model_validate(expanded)
    return root.resolve()
