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
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    content: str
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    reasoning_tokens: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    request_prompt: str | None = None
    request_system: str | None = None

    @property
    def request_hash(self) -> str:
        """SHA256 of the request (prompt + system + model), not the response."""
        parts = [
            self.request_prompt or "",
            self.request_system or "",
            self.model,
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    @property
    def response_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class StepRecord:
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    problem_idx: int = 0
    repeat: int = 0
    timestamp: float = field(default_factory=time.time)

    prompt: str = ""
    system_prompt: str | None = None
    expected_answer: str = ""
    seed_metadata: dict[str, Any] = field(default_factory=dict)

    model_response: ModelResponse | None = None
    model_error: str | None = None
    retries: int = 0

    reward: float = 0.0
    extracted_answer: str | None = None
    scoring_method: str = ""
    scoring_details: dict[str, Any] = field(default_factory=dict)

    seed_ms: float = 0.0
    model_ms: float = 0.0
    verify_ms: float = 0.0
    total_ms: float = 0.0

    failure_category: str | None = None
    trajectory: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_id": self.step_id,
            "problem_idx": self.problem_idx,
            "repeat": self.repeat,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "reward": self.reward,
            "extracted_answer": self.extracted_answer,
            "scoring_method": self.scoring_method,
            "scoring_details": self.scoring_details,
            "seed_metadata": self.seed_metadata,
            "timing": {
                "seed_ms": round(self.seed_ms, 2),
                "model_ms": round(self.model_ms, 2),
                "verify_ms": round(self.verify_ms, 2),
                "total_ms": round(self.total_ms, 2),
            },
            "failure_category": self.failure_category,
            "retries": self.retries,
        }
        if self.model_response:
            d["model"] = {
                "content": self.model_response.content,
                "model": self.model_response.model,
                "finish_reason": self.model_response.finish_reason,
                "tokens": {
                    "prompt": self.model_response.prompt_tokens,
                    "completion": self.model_response.completion_tokens,
                    "total": self.model_response.total_tokens,
                    "reasoning": self.model_response.reasoning_tokens,
                },
                "latency_ms": self.model_response.latency_ms,
                "request_hash": self.model_response.request_hash,
                "response_hash": self.model_response.response_hash,
            }
        if self.model_error:
            d["model_error"] = self.model_error
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.trajectory:
            d["trajectory"] = self.trajectory
        return d


@dataclass
class CostEstimate:
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_cost": round(self.prompt_cost, 6),
            "completion_cost": round(self.completion_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "currency": self.currency,
        }


@dataclass
class RuntimeStats:
    total_steps: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    elapsed_seconds: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    tokens_per_second: float = 0.0
    steps_per_second: float = 0.0
    model_errors: int = 0
    total_retries: int = 0
    finish_reason_counts: dict[str, int] = field(default_factory=dict)
    cost: CostEstimate = field(default_factory=CostEstimate)
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "latency_percentiles_ms": {
                "p50": round(self.latency_p50_ms, 2),
                "p90": round(self.latency_p90_ms, 2),
                "p99": round(self.latency_p99_ms, 2),
            },
            "tokens_per_second": round(self.tokens_per_second, 2),
            "steps_per_second": round(self.steps_per_second, 2),
            "model_errors": self.model_errors,
            "total_retries": self.total_retries,
            "finish_reason_counts": self.finish_reason_counts,
        }
        if self.cost.total_cost > 0:
            d["cost"] = self.cost.to_dict()
        if self.cache_hits > 0 or self.cache_misses > 0:
            d["cache"] = {"hits": self.cache_hits, "misses": self.cache_misses}
        return d


@dataclass
class FailureRecord:
    category: str
    count: int = 0
    percentage: float = 0.0
    exemplars: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FailureReport:
    total_steps: int = 0
    total_failures: int = 0
    categories: dict[str, FailureRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_failures": self.total_failures,
            "failure_rate": round(self.total_failures / self.total_steps, 4) if self.total_steps else 0,
            "categories": {
                k: {
                    "count": v.count,
                    "percentage": round(v.percentage, 4),
                    "exemplars": v.exemplars[:3],
                }
                for k, v in self.categories.items()
            },
        }


@dataclass
class RunArtifacts:
    steps: list[StepRecord] = field(default_factory=list)
    runtime: RuntimeStats = field(default_factory=RuntimeStats)
    failures: FailureReport = field(default_factory=FailureReport)
