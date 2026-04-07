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

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from nemo_evaluator.observability.types import (
    FailureRecord,
    FailureReport,
    RunArtifacts,
    RuntimeStats,
    StepRecord,
)

DEFAULT_REFUSAL_PATTERNS = [
    r"I cannot",
    r"I'm unable",
    r"I can't",
    r"as an AI",
    r"I don't have the ability",
    r"I apologize",
]


class ArtifactCollector:
    def __init__(self, refusal_patterns: list[str] | None = None) -> None:
        self.steps: list[StepRecord] = []
        patterns = refusal_patterns or DEFAULT_REFUSAL_PATTERNS
        self._refusal_re = re.compile("|".join(patterns), re.IGNORECASE)

    def record(self, step: StepRecord) -> None:
        self._classify_failure(step)
        self.steps.append(step)

    def build(self, elapsed: float) -> RunArtifacts:
        arts = RunArtifacts(steps=list(self.steps))
        arts.runtime = self._compute_runtime(elapsed)
        arts.failures = self._compute_failures()
        return arts

    def _classify_failure(self, step: StepRecord) -> None:
        if step.model_error:
            err = step.model_error
            if any(code in err for code in ("408", "timeout", "Timeout", "timed out")):
                step.failure_category = "timeout"
            elif any(code in err for code in ("429", "Too Many Requests")):
                step.failure_category = "rate_limit"
            elif any(code in err for code in ("500", "502", "503", "504")):
                step.failure_category = "server_error"
            else:
                step.failure_category = "model_error"
        elif step.model_response:
            content = step.model_response.content
            if not content.strip():
                step.failure_category = "empty_response"
            elif self._refusal_re.search(content):
                step.failure_category = "refusal"
            elif step.reward == 0 and step.extracted_answer is None:
                step.failure_category = "format_error"

    def _compute_runtime(self, elapsed: float) -> RuntimeStats:
        latencies = [s.model_ms for s in self.steps if s.model_response]
        finish_reasons = Counter(
            s.model_response.finish_reason for s in self.steps if s.model_response and s.model_response.finish_reason
        )

        stats = RuntimeStats(
            total_steps=len(self.steps),
            elapsed_seconds=elapsed,
            model_errors=sum(1 for s in self.steps if s.model_error),
            total_retries=sum(s.retries for s in self.steps),
            finish_reason_counts=dict(finish_reasons),
        )

        for s in self.steps:
            if s.model_response:
                stats.total_tokens += s.model_response.total_tokens
                stats.total_prompt_tokens += s.model_response.prompt_tokens
                stats.total_completion_tokens += s.model_response.completion_tokens
                stats.total_reasoning_tokens += s.model_response.reasoning_tokens

        if latencies:
            arr = np.array(latencies)
            stats.latency_p50_ms = float(np.percentile(arr, 50))
            stats.latency_p90_ms = float(np.percentile(arr, 90))
            stats.latency_p99_ms = float(np.percentile(arr, 99))

        if elapsed > 0:
            stats.tokens_per_second = stats.total_tokens / elapsed
            stats.steps_per_second = stats.total_steps / elapsed

        return stats

    def _compute_failures(self) -> FailureReport:
        report = FailureReport(total_steps=len(self.steps))
        cats: dict[str, list[StepRecord]] = {}
        for s in self.steps:
            if s.failure_category:
                cats.setdefault(s.failure_category, []).append(s)
                report.total_failures += 1

        for cat, items in cats.items():
            report.categories[cat] = FailureRecord(
                category=cat,
                count=len(items),
                percentage=len(items) / len(self.steps) if self.steps else 0,
                exemplars=[
                    {"problem_idx": s.problem_idx, "repeat": s.repeat, "error": s.model_error or s.failure_category}
                    for s in items[:5]
                ],
            )
        return report

    @staticmethod
    def write(artifacts: RunArtifacts, output_dir: str | Path, run_id: str) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        traj_path = out / "trajectories.jsonl"
        with traj_path.open("w", encoding="utf-8") as f:
            for s in artifacts.steps:
                f.write(json.dumps(s.to_dict(), default=str) + "\n")
        paths["trajectories"] = traj_path

        stats_path = out / "runtime_stats.json"
        stats_path.write_text(json.dumps(artifacts.runtime.to_dict(), indent=2), encoding="utf-8")
        paths["runtime_stats"] = stats_path

        if artifacts.failures.total_failures > 0:
            fail_path = out / "failure_analysis.json"
            fail_path.write_text(json.dumps(artifacts.failures.to_dict(), indent=2), encoding="utf-8")
            paths["failure_analysis"] = fail_path

        return paths
