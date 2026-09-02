# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output parser for the RASB 26H1 NeMo Evaluator harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemo_evaluator.api.api_dataclasses import EvaluationResult

TASK_NAME = "rasb_26h1"


def _score(value: float | int, *, count: int | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if count is not None:
        stats["count"] = count
    return {"value": float(value), "stats": stats}


def _load_summary(output_dir: str) -> dict[str, Any]:
    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"RASB summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    total_samples = int(aggregate.get("total_samples", 0) or 0)
    total_passed = int(aggregate.get("total_passed", 0) or 0)
    total_failed = max(total_samples - total_passed, 0)
    environments = summary.get("environments", [])

    return {
        "pass_rate": {
            "scores": {
                "overall_pass_rate": _score(
                    aggregate.get("overall_pass_rate", 0.0),
                    count=total_samples,
                ),
                "mean_environment_pass_rate": _score(
                    aggregate.get("mean_pass_rate", 0.0),
                    count=len(environments),
                ),
                "median_environment_pass_rate": _score(
                    aggregate.get("median_pass_rate", 0.0),
                    count=len(environments),
                ),
                "std_environment_pass_rate": _score(
                    aggregate.get("std_pass_rate", 0.0),
                    count=len(environments),
                ),
            }
        },
        "counts": {
            "scores": {
                "total_samples": _score(total_samples),
                "total_passed": _score(total_passed),
                "total_failed": _score(total_failed),
                "total_environments": _score(len(environments)),
            }
        },
    }


def parse_output(output_dir: str) -> EvaluationResult:
    summary = _load_summary(output_dir)
    metrics = _metrics_from_summary(summary)
    return EvaluationResult(
        tasks={TASK_NAME: {"metrics": metrics}},
        groups={"rasb": {"metrics": metrics}},
    )
