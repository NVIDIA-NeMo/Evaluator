# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_evaluator.rasb_26h1.output import parse_output


def test_parse_output_summary(tmp_path):
    summary = {
        "run_name": "smoke",
        "environments": [
            {"env_id": "env-a", "pass_rate": 1.0, "passed": 2, "total": 2},
            {"env_id": "env-b", "pass_rate": 0.5, "passed": 1, "total": 2},
        ],
        "aggregate": {
            "total_samples": 4,
            "total_passed": 3,
            "overall_pass_rate": 0.75,
            "mean_pass_rate": 0.75,
            "std_pass_rate": 0.25,
            "median_pass_rate": 0.75,
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = parse_output(str(tmp_path))

    task = result.tasks["rasb_26h1"]
    assert (
        task.metrics["pass_rate"].scores["overall_pass_rate"].value
        == pytest.approx(0.75)
    )
    assert task.metrics["counts"].scores["total_samples"].value == 4
    assert task.metrics["counts"].scores["total_passed"].value == 3
    assert task.metrics["counts"].scores["total_failed"].value == 1
    assert result.groups["rasb"].metrics == task.metrics


def test_parse_output_missing_summary(tmp_path):
    with pytest.raises(FileNotFoundError, match="RASB summary not found"):
        parse_output(str(tmp_path))
