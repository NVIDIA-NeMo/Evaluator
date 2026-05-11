# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prompt enrichment from inference_log.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from nemo_evaluator.reports.eval import (
    _enrich_results_with_prompts,
    load_bundles_for_export,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestEnrichResultsWithPrompts:
    def test_enriches_missing_prompts_from_inference_log(self, tmp_path):
        results = [
            {"problem_idx": 0, "repeat": 0, "reward": 1.0},
            {"problem_idx": 0, "repeat": 1, "reward": 0.0},
            {"problem_idx": 1, "repeat": 0, "reward": 0.5},
        ]
        inf_path = tmp_path / "inference_log.jsonl"
        _write_jsonl(
            inf_path,
            [
                {"problem_idx": 0, "repeat": 0, "prompt": "Task zero, run one"},
                {"problem_idx": 0, "repeat": 1, "prompt": "Task zero, run two"},
                {"problem_idx": 1, "repeat": 0, "prompt": "Task one"},
            ],
        )
        _enrich_results_with_prompts(results, inf_path)
        assert [r["prompt"] for r in results] == [
            "Task zero, run one",
            "Task zero, run two",
            "Task one",
        ]

    def test_preserves_existing_prompt(self, tmp_path):
        results = [{"problem_idx": 0, "repeat": 0, "prompt": "already here"}]
        inf_path = tmp_path / "inference_log.jsonl"
        _write_jsonl(inf_path, [{"problem_idx": 0, "repeat": 0, "prompt": "from log"}])
        _enrich_results_with_prompts(results, inf_path)
        assert results[0]["prompt"] == "already here"

    def test_noop_when_log_missing(self, tmp_path):
        results = [{"problem_idx": 0, "repeat": 0, "reward": 1.0}]
        _enrich_results_with_prompts(results, tmp_path / "does_not_exist.jsonl")
        assert "prompt" not in results[0]

    def test_noop_when_all_results_have_prompts(self, tmp_path):
        results = [{"problem_idx": 0, "repeat": 0, "prompt": "x"}]
        _enrich_results_with_prompts(results, tmp_path / "never_read.jsonl")
        assert results[0]["prompt"] == "x"

    def test_tolerates_malformed_log_lines(self, tmp_path):
        results = [
            {"problem_idx": 0, "repeat": 0},
            {"problem_idx": 1, "repeat": 0},
        ]
        inf_path = tmp_path / "inference_log.jsonl"
        inf_path.write_text(
            "{bad json\n"
            + json.dumps({"problem_idx": 0, "repeat": 0, "prompt": "ok0"})
            + "\n"
            + json.dumps({"problem_idx": 1, "repeat": 0, "prompt": ""})
            + "\n"
            + json.dumps({"problem_idx": 1, "repeat": 0, "prompt": "ok1"})
            + "\n",
            encoding="utf-8",
        )
        _enrich_results_with_prompts(results, inf_path)
        assert results[0]["prompt"] == "ok0"
        assert results[1]["prompt"] == "ok1"

    def test_handles_repeat_defaulting_to_zero(self, tmp_path):
        results = [{"problem_idx": 0}]
        inf_path = tmp_path / "inference_log.jsonl"
        _write_jsonl(inf_path, [{"problem_idx": 0, "prompt": "implicit r=0"}])
        _enrich_results_with_prompts(results, inf_path)
        assert results[0]["prompt"] == "implicit r=0"


class TestLoadBundlesForExportEndToEnd:
    def test_bundle_results_enriched_from_sibling_log(self, tmp_path):
        bench_dir = tmp_path / "shard_0" / "pinchbench" / "pinchbench"
        bench_dir.mkdir(parents=True)
        eval_json = bench_dir / "eval-run.json"
        eval_json.write_text(
            json.dumps({"benchmark": {"name": "pinchbench", "samples": 2}}),
            encoding="utf-8",
        )
        _write_jsonl(
            bench_dir / "results.jsonl",
            [
                {"problem_idx": 0, "repeat": 0, "reward": 1.0, "model_response": "ok"},
                {"problem_idx": 1, "repeat": 0, "reward": 0.0, "model_response": "bad"},
            ],
        )
        _write_jsonl(
            bench_dir / "inference_log.jsonl",
            [
                {"problem_idx": 0, "repeat": 0, "prompt": "PROMPT ZERO"},
                {"problem_idx": 1, "repeat": 0, "prompt": "PROMPT ONE"},
            ],
        )

        [bundle] = load_bundles_for_export([eval_json])
        prompts = [r["prompt"] for r in bundle["_results"]]
        assert prompts == ["PROMPT ZERO", "PROMPT ONE"]
