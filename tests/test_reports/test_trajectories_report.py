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
"""Audit report tests — pure file-to-file behavior over fixture jsonls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_evaluator.reports.trajectories import generate_trajectories_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _trial(problem_idx: int, repeat: int, *, reward: float, steps: list[dict]) -> dict:
    return {
        "problem_idx": problem_idx,
        "repeat": repeat,
        "reward": reward,
        "trajectory": [
            {
                "schema_version": "ATIF-v1.6",
                "session_id": f"sess-{problem_idx}-{repeat}",
                "agent": {"name": "test-agent", "version": "0.1", "model_name": "test-model"},
                "steps": steps,
                "final_metrics": {
                    "total_prompt_tokens": sum((s.get("metrics") or {}).get("prompt_tokens", 0) for s in steps),
                    "total_completion_tokens": sum(
                        (s.get("metrics") or {}).get("completion_tokens", 0) for s in steps
                    ),
                    "total_steps": len(steps),
                },
            }
        ],
        "model": {"tokens": {"total": sum((s.get("metrics") or {}).get("total_tokens", 0) for s in steps)}},
    }


def _agent_step(step_id: int, *, msg: str, pt: int, ct: int, tool_calls: list | None = None) -> dict:
    s: dict = {
        "step_id": step_id,
        "source": "agent",
        "message": msg,
        "metrics": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }
    if tool_calls is not None:
        s["tool_calls"] = tool_calls
    return s


def _wire(problem_idx: int, repeat: int, *, prompt: int, completion: int, finish: str = "stop") -> dict:
    return {
        "problem_idx": problem_idx,
        "repeat": repeat,
        "session_id": f"sess-{problem_idx}-{repeat}",
        "finish_reason": finish,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
        "latency_ms": 100.0,
    }


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    bench = tmp_path / "pinchbench"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [
            _trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="hi", pt=10, ct=5)]),
            _trial(1, 0, reward=0.5, steps=[
                _agent_step(0, msg="thinking", pt=20, ct=10, tool_calls=[{"name": "search"}]),
                _agent_step(1, msg="done", pt=30, ct=8),
            ]),
        ],
    )
    _write_jsonl(
        bench / "model_traffic.jsonl",
        [
            _wire(0, 0, prompt=10, completion=5),
            _wire(1, 0, prompt=20, completion=10, finish="tool_calls"),
            _wire(1, 0, prompt=30, completion=8),
        ],
    )
    return tmp_path


def test_audit_writes_report(bundle: Path) -> None:
    out = generate_trajectories_report(bundle)
    assert out == bundle / "trajectories_report.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "trajectories_report-v0.1"
    assert len(payload["benchmarks"]) == 1


def _v(node: dict) -> object:
    """Convenience: unwrap {value, from} to its value."""
    return node["value"] if isinstance(node, dict) and "from" in node else node


def test_counts_and_score(bundle: Path) -> None:
    out = generate_trajectories_report(bundle)
    report = json.loads(out.read_text())["benchmarks"][0]
    assert _v(report["counts"]["n_trials"]) == 2
    assert _v(report["counts"]["n_problems"]) == 2
    assert _v(report["counts"]["repeats"]) == 1
    # Mean of [1.0, 0.5] == 0.75
    assert _v(report["score"]["mean_reward"]) == 0.75
    # every metric must carry its calculation explanation
    assert "from" in report["counts"]["n_trials"]
    assert "from" in report["score"]["mean_reward"]


def test_trajectory_native_stats(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    nt = report["trajectory_native"]
    assert _v(nt["agent_steps_total"]) == 3
    assert _v(nt["tool_calls_total"]) == 1
    fp = _v(nt["fields_present_on_steps"])
    assert fp["step_id"] == "3/3"
    assert fp["message"] == "3/3"
    assert fp["tool_calls"] == "1/3"


def test_wire_captures_match_steps(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    wc = report["wire_captures"]
    assert _v(wc["total_observed"]) == 3
    assert _v(wc["finish_reasons"]) == {"stop": 2, "tool_calls": 1}
    delta = _v(report["mismatches"]["agent_steps_vs_wire_calls"])
    assert delta == {"captures>steps": 0, "captures<steps": 0, "equal": 2}


def test_token_reconciliation(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    tokens = _v(report["mismatches"]["tokens"])
    # 2 trials × (pt+ct): trial0=15, trial1=20+10+30+8=68 → 83 across all
    assert tokens["per_step_total"] == 83
    assert tokens["wire_total"] == 83
    assert tokens["ref_total"] == 83
    assert tokens["match_ref_vs_wire"] is True


def test_atif_per_trial_presence(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    pres = _v(report["atif_per_trial_presence"])
    assert pres["schema_version"] == "2/2"
    assert pres["session_id"] == "2/2"
    assert pres["agent.name"] == "2/2"
    assert pres["final_metrics.total_steps"] == "2/2"


def test_no_benches_returns_none(tmp_path: Path) -> None:
    assert generate_trajectories_report(tmp_path) is None


def test_missing_wire_captures_are_flagged(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="hi", pt=5, ct=3), _agent_step(1, msg="bye", pt=4, ct=2)])],
    )
    # only 1 wire call vs 2 agent steps
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=5, completion=3)])
    report = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]
    assert _v(report["mismatches"]["agent_steps_vs_wire_calls"]) == {
        "captures>steps": 0,
        "captures<steps": 1,
        "equal": 0,
    }


def test_enrich_writes_enriched_jsonl_and_reports_changes(bundle: Path) -> None:
    out = generate_trajectories_report(bundle, enrich=True)
    assert out is not None
    enriched = bundle / "pinchbench" / "trajectories_enriched.jsonl"
    assert enriched.is_file()
    report = json.loads(out.read_text())["benchmarks"][0]
    enrichment = _v(report["enrichment"])
    # Bundle fixture has steps with non-zero pt/ct already, so no backfill
    # *for those*, but latency_ms/finish_reason aren't set on the steps and
    # are present on the wire rows.
    assert enrichment["rows_written"] == 2
    assert enrichment["steps_backfilled_latency_ms"] >= 1
    assert enrichment["steps_backfilled_finish_reason"] >= 1


def test_enrich_backfills_zero_token_steps(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # step with zero tokens, wire call has real tokens — should backfill
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=0, ct=0)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=42, completion=7)])
    out = generate_trajectories_report(tmp_path, enrich=True)
    enriched_rows = []
    with (bench / "trajectories_enriched.jsonl").open() as fh:
        for line in fh:
            enriched_rows.append(json.loads(line))
    step = enriched_rows[0]["trajectory"][0]["steps"][0]
    assert step["metrics"]["prompt_tokens"] == 42
    assert step["metrics"]["completion_tokens"] == 7
    counts = _v(json.loads(out.read_text())["benchmarks"][0]["enrichment"])
    assert counts["steps_backfilled_prompt_tokens"] == 1
    assert counts["steps_backfilled_completion_tokens"] == 1


def test_post_dedup_mismatch_and_per_step_total_consistency(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # 2 identical steps (1 dup), 2 identical wire rows (1 dup).
    # After dedup, sets are both size 1 -> no remaining mismatch.
    s = _agent_step(0, msg="same", pt=10, ct=5)
    _write_jsonl(bench / "trajectories.jsonl", [_trial(0, 0, reward=1.0, steps=[s, s])])
    w = _wire(0, 0, prompt=10, completion=5)
    _write_jsonl(bench / "model_traffic.jsonl", [w, w])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert _v(a["duplicate_steps_in_trial_total"]) == 1
    assert _v(a["duplicate_wire_calls_in_trial_total"]) == 1
    assert _v(a["mismatched_steps_vs_wire_trials"]) == 0
    assert _v(a["mismatched_steps_vs_wire_trials_after_dedup"]) == 0
    # 2 steps × 15 tokens = 30; final_metrics.total = 30 → consistent
    assert _v(a["trials_with_per_step_vs_final_metrics_token_mismatch"]) == 0


def test_anomalies_zero_token_steps(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # trial 0: 2 zero-token steps; trial 1: 1 healthy + 1 zero
    _write_jsonl(
        bench / "trajectories.jsonl",
        [
            _trial(0, 0, reward=0.0, steps=[
                _agent_step(0, msg="a", pt=0, ct=0),
                _agent_step(1, msg="b", pt=0, ct=0),
            ]),
            _trial(1, 0, reward=1.0, steps=[
                _agent_step(0, msg="c", pt=10, ct=5),
                _agent_step(1, msg="d", pt=0, ct=0),
            ]),
        ],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert _v(a["steps_with_zero_or_none_tokens"]) == 3
    assert _v(a["trials_with_any_zero_token_step"]) == 2
    assert _v(a["trials_with_all_zero_token_steps"]) == 1
    # Every anomaly metric carries its calculation note
    for key in a:
        assert "from" in a[key], f"{key} missing 'from' explanation"


def test_anomalies_duplicate_steps(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # trial 0: two steps with identical (message, reasoning, tool_calls) → 1 dup
    dup_step_a = _agent_step(0, msg="thinking…", pt=10, ct=5)
    dup_step_b = _agent_step(1, msg="thinking…", pt=10, ct=5)
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[dup_step_a, dup_step_b])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert _v(a["duplicate_steps_in_trial_total"]) == 1
    assert _v(a["trials_with_duplicate_steps"]) == 1


def test_anomalies_duplicate_wire_calls(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    # Two model_traffic rows with identical fingerprints for trial (0,0)
    dup = _wire(0, 0, prompt=5, completion=3)
    _write_jsonl(bench / "model_traffic.jsonl", [dup, dup])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert _v(a["duplicate_wire_calls_in_trial_total"]) == 1
    assert _v(a["trials_with_duplicate_wire_calls"]) == 1


def test_is_mean_reward_correct(bundle: Path, tmp_path: Path) -> None:
    # The bundle fixture doesn't ship an eval-*.json so reported_mean is None
    # and is_mean_reward_correct stays None. Synthesize a matching bundle.
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    eval_path = bench / "eval-test.json"
    eval_path.write_text(json.dumps({
        "benchmark": {"scores": {"summary": {"mean": 1.0}}},
    }), encoding="utf-8")
    score = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["score"]
    assert _v(score["mean_reward"]) == 1.0
    assert _v(score["reported_mean"]) == 1.0
    assert _v(score["is_mean_reward_correct"]) is True
