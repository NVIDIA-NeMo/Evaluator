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
                    "total_completion_tokens": sum((s.get("metrics") or {}).get("completion_tokens", 0) for s in steps),
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
            _trial(
                1,
                0,
                reward=0.5,
                steps=[
                    _agent_step(0, msg="thinking", pt=20, ct=10, tool_calls=[{"name": "search"}]),
                    _agent_step(1, msg="done", pt=30, ct=8),
                ],
            ),
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


def test_enrich_all_or_nothing_per_trial(tmp_path: Path) -> None:
    """When wire-call count != step count, stash everything in extra and
    leave step metrics untouched (no partial splice that misattributes)."""
    bench = tmp_path / "pb"
    # Trial 0: 2 steps, 1 wire call -> stash (no splice)
    s1 = _agent_step(0, msg="a", pt=0, ct=0)
    s2 = _agent_step(1, msg="b", pt=0, ct=0)
    _write_jsonl(bench / "trajectories.jsonl", [_trial(0, 0, reward=0.0, steps=[s1, s2])])
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=10, completion=5)])

    out = generate_trajectories_report(tmp_path, enrich=True)
    counts = _v(json.loads(out.read_text())["benchmarks"][0]["enrichment"])
    assert counts["trials_spliced"] == 0
    assert counts["trials_stashed_unmatched"] == 1
    # Crucially, step metrics were NOT touched (no partial splice).
    assert counts["steps_backfilled_prompt_tokens"] == 0

    enriched = json.loads((bench / "trajectories_enriched.jsonl").read_text().splitlines()[0])
    steps = [s for s in enriched["trajectory"][0]["steps"] if s.get("source") == "agent"]
    # both steps still have zero/none tokens
    for s in steps:
        m = s.get("metrics") or {}
        assert not m.get("prompt_tokens")
        assert not m.get("completion_tokens")
    # All wire calls stashed in extra.captured_model_calls
    stash = enriched["trajectory"][0].get("extra", {}).get("captured_model_calls")
    assert isinstance(stash, list) and len(stash) == 1


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


def test_stashed_model_calls_audit(tmp_path: Path) -> None:
    """After enrichment with mismatch, the stash audit reflects what's in extra.captured_model_calls."""
    bench = tmp_path / "pb"
    # 1 step, 2 wire calls → mismatch → all 2 calls stashed
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=0, ct=0)])],
    )
    _write_jsonl(
        bench / "model_traffic.jsonl",
        [_wire(0, 0, prompt=10, completion=5), _wire(0, 0, prompt=12, completion=3)],
    )
    out = generate_trajectories_report(tmp_path, enrich=True)
    report = json.loads(out.read_text())["benchmarks"][0]
    # First call: builds report from ORIGINAL trajectories.jsonl (no stash yet),
    # so stash audit is 0/0 — that's still useful as a baseline.
    s = _v(report["stashed_model_calls"])
    assert s["trials_with_stashed_calls"] == 0
    # But a second pass on the same dir (now reading the enriched file as
    # input) is out of scope — the stash audit is over trajectories.jsonl
    # not trajectories_enriched.jsonl. Still verify the enriched file has
    # them where they should be:
    enriched = json.loads((bench / "trajectories_enriched.jsonl").read_text().splitlines()[0])
    stash = enriched["trajectory"][0].get("extra", {}).get("captured_model_calls")
    assert isinstance(stash, list) and len(stash) == 2


def test_required_row_fields_presence(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    rrf = _v(report["row_required_fields_presence"])
    assert rrf["problem_idx"] == "2/2"
    assert rrf["repeat"] == "2/2"
    assert rrf["reward"] == "2/2"
    assert rrf["trajectory"] == "2/2"
    assert rrf["model"] == "2/2"


def test_step_field_presence_extended(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    fp = _v(report["trajectory_native"]["fields_present_on_steps"])
    # The fixture sets metrics.{prompt,completion,total}_tokens on every step.
    assert fp["metrics.prompt_tokens"] == "3/3"
    assert fp["metrics.completion_tokens"] == "3/3"
    assert fp["metrics.total_tokens"] == "3/3"
    # Fixture has no timestamp/model_name/status_code/extra fields → all 0/3
    assert fp["timestamp"] == "0/3"
    assert fp["metrics.extra.latency_ms"] == "0/3"
    assert fp["metrics.extra.finish_reason"] == "0/3"


def test_post_wire_dedup_mismatch_and_per_step_total_consistency(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # 1 step, 2 identical wire rows (1 dup). After wire-dedup,
    # unique wire count is 1 -> matches step count.
    s = _agent_step(0, msg="x", pt=10, ct=5)
    _write_jsonl(bench / "trajectories.jsonl", [_trial(0, 0, reward=1.0, steps=[s])])
    w = {**_wire(0, 0, prompt=10, completion=5), "request_hash": "h0"}
    _write_jsonl(bench / "model_traffic.jsonl", [w, w])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert _v(a["duplicate_wire_calls_in_trial_total"]) == 1
    # raw mismatch: 1 step vs 2 wire rows
    assert _v(a["mismatched_steps_vs_wire_trials"]) == 1
    # after wire-dedup: 1 step vs 1 unique wire row -> resolved
    assert _v(a["mismatched_steps_vs_wire_trials_after_wire_dedup"]) == 0
    # 1 step × 15 tokens = 15; final_metrics.total = 15 → consistent
    assert _v(a["trials_with_per_step_vs_final_metrics_token_mismatch"]) == 0


def test_anomalies_zero_token_steps(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # trial 0: 2 zero-token steps; trial 1: 1 healthy + 1 zero
    _write_jsonl(
        bench / "trajectories.jsonl",
        [
            _trial(
                0,
                0,
                reward=0.0,
                steps=[
                    _agent_step(0, msg="a", pt=0, ct=0),
                    _agent_step(1, msg="b", pt=0, ct=0),
                ],
            ),
            _trial(
                1,
                0,
                reward=1.0,
                steps=[
                    _agent_step(0, msg="c", pt=10, ct=5),
                    _agent_step(1, msg="d", pt=0, ct=0),
                ],
            ),
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


def test_no_step_side_duplicate_tracking(tmp_path: Path) -> None:
    """Duplicates are only tracked for model_traffic.jsonl, not trajectory steps."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert "duplicate_steps_in_trial_total" not in a
    assert "trials_with_duplicate_steps" not in a


def test_stashed_section_only_when_enrich(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=5, completion=3)])

    audit = json.loads(generate_trajectories_report(tmp_path, enrich=False).read_text())["benchmarks"][0]
    assert "stashed_model_calls" not in audit

    enriched_run = json.loads(generate_trajectories_report(tmp_path, enrich=True).read_text())["benchmarks"][0]
    assert "stashed_model_calls" in enriched_run


def test_wire_dedup_prefers_request_hash(tmp_path: Path) -> None:
    """When records carry request_hash, the dedup key is exact (not heuristic)."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    # Two records with the SAME request_hash but different response stats —
    # heuristic would miss this; request_hash catches it as a duplicate.
    rec_a = {**_wire(0, 0, prompt=10, completion=5), "request_hash": "abc123abc123abc1"}
    rec_b = {**_wire(0, 0, prompt=999, completion=999), "request_hash": "abc123abc123abc1", "latency_ms": 42.0}
    _write_jsonl(bench / "model_traffic.jsonl", [rec_a, rec_b])
    a = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["anomalies"]
    assert _v(a["duplicate_wire_calls_in_trial_total"]) == 1
    assert "request_hash" in a["duplicate_wire_calls_in_trial_total"]["from"]


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
    eval_path.write_text(
        json.dumps(
            {
                "benchmark": {"scores": {"summary": {"mean": 1.0}}},
            }
        ),
        encoding="utf-8",
    )
    score = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["score"]
    assert _v(score["mean_reward"]) == 1.0
    assert _v(score["reported_mean"]) == 1.0
    assert _v(score["is_mean_reward_correct"]) is True
