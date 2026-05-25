# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
            {**_wire(0, 0, prompt=10, completion=5), "request_hash": "hA"},
            {**_wire(1, 0, prompt=20, completion=10, finish="tool_calls"), "request_hash": "hB"},
            {**_wire(1, 0, prompt=30, completion=8), "request_hash": "hC"},
        ],
    )
    return tmp_path


def test_audit_writes_report(bundle: Path) -> None:
    out = generate_trajectories_report(bundle)
    assert out == bundle / "trajectories_report.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "trajectories_report-v0.1"
    assert len(payload["benchmarks"]) == 1


def test_counts_and_score(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    assert report["counts"] == {"trials": 2, "problems": 2, "repeats": 1}
    assert report["score"]["mean_reward"] == 0.75


def test_tokens_section(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    t = report["tokens"]
    # 2 trials × (pt+ct): trial0=15, trial1=20+10+30+8=68 → 83 across all
    assert t["per_step_sum"] == 83
    assert t["wire_total"] == 83
    assert t["final_metrics_total"] == 83
    assert t["per_step_matches_wire"] is True
    assert t["final_metrics_matches_wire"] is True
    assert t["trials_with_per_step_vs_final_metrics_mismatch"] == 0


def test_wire_calls_section(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    wc = report["wire_calls"]
    assert wc["total"] == 3
    assert wc["unique"] == 3
    assert wc["duplicates_total"] == 0
    assert wc["trials_with_duplicates"] == 0
    assert wc["trials_with_more_wire_than_steps"] == 0
    assert wc["trials_with_fewer_wire_than_steps"] == 0
    assert wc["finish_reasons"] == {"stop": 2, "tool_calls": 1}


def test_field_coverage_per_trial_and_step(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    fc = report["field_coverage"]
    assert fc["agent_step_count"] == 3
    # Per-trial required + ATIF trajectory fields
    pt = fc["per_trial"]
    assert pt["problem_idx"] == "2/2"
    assert pt["reward"] == "2/2"
    assert pt["trajectory[0].schema_version"] == "2/2"
    assert pt["trajectory[0].agent.name"] == "2/2"
    # Per-step ATIF coverage
    ps = fc["per_agent_step"]
    assert ps["step_id"] == "3/3"
    assert ps["metrics.prompt_tokens"] == "3/3"
    assert ps["metrics.extra.latency_ms"] == "0/3"  # not set in fixture
    assert ps["metrics.extra.finish_reason"] == "0/3"


def test_score_mean_reward_correct(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    (bench / "eval-test.json").write_text(json.dumps({"benchmark": {"scores": {"summary": {"mean": 1.0}}}}))
    score = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["score"]
    assert score["mean_reward"] == 1.0
    assert score["reported_mean"] == 1.0
    assert score["is_mean_reward_correct"] is True


def test_no_benches_returns_none(tmp_path: Path) -> None:
    assert generate_trajectories_report(tmp_path) is None


def test_wire_duplicates_via_request_hash(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    # Two records with same request_hash → exact dedup catches it
    dup = {**_wire(0, 0, prompt=10, completion=5), "request_hash": "abc"}
    _write_jsonl(bench / "model_traffic.jsonl", [dup, dup])
    wc = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["wire_calls"]
    assert wc["total"] == 2
    assert wc["unique"] == 1
    assert wc["duplicates_total"] == 1
    assert wc["trials_with_duplicates"] == 1


def test_wire_dedup_fallback_when_no_request_hash(tmp_path: Path) -> None:
    """Older rows without request_hash use the heuristic response-side key."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    rec = _wire(0, 0, prompt=10, completion=5)
    _write_jsonl(bench / "model_traffic.jsonl", [rec, rec])
    wc = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["wire_calls"]
    assert wc["duplicates_total"] == 1


def test_step_wire_mismatch_after_dedup(tmp_path: Path) -> None:
    """1 step + 2 identical wires → dedup resolves the mismatch."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    rec = {**_wire(0, 0, prompt=10, completion=5), "request_hash": "h0"}
    _write_jsonl(bench / "model_traffic.jsonl", [rec, rec])
    wc = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["wire_calls"]
    # raw counts disagree (1 step vs 2 wires)
    assert wc["trials_with_more_wire_than_steps"] == 1
    # after dedup resolves
    assert wc["trials_with_step_wire_mismatch_after_dedup"] == 0


def test_tokens_per_step_vs_wire_mismatch(tmp_path: Path) -> None:
    """per_step_sum=0 but wire_total>0 → bug surface that enrichment fixes."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=0, ct=0)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=42, completion=7)])
    t = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["tokens"]
    assert t["per_step_sum"] == 0
    assert t["wire_total"] == 49
    assert t["per_step_matches_wire"] is False


def test_enrich_writes_enriched_jsonl_and_reports_changes(bundle: Path) -> None:
    out = generate_trajectories_report(bundle, enrich=True)
    enriched = bundle / "pinchbench" / "trajectories_enriched.jsonl"
    assert enriched.is_file()
    report = json.loads(out.read_text())["benchmarks"][0]
    enrichment = report["enrichment"]
    assert enrichment["trials_spliced_1_to_1"] == 2
    assert enrichment["trials_with_unmatched_calls_stashed"] == 0
    # latency / finish_reason weren't on steps before; backfilled now
    assert enrichment["steps_backfilled"]["latency_ms"] >= 1
    assert enrichment["steps_backfilled"]["finish_reason"] >= 1
    # The after-state coverage is right there in the report
    after = enrichment["step_field_coverage_after_enrichment"]
    assert after["metrics.extra.latency_ms"] == "3/3"
    assert after["metrics.extra.finish_reason"] == "3/3"


def test_enrich_all_or_nothing_per_trial(tmp_path: Path) -> None:
    """When wire-call count != step count, stash everything; never partial-splice."""
    bench = tmp_path / "pb"
    # 2 steps, 1 wire call → stash
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=0.0, steps=[_agent_step(0, msg="a", pt=0, ct=0), _agent_step(1, msg="b", pt=0, ct=0)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=10, completion=5)])
    report = json.loads(generate_trajectories_report(tmp_path, enrich=True).read_text())["benchmarks"][0]
    enr = report["enrichment"]
    assert enr["trials_spliced_1_to_1"] == 0
    assert enr["trials_with_unmatched_calls_stashed"] == 1
    # No step-level backfill (partial splice would misattribute)
    assert enr["steps_backfilled"]["prompt_tokens"] == 0
    # The wire call landed in extra.captured_model_calls
    enriched = json.loads((bench / "trajectories_enriched.jsonl").read_text().splitlines()[0])
    stash = enriched["trajectory"][0].get("extra", {}).get("captured_model_calls")
    assert isinstance(stash, list) and len(stash) == 1


def test_enrich_backfills_zero_token_steps(tmp_path: Path) -> None:
    """1 step with zero tokens + 1 wire call → splice fills them in."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=0, ct=0)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=42, completion=7)])
    generate_trajectories_report(tmp_path, enrich=True)
    enriched_step = json.loads((bench / "trajectories_enriched.jsonl").read_text().splitlines()[0])["trajectory"][0][
        "steps"
    ][0]
    assert enriched_step["metrics"]["prompt_tokens"] == 42
    assert enriched_step["metrics"]["completion_tokens"] == 7


def test_enrichment_section_absent_when_enrich_false(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle, enrich=False).read_text())["benchmarks"][0]
    assert "enrichment" not in report


def test_non_uniform_repeats_returns_histogram(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    # problem 0: 2 repeats; problem 1: 1 repeat → non-uniform
    _write_jsonl(
        bench / "trajectories.jsonl",
        [
            _trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="a", pt=5, ct=3)]),
            _trial(0, 1, reward=1.0, steps=[_agent_step(0, msg="b", pt=5, ct=3)]),
            _trial(1, 0, reward=0.0, steps=[_agent_step(0, msg="c", pt=5, ct=3)]),
        ],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [])
    repeats = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["counts"]["repeats"]
    # Histogram (json keys are strings): 1 problem with 2 trials, 1 problem with 1 trial
    assert repeats == {"2": 1, "1": 1}
