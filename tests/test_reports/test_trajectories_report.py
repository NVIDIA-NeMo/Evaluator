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


def _wire(
    problem_idx: int,
    repeat: int,
    *,
    prompt: int,
    completion: int,
    finish: str = "stop",
    status_code: int = 200,
) -> dict:
    return {
        "problem_idx": problem_idx,
        "repeat": repeat,
        "session_id": f"sess-{problem_idx}-{repeat}",
        "status_code": status_code,
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
    assert t["trials_with_per_step_vs_final_metrics_mismatch"] == 0


def test_wire_calls_section(bundle: Path) -> None:
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    wc = report["wire_calls"]
    assert wc["total"] == 3
    assert wc["unique"] == 3
    assert wc["trials_with_duplicates"] == 0
    assert wc["trials_with_more_wire_than_steps"] == 0
    assert wc["trials_with_fewer_wire_than_steps"] == 0
    assert wc["trials_with_no_agent_steps"] == "0/2 (0.0%)"
    assert wc["trials_with_no_wire_calls"] == "0/2 (0.0%)"
    assert wc["trials_silent_either_way"] == "0/2 (0.0%)"


_OK_STEP = _agent_step(0, msg="x", pt=5, ct=3)
_OK_WIRE = _wire(0, 0, prompt=5, completion=3)
_FAILED_WIRE = _wire(0, 0, prompt=5, completion=3, status_code=500)
_ENRICHMENT_FIELD = {
    "spliced": "trials_spliced_1_to_1",
    "stashed": "trials_with_unmatched_calls_stashed",
    "no_wire": "trials_with_no_wire_data",
}


@pytest.mark.parametrize(
    "steps, wire_rows, no_steps, no_wire, enrich_kind",
    [
        ([_OK_STEP], [_OK_WIRE], False, False, "spliced"),
        ([], [_OK_WIRE], True, False, "stashed"),
        ([_OK_STEP], [], False, True, "no_wire"),
        ([_OK_STEP], [_FAILED_WIRE], False, True, "no_wire"),
        ([], [], True, True, "no_wire"),
    ],
    ids=["healthy", "no_agent_steps", "no_wire", "failed_wire_only", "completely_empty"],
)
def test_silent_failure_metric_per_trial(
    tmp_path: Path,
    steps: list,
    wire_rows: list,
    no_steps: bool,
    no_wire: bool,
    enrich_kind: str,
) -> None:
    """One-trial bench: silent-failure flags + the failed-wire row is ignored by enrichment."""
    bench = tmp_path / "pb"
    _write_jsonl(bench / "trajectories.jsonl", [_trial(0, 0, reward=0.0, steps=steps)])
    _write_jsonl(bench / "model_traffic.jsonl", wire_rows)
    report = json.loads(generate_trajectories_report(tmp_path, enrich=True).read_text())["benchmarks"][0]
    wc = report["wire_calls"]

    def _pct(flag: bool) -> str:
        return "1/1 (100.0%)" if flag else "0/1 (0.0%)"

    assert wc["trials_with_no_agent_steps"] == _pct(no_steps)
    assert wc["trials_with_no_wire_calls"] == _pct(no_wire)
    assert wc["trials_silent_either_way"] == _pct(no_steps or no_wire)
    assert report["enrichment"][_ENRICHMENT_FIELD[enrich_kind]] == 1


def test_field_coverage_only_lists_missing(bundle: Path) -> None:
    """Fully-populated fields are silent; only gaps appear -- with their presence ratio."""
    report = json.loads(generate_trajectories_report(bundle).read_text())["benchmarks"][0]
    fc = report["field_coverage"]
    # Fixture sets problem_idx/reward/agent.name on every trial → those don't appear.
    trial_misses = {e["field"]: e["presence"] for e in fc["per_trial_missing"]}
    assert "problem_idx" not in trial_misses
    assert "trajectory[0].schema_version" not in trial_misses
    # ...but agent.parent_agent and trajectory[0].extra are never set in the fixture.
    assert trial_misses["trajectory[0].agent.parent_agent"] == "0/2"
    assert trial_misses["trajectory[0].extra"] == "0/2"
    # Per-step: prompt_tokens always set, latency_ms/finish_reason never set.
    step_misses = {e["field"]: e["presence"] for e in fc["per_agent_step_missing"]}
    assert "step_id" not in step_misses
    assert "metrics.prompt_tokens" not in step_misses
    assert step_misses["metrics.extra.latency_ms"] == "0/3"
    assert step_misses["metrics.extra.finish_reason"] == "0/3"


def test_score(tmp_path: Path) -> None:
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    (bench / "eval-test.json").write_text(json.dumps({"benchmark": {"scores": {"summary": {"mean": 1.0}}}}))
    score = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["score"]
    assert score["mean_reward"] == 1.0
    assert score["reported_mean"] == 1.0


def test_no_benches_returns_none(tmp_path: Path) -> None:
    assert generate_trajectories_report(tmp_path) is None


def test_wire_dedup_via_request_hash(tmp_path: Path) -> None:
    """Identical request_hash → unique drops, the trial is flagged as having dups."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=5, ct=3)])],
    )
    dup = {**_wire(0, 0, prompt=10, completion=5), "request_hash": "abc"}
    _write_jsonl(bench / "model_traffic.jsonl", [dup, dup])
    wc = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["wire_calls"]
    assert wc["total"] == 2
    assert wc["unique"] == 1
    assert wc["trials_with_duplicates"] == 1


def test_tokens_per_step_vs_wire_mismatch(tmp_path: Path) -> None:
    """per_step_sum=0 but wire_total>0 → the two numbers disagree on their own."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="x", pt=0, ct=0)])],
    )
    _write_jsonl(bench / "model_traffic.jsonl", [_wire(0, 0, prompt=42, completion=7)])
    t = json.loads(generate_trajectories_report(tmp_path).read_text())["benchmarks"][0]["tokens"]
    assert t["per_step_sum"] == 0
    assert t["wire_total"] == 49


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
    # After backfill, no metric field is missing → the "missing" list is empty.
    assert enrichment["step_field_coverage_after_enrichment_missing"] == []


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


def test_enrich_backfills_full_capture_fields(tmp_path: Path) -> None:
    """When the wire row carries reasoning_content / message_content /
    tool_calls_full (because model_traffic was configured to capture them),
    enrichment backfills missing step fields too."""
    bench = tmp_path / "pb"
    _write_jsonl(
        bench / "trajectories.jsonl",
        [_trial(0, 0, reward=1.0, steps=[_agent_step(0, msg="", pt=5, ct=3)])],
    )
    wire = {
        **_wire(0, 0, prompt=5, completion=3),
        "reasoning_content": "I should call the tool first.",
        "message_content": "Final assistant message",
        "tool_calls_full": [{"id": "c1", "name": "search", "arguments": '{"q":"x"}'}],
    }
    _write_jsonl(bench / "model_traffic.jsonl", [wire])
    out = generate_trajectories_report(tmp_path, enrich=True)
    counts = json.loads(out.read_text())["benchmarks"][0]["enrichment"]["steps_backfilled"]
    assert counts["reasoning_content"] == 1
    assert counts["message"] == 1
    assert counts["tool_calls"] == 1
    # And the enriched step actually carries the fields
    step = json.loads((bench / "trajectories_enriched.jsonl").read_text().splitlines()[0])["trajectory"][0]["steps"][0]
    assert step["reasoning_content"] == "I should call the tool first."
    assert step["message"] == "Final assistant message"
    assert step["tool_calls"] == [{"id": "c1", "name": "search", "arguments": '{"q":"x"}'}]


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
