import json
from pathlib import Path

import pytest

from nemo_evaluator.config.gate_policy import GatePolicy
from nemo_evaluator.engine.bundles import discover_bundles
from nemo_evaluator.engine.gate import (
    BenchmarkGateResult,
    GateReport,
    _aggregate_verdict,
    _paired_delta_ci,
    gate_runs,
    write_gate_report,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _write_bundle(path: Path, name: str, scores: dict | None = None):
    bundle = {
        "run_id": f"run-{name}",
        "config": {"model": "test-model"},
        "benchmark": {
            "name": name,
            "samples": 100,
            "scores": scores or {"mean_reward": {"value": 0.7}},
        },
    }
    path.write_text(json.dumps(bundle))


def _write_results(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(pid, reward, repeat=0):
    return {"problem_idx": pid, "repeat": repeat, "reward": reward, "expected_answer": "ans"}


def _make_gate_dirs(tmp_path, benchmarks: dict):
    """Create nested baseline/candidate dirs.

    benchmarks: {name: (base_records, cand_records, base_scores, cand_scores)}
    """
    base_dir = tmp_path / "baseline"
    cand_dir = tmp_path / "candidate"
    for name, (base_recs, cand_recs, base_sc, cand_sc) in benchmarks.items():
        bd = base_dir / name
        cd = cand_dir / name
        bd.mkdir(parents=True)
        cd.mkdir(parents=True)
        _write_bundle(bd / f"eval-{name}.json", name, base_sc)
        _write_bundle(cd / f"eval-{name}.json", name, cand_sc)
        _write_results(bd / "results.jsonl", base_recs)
        _write_results(cd / "results.jsonl", cand_recs)
    return base_dir, cand_dir


def _simple_records(n, reward):
    """Create n records all with the same reward."""
    return [_record(i, reward) for i in range(n)]


def _scores(value):
    return {"mean_reward": {"value": value}}


def _score_entry(value, ci_lower=None, ci_upper=None):
    entry = {"value": value}
    if ci_lower is not None:
        entry["ci_lower"] = ci_lower
    if ci_upper is not None:
        entry["ci_upper"] = ci_upper
    return entry


# ── TestDiscoverBundles ───────────────────────────────────────────────


class TestDiscoverBundles:
    def test_nested_layout(self, tmp_path):
        sub = tmp_path / "mmlu"
        sub.mkdir()
        _write_bundle(sub / "eval-mmlu.json", "mmlu")
        bundles = discover_bundles(tmp_path)
        assert "mmlu" in bundles

    def test_flat_layout(self, tmp_path):
        _write_bundle(tmp_path / "eval-mmlu.json", "mmlu")
        bundles = discover_bundles(tmp_path)
        assert "mmlu" in bundles

    def test_duplicate_benchmark_fails(self, tmp_path):
        # Two bundles with same benchmark name in different locations
        _write_bundle(tmp_path / "eval-a.json", "mmlu")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_bundle(sub / "eval-b.json", "mmlu")
        with pytest.raises(ValueError, match="Duplicate benchmark"):
            discover_bundles(tmp_path)

    def test_empty_dir(self, tmp_path):
        bundles = discover_bundles(tmp_path)
        assert bundles == {}


# ── TestGateVerdicts ──────────────────────────────────────────────────


class TestGateVerdicts:
    def _policy(self, benchmarks: dict | None = None, **defaults):
        raw = {
            "version": 1,
            "defaults": {"tier": "critical", "metric": "mean_reward", "max_drop": 0.01, **defaults},
        }
        if benchmarks:
            raw["benchmarks"] = benchmarks
        return GatePolicy.model_validate(raw)

    def test_all_pass(self, tmp_path):
        """All benchmarks within threshold → GO."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (_simple_records(50, 0.8), _simple_records(50, 0.795), _scores(0.8), _scores(0.795)),
            },
        )
        policy = self._policy(benchmarks={"mmlu": {}})
        report = gate_runs(base_dir, cand_dir, policy)
        assert report.verdict == "GO"
        assert report.benchmarks[0].status == "PASS"

    def test_critical_breach(self, tmp_path):
        """Critical benchmark exceeds threshold → NO-GO."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (_simple_records(50, 1.0), _simple_records(50, 0.95), _scores(1.0), _scores(0.95)),
            },
        )
        policy = self._policy(benchmarks={"mmlu": {"tier": "critical"}})
        report = gate_runs(base_dir, cand_dir, policy)
        assert report.verdict == "NO-GO"
        mmlu = report.benchmarks[0]
        assert mmlu.status == "BREACH"
        assert mmlu.absolute_breached

    def test_supporting_breach_is_nogo(self, tmp_path):
        """Supporting benchmark breach also blocks (not just a warning)."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "triviaqa": (_simple_records(50, 1.0), _simple_records(50, 0.9), _scores(1.0), _scores(0.9)),
            },
        )
        policy = self._policy(benchmarks={"triviaqa": {"tier": "supporting", "max_drop": 0.015}})
        report = gate_runs(base_dir, cand_dir, policy)
        assert report.verdict == "NO-GO"

    def test_advisory_breach_is_go(self, tmp_path):
        """Advisory benchmark breach does NOT affect verdict."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "ifeval": (_simple_records(50, 1.0), _simple_records(50, 0.5), _scores(1.0), _scores(0.5)),
            },
        )
        policy = GatePolicy.model_validate(
            {
                "version": 1,
                "benchmarks": {"ifeval": {"tier": "advisory"}},
            }
        )
        report = gate_runs(base_dir, cand_dir, policy)
        assert report.verdict == "GO"

    def test_missing_required_in_candidate(self, tmp_path):
        """Policy requires benchmark but not found in candidate → NO-GO."""
        base_dir = tmp_path / "baseline" / "mmlu"
        base_dir.mkdir(parents=True)
        _write_bundle(base_dir / "eval-mmlu.json", "mmlu")
        _write_results(base_dir / "results.jsonl", _simple_records(20, 0.8))

        cand_dir = tmp_path / "candidate"
        cand_dir.mkdir(parents=True)

        policy = self._policy(benchmarks={"mmlu": {"tier": "critical"}})
        report = gate_runs(tmp_path / "baseline", cand_dir, policy)
        assert report.verdict == "NO-GO"
        assert "mmlu" in report.missing

    def test_missing_required_in_baseline(self, tmp_path):
        """Policy requires benchmark but not found in baseline → NO-GO."""
        base_dir = tmp_path / "baseline"
        base_dir.mkdir(parents=True)

        cand_dir = tmp_path / "candidate" / "mmlu"
        cand_dir.mkdir(parents=True)
        _write_bundle(cand_dir / "eval-mmlu.json", "mmlu")
        _write_results(cand_dir / "results.jsonl", _simple_records(20, 0.8))

        policy = self._policy(benchmarks={"mmlu": {"tier": "critical"}})
        report = gate_runs(base_dir, tmp_path / "candidate", policy)
        assert report.verdict == "NO-GO"
        assert "mmlu" in report.missing

    def test_relative_drop_breach(self, tmp_path):
        """Low baseline + relative drop exceeded → BREACH."""
        # Baseline at 10%, candidate at 8%: absolute drop 2pp is within 1.5pp? No, 0.02 > 0.015.
        # But let's test relative specifically: baseline 15%, candidate 13%.
        # Absolute: 0.02 drop > 0.015 threshold → would breach anyway.
        # Better: baseline 10%, candidate 9.5%. Absolute drop = 0.005 < 0.015. Relative = 5% > 2%.
        base_recs = _simple_records(50, 0.10)
        cand_recs = _simple_records(50, 0.095)
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "hle": (base_recs, cand_recs, _scores(0.10), _scores(0.095)),
            },
        )
        policy = self._policy(
            benchmarks={
                "hle": {
                    "tier": "critical",
                    "max_drop": 0.015,  # absolute: 0.005 < 0.015, passes
                    "max_relative_drop": 0.02,  # relative: 5% > 2%, breaches
                    "relative_guard_below": 0.20,
                },
            }
        )
        report = gate_runs(base_dir, cand_dir, policy)
        hle = next(b for b in report.benchmarks if b.benchmark == "hle")
        assert hle.relative_breached
        assert hle.status == "BREACH"

    def test_relative_guard_skips_high_baseline(self, tmp_path):
        """Relative check skipped when baseline is above guard threshold."""
        # Baseline 80%, candidate 76%: absolute 4pp > 1.5pp → BREACH on absolute.
        # But if we set max_drop high to avoid absolute breach:
        base_recs = _simple_records(50, 0.80)
        cand_recs = _simple_records(50, 0.76)
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (base_recs, cand_recs, _scores(0.80), _scores(0.76)),
            },
        )
        policy = self._policy(
            benchmarks={
                "mmlu": {
                    "max_drop": 0.05,  # absolute: 0.04 < 0.05, passes
                    "max_relative_drop": 0.02,  # relative: 5% > 2%, BUT...
                    "relative_guard_below": 0.20,  # baseline 0.80 > 0.20, skip relative
                },
            }
        )
        report = gate_runs(base_dir, cand_dir, policy)
        mmlu = next(b for b in report.benchmarks if b.benchmark == "mmlu")
        assert not mmlu.relative_breached
        assert mmlu.status == "PASS"

    def test_improvement_does_not_breach(self, tmp_path):
        """Positive improvement must NOT trigger BREACH (direction-aware)."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (_simple_records(50, 0.7), _simple_records(50, 0.9), _scores(0.7), _scores(0.9)),
            },
        )
        policy = self._policy(benchmarks={"mmlu": {}})
        report = gate_runs(base_dir, cand_dir, policy)
        mmlu = report.benchmarks[0]
        assert mmlu.status == "PASS"
        assert not mmlu.absolute_breached
        assert not mmlu.relative_breached

    def test_lower_is_better_increase_is_regression(self, tmp_path):
        """For lower-is-better metrics, an increase is a regression."""
        base_recs = _simple_records(50, 5.0)
        cand_recs = _simple_records(50, 5.5)
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "loss": (base_recs, cand_recs, _scores(5.0), _scores(5.5)),
            },
        )
        policy = self._policy(
            benchmarks={"loss": {"direction": "lower_is_better", "max_drop": 0.3}},
        )
        report = gate_runs(base_dir, cand_dir, policy)
        loss = report.benchmarks[0]
        assert loss.status == "BREACH"
        assert loss.absolute_breached

    def test_lower_is_better_decrease_is_improvement(self, tmp_path):
        """For lower-is-better, a decrease is improvement → PASS."""
        base_recs = _simple_records(50, 5.0)
        cand_recs = _simple_records(50, 4.5)
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "loss": (base_recs, cand_recs, _scores(5.0), _scores(4.5)),
            },
        )
        policy = self._policy(
            benchmarks={"loss": {"direction": "lower_is_better", "max_drop": 0.3}},
        )
        report = gate_runs(base_dir, cand_dir, policy)
        assert report.benchmarks[0].status == "PASS"

    def test_metric_selection_pass1_does_not_gate_on_mean_reward(self, tmp_path):
        """Selected metric must drive the gate math."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "code": (
                    _simple_records(50, 0.6),
                    _simple_records(50, 0.4),
                    {
                        "mean_reward": _score_entry(0.6, 0.6, 0.6),
                        "pass@1": _score_entry(1.0, 1.0, 1.0),
                    },
                    {
                        "mean_reward": _score_entry(0.4, 0.4, 0.4),
                        "pass@1": _score_entry(1.0, 1.0, 1.0),
                    },
                ),
            },
        )
        policy = self._policy(benchmarks={"code": {"metric": "pass@1"}})
        report = gate_runs(base_dir, cand_dir, policy)
        code = report.benchmarks[0]
        assert code.metric == "pass@1"
        assert code.baseline_score == 1.0
        assert code.candidate_score == 1.0
        assert code.status == "PASS"

    def test_per_benchmark_thresholds(self, tmp_path):
        """Different max_drop per benchmark honored."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                # mmlu: drop 0.02 > 0.01 threshold → BREACH
                "mmlu": (_simple_records(50, 0.8), _simple_records(50, 0.78), _scores(0.8), _scores(0.78)),
                # gpqa: drop 0.02 < 0.05 threshold → PASS
                "gpqa": (_simple_records(50, 0.5), _simple_records(50, 0.48), _scores(0.5), _scores(0.48)),
            },
        )
        policy = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"metric": "mean_reward"},
                "benchmarks": {
                    "mmlu": {"tier": "critical", "max_drop": 0.01},
                    "gpqa": {"tier": "critical", "max_drop": 0.05},
                },
            }
        )
        report = gate_runs(base_dir, cand_dir, policy)
        mmlu = next(b for b in report.benchmarks if b.benchmark == "mmlu")
        gpqa = next(b for b in report.benchmarks if b.benchmark == "gpqa")
        assert mmlu.status == "BREACH"
        assert gpqa.status == "PASS"
        assert report.verdict == "NO-GO"

    def test_insufficient_evidence_no_results(self, tmp_path):
        """Missing results.jsonl → INSUFFICIENT_EVIDENCE."""
        base_dir = tmp_path / "baseline" / "mmlu"
        cand_dir = tmp_path / "candidate" / "mmlu"
        base_dir.mkdir(parents=True)
        cand_dir.mkdir(parents=True)
        _write_bundle(base_dir / "eval-mmlu.json", "mmlu")
        _write_bundle(cand_dir / "eval-mmlu.json", "mmlu")
        # No results.jsonl written

        policy = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"metric": "mean_reward"},
                "benchmarks": {"mmlu": {"tier": "critical", "max_drop": 0.01}},
            }
        )
        report = gate_runs(tmp_path / "baseline", tmp_path / "candidate", policy)
        assert report.benchmarks[0].status == "INSUFFICIENT_EVIDENCE"
        assert report.verdict == "INCONCLUSIVE"

    def test_insufficient_evidence_few_items(self, tmp_path):
        """< 10 paired items → INSUFFICIENT_EVIDENCE."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (
                    _simple_records(5, 0.8),
                    _simple_records(5, 0.7),
                    _scores(0.8),
                    _scores(0.7),
                ),
            },
        )
        policy = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"metric": "mean_reward"},
                "benchmarks": {"mmlu": {"tier": "critical", "max_drop": 0.01}},
            }
        )
        report = gate_runs(base_dir, cand_dir, policy)
        assert report.benchmarks[0].status == "INSUFFICIENT_EVIDENCE"

    def test_ci_straddling_threshold_is_inconclusive(self, tmp_path):
        """A paired delta CI that crosses the threshold should not produce PASS/BREACH."""
        base = []
        cand = []
        for i in range(25):
            base.append(_record(i, 1.0))
            cand.append(_record(i, 0.98))
        for i in range(25, 50):
            base.append(_record(i, 1.0))
            cand.append(_record(i, 1.0))

        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (base, cand, _scores(1.0), _scores(0.99)),
            },
        )
        policy = self._policy(benchmarks={"mmlu": {"max_drop": 0.01}})
        report = gate_runs(base_dir, cand_dir, policy)
        mmlu = report.benchmarks[0]
        assert mmlu.status == "INSUFFICIENT_EVIDENCE"
        assert report.verdict == "INCONCLUSIVE"

    def test_bundle_ci_fallback_can_breach(self, tmp_path):
        """Bundle score CIs should be used when paired records are unavailable."""
        base_dir = tmp_path / "baseline" / "mmlu"
        cand_dir = tmp_path / "candidate" / "mmlu"
        base_dir.mkdir(parents=True)
        cand_dir.mkdir(parents=True)
        _write_bundle(
            base_dir / "eval-mmlu.json",
            "mmlu",
            {"mean_reward": _score_entry(0.80, 0.80, 0.80)},
        )
        _write_bundle(
            cand_dir / "eval-mmlu.json",
            "mmlu",
            {"mean_reward": _score_entry(0.78, 0.78, 0.78)},
        )

        policy = self._policy(benchmarks={"mmlu": {"max_drop": 0.01}})
        report = gate_runs(tmp_path / "baseline", tmp_path / "candidate", policy)
        mmlu = report.benchmarks[0]
        assert mmlu.status == "BREACH"
        assert mmlu.absolute_breached

    def test_missing_metric_for_required_benchmark_fails_fast(self, tmp_path):
        """Required benchmarks must resolve to an explicit supported metric."""
        base_dir, cand_dir = _make_gate_dirs(
            tmp_path,
            {
                "mmlu": (_simple_records(50, 0.8), _simple_records(50, 0.79), _scores(0.8), _scores(0.79)),
            },
        )
        policy = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"tier": "critical", "max_drop": 0.01},
                "benchmarks": {"mmlu": {}},
            }
        )
        with pytest.raises(ValueError, match="explicit metric"):
            gate_runs(base_dir, cand_dir, policy)


# ── TestPairedDeltaCI ─────────────────────────────────────────────────


class TestPairedDeltaCI:
    def test_basic_ci(self):
        deltas = [0.1] * 100  # All identical → very tight CI
        lo, hi = _paired_delta_ci(deltas)
        assert lo is not None and hi is not None
        assert abs(lo - 0.1) < 0.01
        assert abs(hi - 0.1) < 0.01

    def test_wide_ci_small_n(self):
        deltas = [0.5, -0.5, 0.3, -0.3, 0.1]
        lo, hi = _paired_delta_ci(deltas)
        assert lo is not None and hi is not None
        assert lo < 0  # CI crosses zero
        assert hi > 0

    def test_single_item_returns_none(self):
        lo, hi = _paired_delta_ci([0.5])
        assert lo is None and hi is None


class TestWriteGateReport:
    def test_serialized_report_keeps_regression_evidence(self, tmp_path):
        report = GateReport(
            benchmarks=[
                BenchmarkGateResult(
                    benchmark="mmlu",
                    tier="critical",
                    regression_report={"verdict": "PASS"},
                )
            ]
        )
        path = write_gate_report(report, tmp_path / "gate.json")
        data = json.loads(path.read_text())
        assert data["benchmarks"][0]["regression_report"] == {"verdict": "PASS"}


# ── TestAggregateVerdict ──────────────────────────────────────────────


class TestAggregateVerdict:
    def test_all_pass(self):
        benchmarks = [
            BenchmarkGateResult(benchmark="a", tier="critical", status="PASS"),
            BenchmarkGateResult(benchmark="b", tier="supporting", status="PASS"),
        ]
        verdict, _ = _aggregate_verdict(benchmarks)
        assert verdict == "GO"

    def test_critical_breach(self):
        benchmarks = [
            BenchmarkGateResult(benchmark="a", tier="critical", status="BREACH"),
        ]
        verdict, reasons = _aggregate_verdict(benchmarks)
        assert verdict == "NO-GO"
        assert "BREACH" in reasons[0]

    def test_missing_is_nogo(self):
        benchmarks = [
            BenchmarkGateResult(benchmark="a", tier="critical", status="MISSING"),
        ]
        verdict, _ = _aggregate_verdict(benchmarks)
        assert verdict == "NO-GO"

    def test_insufficient_evidence_is_inconclusive(self):
        benchmarks = [
            BenchmarkGateResult(benchmark="a", tier="critical", status="INSUFFICIENT_EVIDENCE"),
        ]
        verdict, _ = _aggregate_verdict(benchmarks)
        assert verdict == "INCONCLUSIVE"

    def test_advisory_ignored(self):
        benchmarks = [
            BenchmarkGateResult(benchmark="a", tier="advisory", status="BREACH"),
        ]
        verdict, _ = _aggregate_verdict(benchmarks)
        assert verdict == "GO"

    def test_breach_takes_precedence_over_insufficient(self):
        benchmarks = [
            BenchmarkGateResult(benchmark="a", tier="critical", status="BREACH"),
            BenchmarkGateResult(benchmark="b", tier="critical", status="INSUFFICIENT_EVIDENCE"),
        ]
        verdict, _ = _aggregate_verdict(benchmarks)
        assert verdict == "NO-GO"


# ── TestWriteGateReport ───────────────────────────────────────────────


class TestWriteGateReportRoundTrip:
    def test_round_trip(self, tmp_path):
        report = GateReport(
            verdict="GO",
            verdict_reasons=["All passed"],
            benchmarks=[BenchmarkGateResult(benchmark="mmlu", tier="critical", status="PASS")],
        )
        path = write_gate_report(report, tmp_path / "gate.json")
        loaded = json.loads(path.read_text())
        assert loaded["verdict"] == "GO"
        assert len(loaded["benchmarks"]) == 1
        assert loaded["benchmarks"][0]["benchmark"] == "mmlu"


class TestGateMarkdownReport:
    def test_generates_markdown(self, tmp_path):
        from nemo_evaluator.reports.gate import render_markdown as generate_gate_report, write_gate_markdown

        report_dict = GateReport(
            verdict="NO-GO",
            verdict_reasons=["BREACH: gpqa [critical]"],
            benchmarks=[
                BenchmarkGateResult(
                    benchmark="mmlu_pro",
                    tier="critical",
                    status="PASS",
                    metric="mean_reward",
                    baseline_score=0.782,
                    candidate_score=0.775,
                    delta=-0.007,
                    delta_ci_lower=-0.012,
                    delta_ci_upper=-0.002,
                    n_paired=12000,
                ),
                BenchmarkGateResult(
                    benchmark="gpqa",
                    tier="critical",
                    status="BREACH",
                    metric="mean_reward",
                    baseline_score=0.412,
                    candidate_score=0.398,
                    delta=-0.014,
                    delta_ci_lower=-0.038,
                    delta_ci_upper=0.010,
                    n_paired=198,
                    reasons=["95% CI on damage exceeds threshold"],
                ),
            ],
            warnings=["Candidate-only benchmarks (no baseline): hle"],
        ).to_dict()

        md = generate_gate_report(report_dict)
        assert "# Quality Gate Report: NO-GO" in md
        assert "mmlu_pro" in md
        assert "gpqa" in md
        assert "BREACH" in md
        assert "12,000" in md  # formatted N

        path = write_gate_markdown(report_dict, tmp_path / "damage.md")
        assert path.exists()
        assert "NO-GO" in path.read_text()

    def test_empty_report(self):
        from nemo_evaluator.reports.gate import render_markdown as generate_gate_report

        md = generate_gate_report({"verdict": "GO", "benchmarks": [], "warnings": []})
        assert "# Quality Gate Report: GO" in md


class TestClusteredCIIntegration:
    def test_extract_clusters_with_categories(self):
        from nemo_evaluator.engine.gate import _extract_clusters

        records = {
            (0, 0): {"reward": 1.0, "metadata": {"category": "algebra"}},
            (1, 0): {"reward": 0.0, "metadata": {"category": "geometry"}},
            (2, 0): {"reward": 1.0, "metadata": {"category": "algebra"}},
        }
        clusters = _extract_clusters(records, [0, 1, 2])
        assert clusters == ["algebra", "geometry", "algebra"]

    def test_extract_clusters_returns_none_without_categories(self):
        from nemo_evaluator.engine.gate import _extract_clusters

        records = {
            (0, 0): {"reward": 1.0},
            (1, 0): {"reward": 0.0},
        }
        clusters = _extract_clusters(records, [0, 1])
        assert clusters is None

    def test_paired_delta_ci_with_clusters(self):
        from nemo_evaluator.engine.gate import _paired_delta_ci

        # 20 deltas across 2 clusters — clustered CI should be wider than normal
        deltas = [
            0.1,
            0.12,
            0.09,
            0.11,
            0.1,
            0.08,
            0.13,
            0.11,
            0.1,
            0.09,
            -0.05,
            -0.04,
            -0.06,
            -0.03,
            -0.05,
            -0.04,
            -0.06,
            -0.03,
            -0.05,
            -0.04,
        ]
        clusters = ["A"] * 10 + ["B"] * 10

        ci_normal = _paired_delta_ci(deltas)
        ci_clustered = _paired_delta_ci(deltas, clusters=clusters)

        assert ci_normal[0] is not None
        assert ci_clustered[0] is not None
        # Clustered CI should be wider (or at least different) when clusters have different means
        width_normal = ci_normal[1] - ci_normal[0]
        width_clustered = ci_clustered[1] - ci_clustered[0]
        assert width_clustered > width_normal, "Clustered CI should be strictly wider when cluster means differ"
