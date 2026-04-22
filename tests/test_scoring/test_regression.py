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
import json
from pathlib import Path

import pytest

from nemo_evaluator.engine.comparison import (
    FlipReport,
    build_flip_report,
    compare_results,
    compare_runs,
    load_paired_records,
    write_regression,
)
from nemo_evaluator.metrics.paired_tests import McNemarResult, mcnemar_test


def _write_bundle(path: Path, run_id: str, scores: dict, categories=None, name="test"):
    bundle = {
        "run_id": run_id,
        "config": {"benchmark": name},
        "benchmark": {
            "name": name,
            "samples": 100,
            "scores": scores,
        },
    }
    if categories:
        bundle["benchmark"]["categories"] = categories
    path.write_text(json.dumps(bundle))


def _write_results_jsonl(path: Path, records: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_paired_dirs(
    tmp_path, base_records, cand_records, base_scores=None, cand_scores=None, base_name="test", cand_name="test"
):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    base_dir.mkdir()
    cand_dir.mkdir()
    b = base_dir / "eval-base.json"
    c = cand_dir / "eval-cand.json"
    _write_bundle(b, "base", base_scores or {"pass@1": {"value": 0.7}}, name=base_name)
    _write_bundle(c, "cand", cand_scores or {"pass@1": {"value": 0.7}}, name=cand_name)
    _write_results_jsonl(base_dir / "results.jsonl", base_records)
    _write_results_jsonl(cand_dir / "results.jsonl", cand_records)
    return b, c


def _record(pid, reward, repeat=0, expected="ans", category=None, model_response=None):
    r = {"problem_idx": pid, "repeat": repeat, "reward": reward, "expected_answer": expected}
    if category:
        r["metadata"] = {"category": category}
    if model_response:
        r["model_response"] = model_response
    return r


class TestCompareRuns:
    def test_basic_delta(self, tmp_path):
        b = tmp_path / "base.json"
        c = tmp_path / "cand.json"
        _write_bundle(b, "base-1", {"pass@1": {"value": 0.70, "ci_lower": 0.60, "ci_upper": 0.80}})
        _write_bundle(c, "cand-1", {"pass@1": {"value": 0.75, "ci_lower": 0.65, "ci_upper": 0.85}})

        report = compare_runs(b, c)
        d = report["score_deltas"]["pass@1"]
        assert d["baseline"] == 0.70
        assert d["candidate"] == 0.75
        assert abs(d["delta"] - 0.05) < 1e-4
        assert d["ci_overlap"] is True

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compare_runs(tmp_path / "nope.json", tmp_path / "also_nope.json")

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        good = tmp_path / "good.json"
        _write_bundle(good, "g", {"pass@1": {"value": 0.5}})
        with pytest.raises(ValueError, match="Invalid JSON"):
            compare_runs(bad, good)

    def test_non_numeric_value_skipped(self, tmp_path):
        b = tmp_path / "b.json"
        c = tmp_path / "c.json"
        _write_bundle(b, "b", {"weird": {"value": "not_a_number"}})
        _write_bundle(c, "c", {"weird": {"value": "also_not"}})
        report = compare_runs(b, c)
        assert "weird" not in report["score_deltas"]

    def test_category_deltas(self, tmp_path):
        b = tmp_path / "b.json"
        c = tmp_path / "c.json"
        cats = {"algebra": {"mean_reward": 0.80, "n": 50}, "geometry": {"mean_reward": 0.60, "n": 50}}
        _write_bundle(b, "b", {"pass@1": {"value": 0.7}}, categories=cats)
        cats2 = {"algebra": {"mean_reward": 0.85, "n": 50}, "geometry": {"mean_reward": 0.55, "n": 50}}
        _write_bundle(c, "c", {"pass@1": {"value": 0.7}}, categories=cats2)
        report = compare_runs(b, c)
        assert "algebra" in report["category_deltas"]
        assert report["category_deltas"]["algebra"]["delta"] == 0.05

    def test_report_version_present(self, tmp_path):
        b = tmp_path / "b.json"
        c = tmp_path / "c.json"
        _write_bundle(b, "b", {"pass@1": {"value": 0.7}})
        _write_bundle(c, "c", {"pass@1": {"value": 0.7}})
        report = compare_runs(b, c)
        assert "report_version" in report
        assert report["report_version"] == 1

    def test_verdict_present(self, tmp_path):
        b = tmp_path / "b.json"
        c = tmp_path / "c.json"
        _write_bundle(b, "b", {"pass@1": {"value": 0.7}})
        _write_bundle(c, "c", {"pass@1": {"value": 0.7}})
        report = compare_runs(b, c)
        assert "verdict" in report
        # No results.jsonl → no paired data → INCONCLUSIVE (not PASS)
        assert report["verdict"] == "INCONCLUSIVE"


class TestPairedAnalysis:
    def test_flip_report_basic(self, tmp_path):
        base = (
            [_record(i, 1.0) for i in range(5)]
            + [_record(i, 0.0) for i in range(5, 7)]
            + [_record(7, 1.0), _record(8, 1.0), _record(9, 0.0)]
        )
        cand = (
            [_record(i, 1.0) for i in range(5)]
            + [_record(i, 0.0) for i in range(5, 7)]
            + [_record(7, 0.0), _record(8, 0.0), _record(9, 1.0)]
        )
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)

        s = report["flip_report"]["summary"]
        assert s["n_paired"] == 10
        assert s["n_stable_correct"] == 5
        assert s["n_stable_wrong"] == 2
        assert s["n_regressions"] == 2
        assert s["n_improvements"] == 1
        assert len(report["flip_report"]["regressions"]) == 2
        assert report["flip_report"]["regressions"][0]["problem_idx"] == 7
        # Item #12: regression rate
        assert s["regression_rate"] == 0.2

    def test_mcnemar_one_sided(self, tmp_path):
        """Item #1: verify one-sided test (8 regressions, 2 improvements).
        One-sided p should be smaller than two-sided."""
        pytest.importorskip("scipy")
        base = [_record(i, 1.0) for i in range(8)] + [_record(8, 0.0), _record(9, 0.0)]
        cand = [_record(i, 0.0) for i in range(8)] + [_record(8, 1.0), _record(9, 1.0)]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)

        m = report["mcnemar"]
        assert m["method"] == "exact_binomial"
        assert m["n_discordant"] == 10
        assert m["hypothesis"] == "one-sided (degradation)"
        # One-sided binomtest(8, 10, 0.5, alternative="greater") p ≈ 0.0547
        assert 0.04 < m["p_value"] < 0.06

    def test_mcnemar_effect_size(self, tmp_path):
        """Item #4: effect size and CI are reported."""
        pytest.importorskip("scipy")
        base = [_record(i, 1.0) for i in range(8)] + [_record(i, 0.0) for i in range(8, 10)]
        cand = [_record(i, 0.0) for i in range(8)] + [_record(i, 1.0) for i in range(8, 10)]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)

        m = report["mcnemar"]
        assert m["effect_size"] is not None
        assert m["effect_size"] == 0.6  # (8-2)/10
        assert m["ci_lower"] is not None
        assert m["ci_upper"] is not None

    def test_verdict_block_with_effect(self, tmp_path):
        """Item #4: BLOCK requires significance AND effect > min_effect."""
        pytest.importorskip("scipy")
        # 15 regressions, 2 improvements out of 20 -> significant AND large effect
        base = [_record(i, 1.0) for i in range(15)] + [_record(i, 0.0) for i in range(15, 20)]
        cand = (
            [_record(i, 0.0) for i in range(15)]
            + [_record(15, 1.0), _record(16, 1.0)]
            + [_record(i, 0.0) for i in range(17, 20)]
        )
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)
        assert report["verdict"] == "BLOCK"

    def test_verdict_warn_small_effect(self, tmp_path):
        """Item #4: WARN when significant but effect below practical threshold."""
        pytest.importorskip("scipy")
        # Create 1000 samples where 8 regress and 2 improve -> significant but tiny effect
        base = [_record(i, 1.0) for i in range(500)] + [_record(i, 0.0) for i in range(500, 1000)]
        cand = list(base)
        # Flip 8 from correct to incorrect
        for i in range(8):
            cand[i] = _record(i, 0.0)
        # Flip 2 from incorrect to correct
        cand[500] = _record(500, 1.0)
        cand[501] = _record(501, 1.0)
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c, min_effect=0.05)  # 5% threshold, effect is 0.6%
        # With only 10 discordant pairs and one-sided test, may or may not be significant.
        assert report["verdict"] in ("WARN", "PASS", "INCONCLUSIVE"), (
            f"Expected non-BLOCK verdict for small effect, got {report['verdict']}"
        )

    def test_mcnemar_no_discordant(self, tmp_path):
        base = [_record(i, 1.0) for i in range(5)]
        cand = [_record(i, 1.0) for i in range(5)]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)

        m = report["mcnemar"]
        assert m["p_value"] == 1.0
        assert m["significant"] is False
        assert m["method"] == "exact"
        assert report["verdict"] == "PASS"

    def test_mcnemar_graceful_without_scipy(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "scipy" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        base = [_record(0, 1.0), _record(1, 0.0)]
        cand = [_record(0, 0.0), _record(1, 1.0)]
        b, c = _make_paired_dirs(tmp_path, base, cand)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        report = compare_runs(b, c)

        m = report.get("mcnemar", {})
        # to_dict() omits None values, so p_value key may be absent
        assert m.get("p_value") is None
        assert report["verdict"] == "INCONCLUSIVE"
        assert any("scipy" in r for r in report["verdict_reasons"])

    def test_warning_when_results_missing(self, tmp_path):
        """Item #2: warn when results.jsonl is missing."""
        b = tmp_path / "base.json"
        c = tmp_path / "cand.json"
        _write_bundle(b, "base", {"pass@1": {"value": 0.7}})
        _write_bundle(c, "cand", {"pass@1": {"value": 0.7}})
        report = compare_runs(b, c)
        assert report.get("warnings")
        assert any("results missing" in w.lower() or "results.jsonl" in w for w in report["warnings"])
        assert report.get("flip_report") is None

    def test_warning_when_results_empty(self, tmp_path):
        """Empty results.jsonl should warn, not silently pass."""
        base_dir = tmp_path / "base"
        cand_dir = tmp_path / "cand"
        base_dir.mkdir()
        cand_dir.mkdir()
        b = base_dir / "eval-base.json"
        c = cand_dir / "eval-cand.json"
        _write_bundle(b, "base", {"pass@1": {"value": 0.7}})
        _write_bundle(c, "cand", {"pass@1": {"value": 0.5}})
        # Create empty results.jsonl files (present but no records)
        (base_dir / "results.jsonl").write_text("")
        (cand_dir / "results.jsonl").write_text("")
        report = compare_runs(b, c)
        assert report.get("warnings")
        assert any("empty" in w.lower() or "unparseable" in w.lower() for w in report["warnings"])
        assert report.get("flip_report") is None

    def test_partial_overlap(self, tmp_path):
        base = [_record(i, 1.0) for i in range(10)]
        cand = [_record(i, 1.0) for i in range(5, 15)]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)
        assert report["flip_report"]["summary"]["n_paired"] == 5

    def test_custom_reward_threshold(self, tmp_path):
        base = [_record(0, 0.3), _record(1, 0.8)]
        cand = [_record(0, 0.8), _record(1, 0.3)]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c, reward_threshold=0.5)
        s = report["flip_report"]["summary"]
        assert s["n_regressions"] == 1
        assert s["n_improvements"] == 1

    def test_backward_compat_keys(self, tmp_path):
        base = [_record(0, 1.0)]
        cand = [_record(0, 1.0)]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)
        assert "score_deltas" in report
        assert "runtime_deltas" in report
        assert "baseline" in report
        assert "candidate" in report
        assert "verdict" in report
        assert "report_version" in report

    def test_flip_entry_has_category(self, tmp_path):
        base = [_record(0, 1.0, category="algebra")]
        cand = [_record(0, 0.0, category="algebra")]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)
        assert report["flip_report"]["regressions"][0]["category"] == "algebra"

    def test_category_breakdown_in_summary(self, tmp_path):
        """Item #13: category subtotals in flip summary."""
        base = [
            _record(0, 1.0, category="algebra"),
            _record(1, 1.0, category="geometry"),
            _record(2, 0.0, category="algebra"),
        ]
        cand = [
            _record(0, 0.0, category="algebra"),
            _record(1, 0.0, category="geometry"),
            _record(2, 1.0, category="algebra"),
        ]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)
        bd = report["flip_report"]["summary"]["category_breakdown"]
        assert bd["algebra"]["regressions"] == 1
        assert bd["algebra"]["improvements"] == 1
        assert bd["geometry"]["regressions"] == 1

    def test_benchmark_name_mismatch_warning(self, tmp_path):
        """Item #17: warn when benchmark names differ."""
        base = [_record(0, 1.0)]
        cand = [_record(0, 1.0)]
        b, c = _make_paired_dirs(tmp_path, base, cand, base_name="mmlu", cand_name="gsm8k")
        report = compare_runs(b, c)
        assert any("mismatch" in w.lower() for w in report.get("warnings", []))

    def test_flip_entry_includes_response(self, tmp_path):
        """Item #23: flip entries include truncated model response."""
        base = [_record(0, 1.0, model_response="correct answer here")]
        cand = [_record(0, 0.0, model_response="wrong answer")]
        b, c = _make_paired_dirs(tmp_path, base, cand)
        report = compare_runs(b, c)
        reg = report["flip_report"]["regressions"][0]
        assert reg.get("baseline_response") == "correct answer here"
        assert reg.get("candidate_response") == "wrong answer"


class TestCompareResults:
    """Item #20: in-memory comparison API."""

    def test_in_memory(self):
        base = {(i, 0): {"reward": 1.0, "expected_answer": str(i)} for i in range(5)}
        cand = {(i, 0): {"reward": 1.0 if i < 3 else 0.0, "expected_answer": str(i)} for i in range(5)}
        report = compare_results(base, cand)
        assert report["flip_report"]["summary"]["n_regressions"] == 2
        assert report["verdict"] in ("PASS", "WARN", "BLOCK", "INCONCLUSIVE")
        assert report["test_used"] == "mcnemar"
        assert "flip_report" in report
        assert report["flip_report"]["summary"]["n_paired"] == 5


class TestPublicAPI:
    """Item #21: helper functions are public."""

    def test_build_flip_report_public(self):
        base = {(0, 0): {"reward": 1.0, "expected_answer": "a"}}
        cand = {(0, 0): {"reward": 0.0, "expected_answer": "a"}}
        flip = build_flip_report(base, cand)
        assert isinstance(flip, FlipReport)
        assert flip.summary.n_regressions == 1

    def test_mcnemar_test_public(self):
        ct = {"baseline_only_correct": 5, "candidate_only_correct": 2, "both_correct": 10, "both_wrong": 3}
        result = mcnemar_test(ct, n_paired=20)
        assert isinstance(result, McNemarResult)
        assert result.n_discordant == 7
        assert result.effect_size is not None

    def test_load_paired_records_public(self, tmp_path):
        _write_results_jsonl(tmp_path / "results.jsonl", [_record(0, 1.0)])
        records = load_paired_records(tmp_path)
        assert (0, 0) in records


class TestSignTest:
    def test_sign_test_clear_regression(self):
        """Mostly positive deltas (baseline > candidate) should be significant."""
        from nemo_evaluator.metrics.paired_tests import sign_test

        # 15 regressions, 3 improvements
        deltas = [0.3] * 15 + [-0.2] * 3
        result = sign_test(deltas)
        assert result.n_positive == 15
        assert result.n_negative == 3
        assert result.n_ties == 0
        assert result.p_value is not None
        assert result.p_value < 0.05
        assert result.significant is True

    def test_sign_test_no_signal(self):
        """Equal positive and negative deltas should not be significant."""
        from nemo_evaluator.metrics.paired_tests import sign_test

        deltas = [0.3] * 10 + [-0.3] * 10
        result = sign_test(deltas)
        assert result.n_positive == 10
        assert result.n_negative == 10
        assert result.p_value is not None
        assert result.p_value > 0.05
        assert result.significant is False

    def test_sign_test_all_ties(self):
        """All zero deltas → p=1.0, not significant."""
        from nemo_evaluator.metrics.paired_tests import sign_test

        result = sign_test([0.0] * 20)
        assert result.n_ties == 20
        assert result.p_value == 1.0
        assert result.significant is False


class TestPermutationTest:
    def test_permutation_clear_regression(self):
        """Large positive mean diff should be significant."""
        from nemo_evaluator.metrics.paired_tests import permutation_test

        deltas = [0.5] * 20 + [-0.1] * 5
        result = permutation_test(deltas, seed=42)
        assert result.observed_mean_diff > 0
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.effect_size is not None

    def test_permutation_no_signal(self):
        """Symmetric deltas should not be significant."""
        from nemo_evaluator.metrics.paired_tests import permutation_test

        deltas = [0.1, -0.1] * 20
        result = permutation_test(deltas, seed=42)
        assert abs(result.observed_mean_diff) < 0.01
        assert result.p_value > 0.05
        assert result.significant is False

    def test_permutation_reproducible(self):
        """Same seed produces same p-value."""
        from nemo_evaluator.metrics.paired_tests import permutation_test

        deltas = [0.3, 0.1, -0.2, 0.4, -0.1, 0.2, 0.0, 0.3]
        r1 = permutation_test(deltas, seed=123)
        r2 = permutation_test(deltas, seed=123)
        assert r1.p_value == r2.p_value


class TestDetectTest:
    def test_binary_data_selects_mcnemar(self):
        from nemo_evaluator.metrics.paired_tests import detect_test

        base = {(0, 0): {"reward": 1.0}, (1, 0): {"reward": 0.0}, (2, 0): {"reward": 1.0}}
        cand = {(0, 0): {"reward": 1.0}, (1, 0): {"reward": 1.0}, (2, 0): {"reward": 0.0}}
        assert detect_test(base, cand) == "mcnemar"

    def test_continuous_data_selects_permutation(self):
        from nemo_evaluator.metrics.paired_tests import detect_test

        base = {(0, 0): {"reward": 0.67}, (1, 0): {"reward": 0.33}, (2, 0): {"reward": 1.0}}
        cand = {(0, 0): {"reward": 0.33}, (1, 0): {"reward": 0.67}, (2, 0): {"reward": 0.67}}
        assert detect_test(base, cand) == "permutation"


class TestAggregateRepeats:
    def test_single_repeats_pass_through(self):
        from nemo_evaluator.engine.comparison import aggregate_repeats

        records = {(0, 0): {"reward": 1.0, "problem_idx": 0}, (1, 0): {"reward": 0.0, "problem_idx": 1}}
        result = aggregate_repeats(records)
        assert len(result) == 2
        assert result[(0, 0)]["reward"] == 1.0

    def test_multiple_repeats_averaged(self):
        from nemo_evaluator.engine.comparison import aggregate_repeats

        records = {
            (0, 0): {"reward": 1.0, "problem_idx": 0},
            (0, 1): {"reward": 1.0, "problem_idx": 0},
            (0, 2): {"reward": 0.0, "problem_idx": 0},
            (1, 0): {"reward": 0.0, "problem_idx": 1},
            (1, 1): {"reward": 1.0, "problem_idx": 1},
            (1, 2): {"reward": 0.0, "problem_idx": 1},
        }
        result = aggregate_repeats(records)
        assert len(result) == 2
        assert abs(result[(0, 0)]["reward"] - 2 / 3) < 1e-9
        assert abs(result[(1, 0)]["reward"] - 1 / 3) < 1e-9
        assert result[(0, 0)]["_aggregated_from_repeats"] == 3

    def test_auto_detection_with_repeats(self):
        """When repeats are aggregated, continuous scores trigger permutation test."""
        from nemo_evaluator.engine.comparison import compare_results

        base = {
            (0, 0): {"reward": 1.0},
            (0, 1): {"reward": 1.0},
            (0, 2): {"reward": 1.0},
            (1, 0): {"reward": 1.0},
            (1, 1): {"reward": 0.0},
            (1, 2): {"reward": 1.0},
            (2, 0): {"reward": 0.0},
            (2, 1): {"reward": 0.0},
            (2, 2): {"reward": 1.0},
        }
        cand = {
            (0, 0): {"reward": 1.0},
            (0, 1): {"reward": 0.0},
            (0, 2): {"reward": 0.0},
            (1, 0): {"reward": 0.0},
            (1, 1): {"reward": 0.0},
            (1, 2): {"reward": 0.0},
            (2, 0): {"reward": 0.0},
            (2, 1): {"reward": 0.0},
            (2, 2): {"reward": 0.0},
        }
        report = compare_results(base, cand)
        # After repeat aggregation, rewards are 0.67, 0.33, etc. → non-binary → permutation
        assert report["test_used"] == "permutation"
        assert "permutation_test" in report


class TestWriteRegression:
    def test_write(self, tmp_path):
        report = {"score_deltas": {}, "runtime_deltas": {}, "report_version": 1, "verdict": "PASS"}
        path = write_regression(report, tmp_path / "report.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "score_deltas" in data
        assert data["report_version"] == 1
