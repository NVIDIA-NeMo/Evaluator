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
"""Tests for nel compare CLI, including multi-benchmark support."""

import json

from click.testing import CliRunner

from nemo_evaluator.cli.regression import compare_cmd


def _write_bundle(path, name, score, metric="mean_reward"):
    bundle = {
        "run_id": f"run-{name}",
        "config": {"model": "test-model"},
        "benchmark": {
            "name": name,
            "samples": 20,
            "scores": {metric: {"value": score}},
        },
    }
    path.write_text(json.dumps(bundle))


def _write_results(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _record(pid, reward):
    return {"problem_idx": pid, "repeat": 0, "reward": reward, "expected_answer": "ans"}


def _make_single_benchmark(tmp_path, benchmark, base_reward, cand_reward, n=20):
    """Create baseline/<bench>/eval-*.json and candidate/<bench>/eval-*.json."""
    baseline = tmp_path / "baseline" / benchmark
    candidate = tmp_path / "candidate" / benchmark
    baseline.mkdir(parents=True)
    candidate.mkdir(parents=True)

    _write_bundle(baseline / f"eval-{benchmark}.json", benchmark, base_reward)
    _write_bundle(candidate / f"eval-{benchmark}.json", benchmark, cand_reward)
    _write_results(baseline / "results.jsonl", [_record(i, base_reward) for i in range(n)])
    _write_results(candidate / "results.jsonl", [_record(i, cand_reward) for i in range(n)])
    return tmp_path / "baseline", tmp_path / "candidate"


def _make_regression_benchmark(tmp_path, name, n=20):
    """Create a benchmark where all baseline items are correct and all candidate items are wrong."""
    baseline = tmp_path / "baseline" / name
    candidate = tmp_path / "candidate" / name
    baseline.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=True, exist_ok=True)
    _write_bundle(baseline / f"eval-{name}.json", name, 1.0)
    _write_bundle(candidate / f"eval-{name}.json", name, 0.0)
    _write_results(baseline / "results.jsonl", [_record(i, 1.0) for i in range(n)])
    _write_results(candidate / "results.jsonl", [_record(i, 0.0) for i in range(n)])


def _make_multi_benchmark(tmp_path, benchmarks: dict[str, tuple[float, float]], n=20):
    """Create a multi-benchmark directory pair.

    benchmarks: {name: (base_reward, cand_reward)}
    """
    for name, (base_reward, cand_reward) in benchmarks.items():
        _make_single_benchmark(tmp_path, name, base_reward, cand_reward, n=n)
    return tmp_path / "baseline", tmp_path / "candidate"


class TestSingleBenchmarkCompare:
    def test_text_output(self, tmp_path):
        baseline, candidate = _make_single_benchmark(tmp_path, "gsm8k", 0.8, 0.8)
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
            ],
        )
        assert result.exit_code == 0
        assert "gsm8k" in result.output.lower() or "PASS" in result.output

    def test_json_output(self, tmp_path):
        baseline, candidate = _make_single_benchmark(tmp_path, "gsm8k", 0.8, 0.8)
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "verdict" in data


class TestMultiBenchmarkCompare:
    def test_compares_all_matched_benchmarks(self, tmp_path):
        baseline, candidate = _make_multi_benchmark(
            tmp_path,
            {
                "gsm8k": (0.8, 0.8),
                "mmlu": (0.9, 0.9),
            },
        )
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
            ],
        )
        assert result.exit_code == 0
        assert "gsm8k" in result.output
        assert "mmlu" in result.output

    def test_json_keyed_by_benchmark(self, tmp_path):
        baseline, candidate = _make_multi_benchmark(
            tmp_path,
            {
                "gsm8k": (0.8, 0.8),
                "mmlu": (0.9, 0.9),
            },
        )
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "gsm8k" in data
        assert "mmlu" in data
        assert "verdict" in data["gsm8k"]
        assert "verdict" in data["mmlu"]

    def test_warns_on_unmatched_benchmarks(self, tmp_path):
        _make_multi_benchmark(
            tmp_path,
            {
                "gsm8k": (0.8, 0.8),
                "mmlu": (0.9, 0.9),
            },
        )
        extra = tmp_path / "candidate" / "gpqa"
        extra.mkdir(parents=True)
        _write_bundle(extra / "eval-gpqa.json", "gpqa", 0.7)
        _write_results(extra / "results.jsonl", [_record(i, 0.7) for i in range(20)])

        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(tmp_path / "baseline"),
                str(tmp_path / "candidate"),
                "--no-strict",
                "--no-report",
            ],
        )
        assert result.exit_code == 0
        # "gpqa" warning goes to stderr; in CliRunner default mode it's mixed into output
        assert "gpqa" in result.output

    def test_worst_verdict_propagated(self, tmp_path):
        # All baseline items correct (1.0), all candidate mmlu items wrong (0.0)
        # -> every sample is a regression flip -> BLOCK
        _make_multi_benchmark(tmp_path, {"gsm8k": (1.0, 1.0)})
        _make_regression_benchmark(tmp_path, "mmlu", n=20)

        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(tmp_path / "baseline"),
                str(tmp_path / "candidate"),
                "--no-strict",
                "--no-report",
            ],
        )
        assert result.exit_code == 0
        assert "BLOCK" in result.output

    def test_strict_exits_nonzero_on_block(self, tmp_path):
        _make_multi_benchmark(tmp_path, {"gsm8k": (1.0, 1.0)})
        _make_regression_benchmark(tmp_path, "mmlu", n=20)

        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(tmp_path / "baseline"),
                str(tmp_path / "candidate"),
                "--strict",
                "--no-report",
            ],
        )
        assert result.exit_code == 1

    def test_output_writes_json(self, tmp_path):
        baseline, candidate = _make_multi_benchmark(
            tmp_path,
            {
                "gsm8k": (0.8, 0.8),
                "mmlu": (0.9, 0.9),
            },
        )
        out_file = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
                "-o",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "gsm8k" in data
        assert "mmlu" in data

    def test_no_match_fails(self, tmp_path):
        # 2 benchmarks on each side, none overlapping -> no matches
        for name in ("gsm8k", "gpqa"):
            d = tmp_path / "baseline" / name
            d.mkdir(parents=True)
            _write_bundle(d / f"eval-{name}.json", name, 0.8)
            _write_results(d / "results.jsonl", [_record(i, 0.8) for i in range(20)])
        for name in ("mmlu", "simpleqa"):
            d = tmp_path / "candidate" / name
            d.mkdir(parents=True)
            _write_bundle(d / f"eval-{name}.json", name, 0.9)
            _write_results(d / "results.jsonl", [_record(i, 0.9) for i in range(20)])

        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(tmp_path / "baseline"),
                str(tmp_path / "candidate"),
                "--no-strict",
                "--no-report",
            ],
        )
        assert result.exit_code != 0
        assert "No matching benchmarks" in result.output

    def test_compact_mode(self, tmp_path):
        baseline, candidate = _make_multi_benchmark(
            tmp_path,
            {
                "gsm8k": (0.8, 0.8),
                "mmlu": (0.9, 0.9),
            },
        )
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
                "--compact",
            ],
        )
        assert result.exit_code == 0
        assert "gsm8k" in result.output
        assert "mmlu" in result.output

    def test_markdown_format(self, tmp_path):
        baseline, candidate = _make_multi_benchmark(
            tmp_path,
            {
                "gsm8k": (0.8, 0.8),
                "mmlu": (0.9, 0.9),
            },
        )
        runner = CliRunner()
        result = runner.invoke(
            compare_cmd,
            [
                str(baseline),
                str(candidate),
                "--no-strict",
                "--no-report",
                "--format",
                "markdown",
            ],
        )
        assert result.exit_code == 0
        assert "gsm8k" in result.output
        assert "mmlu" in result.output
