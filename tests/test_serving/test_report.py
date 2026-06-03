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
"""Tests for nel report command."""

import json

import pytest
from click.testing import CliRunner

from nemo_evaluator.cli.report import report_cmd
from nemo_evaluator.reports.eval import _primary_metric


@pytest.fixture
def bundle_dir(tmp_path):
    b1 = {
        "run_id": "eval-gsm8k",
        "config": {"model": "test-model"},
        "benchmark": {
            "name": "gsm8k",
            "samples": 100,
            "repeats": 4,
            "scores": {"pass@1": {"value": 0.85, "ci_lower": 0.82, "ci_upper": 0.88}},
            "categories": {"arithmetic": {"mean_reward": 0.9, "n": 50}, "algebra": {"mean_reward": 0.8, "n": 50}},
        },
    }
    b2 = {
        "run_id": "eval-triviaqa",
        "config": {"model": "test-model"},
        "benchmark": {
            "name": "triviaqa",
            "samples": 500,
            "repeats": 1,
            "scores": {"pass@1": {"value": 0.72, "ci_lower": 0.69, "ci_upper": 0.75}},
        },
    }
    (tmp_path / "eval-gsm8k.json").write_text(json.dumps(b1))
    (tmp_path / "eval-triviaqa.json").write_text(json.dumps(b2))
    return tmp_path


class TestReportCommand:
    def test_markdown_output(self, bundle_dir):
        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(bundle_dir)])
        assert result.exit_code == 0
        assert "gsm8k" in result.output
        assert "triviaqa" in result.output
        assert "0.8500" in result.output

    def test_csv_output(self, bundle_dir):
        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(bundle_dir), "-f", "csv"])
        assert result.exit_code == 0
        assert "benchmark,samples" in result.output

    def test_json_output(self, bundle_dir):
        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(bundle_dir), "-f", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["n_benchmarks"] == 2

    def test_run_directory_discovers_benchmark_subdirectories(self, tmp_path):
        run_dir = tmp_path / "run"
        bench_dir = run_dir / "gsm8k"
        bench_dir.mkdir(parents=True)
        bundle = {
            "run_id": "eval-gsm8k",
            "config": {"model": "test-model"},
            "benchmark": {
                "name": "gsm8k",
                "samples": 100,
                "repeats": 1,
                "scores": {"pass@1": {"value": 0.85}},
            },
        }
        (bench_dir / "eval-gsm8k.json").write_text(json.dumps(bundle))

        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(run_dir)])

        assert result.exit_code == 0
        assert "gsm8k" in result.output
        assert "0.8500" in result.output

    def test_write_to_file(self, bundle_dir, tmp_path):
        out = tmp_path / "report.md"
        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(bundle_dir), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "gsm8k" in out.read_text()


class TestPrimaryMetricFallback:
    """``_primary_metric`` falls back to the first scored metric when no
    canonical ``mean_reward``/``pass@1`` is present (legacy container
    bundles surface harness-specific metric names)."""

    # nemo-skills math containers emit BOTH a quality metric (symbolic_correct)
    # and a timing metric (gen_seconds) under the same pass@1 group.  Insertion
    # order puts gen_seconds first, so the naïve "first scored metric" fallback
    # would surface latency as the headline — actively misleading.  Fallback
    # must skip timing-flavored leaves.
    @pytest.mark.parametrize(
        "row, expected",
        [
            pytest.param(
                {
                    "samples": 3,
                    "repeats": 1,
                    "ifeval/inst_level_loose_acc": {"value": 0.6},
                    "ifeval/prompt_level_strict_acc": {"value": 0.3333},
                },
                ("ifeval/inst_level_loose_acc", 0.6),
                id="legacy_names",
            ),
            pytest.param(
                {
                    "samples": 3,
                    "repeats": 1,
                    "categories": {"x": "y"},
                    "total_tokens": 100,
                    "latency_p50_ms": 50,
                    "scorer:foo": {"value": 0.5},
                    "real_metric": {"value": 0.9},
                },
                ("real_metric", 0.9),
                id="skip_non_metric_keys",
            ),
            pytest.param(
                {
                    "samples": 3,
                    "repeats": 1,
                    "hmmt_feb25/pass@1/gen_seconds": {"value": 140.0},
                    "hmmt_feb25/pass@1/symbolic_correct": {"value": 100.0},
                },
                ("hmmt_feb25/pass@1/symbolic_correct", 100.0),
                id="skip_timing_flavored",
            ),
            pytest.param(
                {
                    "samples": 3,
                    "repeats": 1,
                    "hmmt_feb25/pass@1/gen_seconds": {"value": 140.0},
                    "hmmt_feb25/pass@1/latency_ms": {"value": 2500.0},
                    "hmmt_feb25/pass@1/output_tokens": {"value": 512.0},
                },
                ("", None),
                id="timing_only",
            ),
        ],
    )
    def test_primary_metric(self, row, expected):
        expected_name, expected_value = expected
        name, m = _primary_metric(row)
        assert name == expected_name
        if expected_value is None:
            assert m == {}
        else:
            assert m["value"] == expected_value

    def test_legacy_bundle_renders_score_in_markdown(self, tmp_path):
        """End-to-end: a bundle whose only scores are ifeval-style
        metric names renders a numeric headline in markdown output."""
        bundle = {
            "run_id": "eval-container-ifeval",
            "config": {"model": "test"},
            "benchmark": {
                "name": "container/ifeval",
                "samples": 3,
                "repeats": 1,
                "scores": {
                    "ifeval/inst_level_loose_acc/inst_level_loose_acc": {"value": 0.6},
                    "ifeval/prompt_level_strict_acc/prompt_level_strict_acc": {"value": 0.3333},
                },
            },
        }
        (tmp_path / "eval-container-ifeval.json").write_text(json.dumps(bundle))
        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(tmp_path)])
        assert result.exit_code == 0
        assert "container/ifeval" in result.output
        # The fallback should surface 0.6000 (the first metric); without
        # the fallback this row would render "| ... | - | 3 | - | |".
        assert "0.6000" in result.output
