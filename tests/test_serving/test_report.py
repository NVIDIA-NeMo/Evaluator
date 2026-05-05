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

    def test_write_to_file(self, bundle_dir, tmp_path):
        out = tmp_path / "report.md"
        runner = CliRunner()
        result = runner.invoke(report_cmd, [str(bundle_dir), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "gsm8k" in out.read_text()
