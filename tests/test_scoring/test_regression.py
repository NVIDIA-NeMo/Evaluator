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

from nemo_evaluator.engine.comparison import compare_runs, write_regression


def _write_bundle(path: Path, run_id: str, scores: dict, categories=None):
    bundle = {
        "run_id": run_id,
        "config": {"benchmark": "test"},
        "benchmark": {
            "name": "test",
            "samples": 100,
            "scores": scores,
        },
    }
    if categories:
        bundle["benchmark"]["categories"] = categories
    path.write_text(json.dumps(bundle))


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


class TestWriteRegression:
    def test_write(self, tmp_path):
        report = {"score_deltas": {}, "runtime_deltas": {}}
        path = write_regression(report, tmp_path / "report.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "score_deltas" in data
