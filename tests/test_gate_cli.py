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

from click.testing import CliRunner

from nemo_evaluator.cli.gate import gate_cmd


def _write_bundle(path, name, score):
    bundle = {
        "run_id": f"run-{name}",
        "config": {"model": "test-model"},
        "benchmark": {
            "name": name,
            "samples": 100,
            "scores": {"mean_reward": {"value": score}},
        },
    }
    path.write_text(json.dumps(bundle))


def _write_results(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _record(pid, reward):
    return {"problem_idx": pid, "repeat": 0, "reward": reward, "expected_answer": "ans"}


def _make_results_dirs(tmp_path, benchmark, base_reward, cand_reward, n=20):
    baseline = tmp_path / "baseline" / benchmark
    candidate = tmp_path / "candidate" / benchmark
    baseline.mkdir(parents=True)
    candidate.mkdir(parents=True)

    _write_bundle(baseline / f"eval-{benchmark}.json", benchmark, base_reward)
    _write_bundle(candidate / f"eval-{benchmark}.json", benchmark, cand_reward)
    _write_results(
        baseline / "results.jsonl",
        [_record(i, base_reward) for i in range(n)],
    )
    _write_results(
        candidate / "results.jsonl",
        [_record(i, cand_reward) for i in range(n)],
    )
    return tmp_path / "baseline", tmp_path / "candidate"


def _write_policy(path, max_drop=0.01):
    path.write_text(
        "version: 1\n"
        "defaults:\n"
        "  tier: critical\n"
        "  metric: mean_reward\n"
        "  max_drop: 0.015\n"
        "benchmarks:\n"
        "  mmlu:\n"
        "    tier: critical\n"
        f"    max_drop: {max_drop}\n",
        encoding="utf-8",
    )


class TestGateCommand:
    def test_text_output(self, tmp_path):
        baseline, candidate = _make_results_dirs(tmp_path, "mmlu", 0.80, 0.795)
        policy = tmp_path / "policy.yaml"
        _write_policy(policy)

        runner = CliRunner()
        result = runner.invoke(gate_cmd, [str(baseline), str(candidate), "--policy", str(policy)])

        assert result.exit_code == 0
        assert "GO" in result.output
        assert "mmlu" in result.output
        assert "PASS" in result.output

    def test_json_output(self, tmp_path):
        baseline, candidate = _make_results_dirs(tmp_path, "mmlu", 0.80, 0.795)
        policy = tmp_path / "policy.yaml"
        _write_policy(policy)

        runner = CliRunner()
        result = runner.invoke(
            gate_cmd,
            [str(baseline), str(candidate), "--policy", str(policy), "--format", "json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["verdict"] == "GO"
        assert data["benchmarks"][0]["benchmark"] == "mmlu"

    def test_output_file_written(self, tmp_path):
        baseline, candidate = _make_results_dirs(tmp_path, "mmlu", 0.80, 0.795)
        policy = tmp_path / "policy.yaml"
        _write_policy(policy)
        output = tmp_path / "gate.json"

        runner = CliRunner()
        result = runner.invoke(
            gate_cmd,
            [str(baseline), str(candidate), "--policy", str(policy), "--output", str(output)],
        )

        assert result.exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["verdict"] == "GO"

    def test_strict_nogo_exits_one(self, tmp_path):
        baseline, candidate = _make_results_dirs(tmp_path, "mmlu", 0.80, 0.70)
        policy = tmp_path / "policy.yaml"
        _write_policy(policy, max_drop=0.01)

        runner = CliRunner()
        result = runner.invoke(
            gate_cmd,
            [str(baseline), str(candidate), "--policy", str(policy), "--strict"],
        )

        assert result.exit_code == 1
        assert "NO-GO" in result.output

    def test_strict_inconclusive_exits_two(self, tmp_path):
        baseline, candidate = _make_results_dirs(tmp_path, "mmlu", 0.80, 0.79, n=5)
        policy = tmp_path / "policy.yaml"
        _write_policy(policy)

        runner = CliRunner()
        result = runner.invoke(
            gate_cmd,
            [str(baseline), str(candidate), "--policy", str(policy), "--strict"],
        )

        assert result.exit_code == 2
        assert "INCONCLUSIVE" in result.output
