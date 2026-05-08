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

from nemo_evaluator.engine.sharding import get_shard_range, merge_results, shard_from_env


class TestGetShardRange:
    def test_even_split(self):
        assert get_shard_range(100, 0, 4) == (0, 25)
        assert get_shard_range(100, 1, 4) == (25, 50)
        assert get_shard_range(100, 2, 4) == (50, 75)
        assert get_shard_range(100, 3, 4) == (75, 100)

    def test_uneven_split(self):
        # 10 problems across 3 shards: 4, 3, 3
        assert get_shard_range(10, 0, 3) == (0, 4)
        assert get_shard_range(10, 1, 3) == (4, 7)
        assert get_shard_range(10, 2, 3) == (7, 10)

    def test_coverage(self):
        total = 14001
        shards = 16
        ranges = [get_shard_range(total, i, shards) for i in range(shards)]
        # All problems covered
        all_indices = set()
        for s, e in ranges:
            all_indices.update(range(s, e))
        assert len(all_indices) == total
        # No overlaps
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_total_zero(self):
        assert get_shard_range(0, 0, 4) == (0, 0)

    def test_more_shards_than_problems(self):
        # 3 problems, 10 shards: first 3 get 1 each, rest get empty
        assert get_shard_range(3, 0, 10) == (0, 1)
        assert get_shard_range(3, 2, 10) == (2, 3)
        assert get_shard_range(3, 3, 10) == (3, 3)  # empty
        assert get_shard_range(3, 9, 10) == (3, 3)  # empty

    def test_single_shard(self):
        assert get_shard_range(100, 0, 1) == (0, 100)

    def test_invalid_shard_idx(self):
        with pytest.raises(ValueError):
            get_shard_range(100, -1, 4)
        with pytest.raises(ValueError):
            get_shard_range(100, 4, 4)


class TestShardFromEnv:
    def test_nel_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEL_SHARD_IDX", "3")
        monkeypatch.setenv("NEL_TOTAL_SHARDS", "8")
        assert shard_from_env() == (3, 8)

    def test_slurm_env_vars(self, monkeypatch):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "16")
        assert shard_from_env() == (5, 16)

    def test_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("NEL_SHARD_IDX", raising=False)
        monkeypatch.delenv("NEL_TOTAL_SHARDS", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
        assert shard_from_env() is None

    def test_invalid_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEL_SHARD_IDX", "abc")
        monkeypatch.setenv("NEL_TOTAL_SHARDS", "xyz")
        assert shard_from_env() is None


def _write_shard(
    shard_dir: Path,
    *,
    rewards_by_problem: dict[int, list[float]],
    benchmark: str = "testbench",
    repeats: int = 1,
) -> None:
    """Minimal on-disk shard fixture: ``results.jsonl`` + one ``eval-*.json``."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for pid, rewards in rewards_by_problem.items():
        for rep, reward in enumerate(rewards):
            rows.append(
                {
                    "problem_idx": pid,
                    "repeat": rep,
                    "reward": reward,
                    "metadata": {},
                }
            )
    (shard_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8"
    )
    (shard_dir / f"eval-{benchmark}.json").write_text(
        json.dumps({"config": {"benchmark": benchmark, "repeats": repeats}}), encoding="utf-8"
    )


class TestMergeResults:
    """``merge_results`` has to stay in lockstep with ``run_evaluation``'s
    single-shard output: fractional rewards must emit ``mean_reward`` (not
    ``pass@k``), and binary rewards must emit ``pass@1``.
    """

    def test_fractional_rewards_emit_mean_reward_only(self, tmp_path):
        # Two shards, fractional rubric rewards. With the old behavior this
        # bundle reported a highly inflated pass@1 (any reward > 0 counts as
        # pass) and omitted mean_reward entirely.
        s0 = tmp_path / "shard_0"
        s1 = tmp_path / "shard_1"
        _write_shard(s0, rewards_by_problem={0: [0.7, 0.8], 1: [0.5, 0.6]}, repeats=2)
        _write_shard(s1, rewards_by_problem={2: [0.0, 1.0], 3: [0.3, 0.4]}, repeats=2)

        out = tmp_path / "merged"
        bundle = merge_results([s0, s1], out, n_repeats=2)
        scores = bundle["benchmark"]["scores"]

        assert "mean_reward" in scores
        assert "pass@1" not in scores
        assert "pass@2" not in scores
        # Per-problem means are {0.75, 0.55, 0.5, 0.35} → mean = 0.5375
        assert scores["mean_reward"]["value"] == pytest.approx(0.5375, abs=1e-4)
        assert "ci_lower" in scores["mean_reward"]
        assert "ci_upper" in scores["mean_reward"]
        assert scores["summary"]["n"] == 8

    def test_binary_rewards_still_emit_pass_at_k(self, tmp_path):
        s0 = tmp_path / "shard_0"
        s1 = tmp_path / "shard_1"
        _write_shard(s0, rewards_by_problem={0: [1.0, 0.0], 1: [1.0, 1.0]}, repeats=2)
        _write_shard(s1, rewards_by_problem={2: [0.0, 0.0], 3: [1.0, 0.0]}, repeats=2)

        out = tmp_path / "merged"
        bundle = merge_results([s0, s1], out, n_repeats=2)
        scores = bundle["benchmark"]["scores"]

        assert "mean_reward" not in scores
        assert "pass@1" in scores
        assert "pass@2" in scores
        # pass@1 averaged across problems with 2 attempts each:
        # problem passes = [1/2, 2/2, 0/2, 1/2] → mean = 0.5
        assert scores["pass@1"]["value"] == pytest.approx(0.5, abs=1e-4)

    def test_fractional_merge_matches_single_shard_semantics(self, tmp_path):
        """Single shard of fractional rewards should yield the same
        ``mean_reward`` whether routed through the single-shard path or
        through the merge path (same helper, same formula)."""
        s0 = tmp_path / "shard_0"
        _write_shard(s0, rewards_by_problem={0: [0.2, 0.4], 1: [0.8, 0.6]}, repeats=2)

        bundle = merge_results([s0], tmp_path / "merged", n_repeats=2)
        scores = bundle["benchmark"]["scores"]

        assert "mean_reward" in scores
        assert "pass@1" not in scores
        # Per-problem means: {0.3, 0.7} → mean = 0.5
        assert scores["mean_reward"]["value"] == pytest.approx(0.5, abs=1e-4)

    def test_swebench_style_bundle_shape_unchanged(self, tmp_path):
        """SWE-bench writes strictly ``0`` or ``1`` to ``reward.txt`` (see
        ``harbor/adapters/swebench*/**/test.sh``), so the merged bundle
        must keep emitting ``pass@1`` / ``pass@k`` and must NOT start
        emitting ``mean_reward``.  Guards against this change accidentally
        swapping the SWE-bench headline metric.
        """
        # Mirror a real SWE-bench-verified shape: 5 shards x ~20 problems,
        # 3 repeats each, rewards strictly in {0.0, 1.0}. Exact resolve
        # rate is set by the reward distribution (65 / 100 = 0.65 per
        # shard, same across shards so aggregate pass@1 is 0.65).
        rng_rewards: list[list[float]] = []
        # 65/100 problems fully resolved (all repeats = 1.0), 10 partially
        # resolved (one repeat = 1.0), 25 fully unresolved.
        rng_rewards.extend([[1.0, 1.0, 1.0]] * 65)
        rng_rewards.extend([[1.0, 0.0, 0.0]] * 10)
        rng_rewards.extend([[0.0, 0.0, 0.0]] * 25)

        shard_dirs: list[Path] = []
        for shard_idx in range(5):
            sd = tmp_path / f"shard_{shard_idx}"
            rewards_by_problem = {shard_idx * 100 + i: rng_rewards[i] for i in range(len(rng_rewards))}
            _write_shard(
                sd,
                rewards_by_problem=rewards_by_problem,
                benchmark="swebench-verified_1.0",
                repeats=3,
            )
            shard_dirs.append(sd)

        bundle = merge_results(shard_dirs, tmp_path / "merged", n_repeats=3)
        scores = bundle["benchmark"]["scores"]

        # Headline metrics must match pre-change SWE-bench rendering:
        assert "pass@1" in scores
        assert "pass@3" in scores
        assert "mean_reward" not in scores

        # 500 problems total, pass@1 over binary {0,1} rewards == mean
        # reward across all samples. (65 resolved fully + 10 resolved 1/3)
        # / 100 problems per shard = (65 + 10/3)/100 ≈ 0.6833.
        assert scores["pass@1"]["value"] == pytest.approx(0.6833, abs=1e-3)
        # pass@3 counts a problem as passing if any repeat resolved:
        # 75/100 problems have at least one resolve → 0.75.
        assert scores["pass@3"]["value"] == pytest.approx(0.75, abs=1e-3)
        # Summary is binary-shaped (min=0, max=1, median=1.0 here).
        assert scores["summary"]["min"] == 0.0
        assert scores["summary"]["max"] == 1.0
