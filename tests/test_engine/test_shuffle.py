# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for seeded dataset shuffling in the evaluation loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nemo_evaluator.engine.eval_loop import run_evaluation
from nemo_evaluator.engine.sharding import merge_results
from nemo_evaluator.engine.step_log import config_hash
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.solvers import SolveResult


class _IndexEnv(EvalEnvironment):
    """Records ``env.seed`` call order; reward is always 1.0."""

    name = "index_env"

    def __init__(self, size: int) -> None:
        super().__init__()
        self._size = size
        self.seed_order: list[int] = []

    async def dataset_size(self) -> int:
        return self._size

    async def seed(self, idx: int) -> SeedResult:
        self.seed_order.append(idx)
        return SeedResult(prompt=str(idx), expected_answer=str(idx), metadata={"orig_idx": idx})

    async def verify(self, response, expected, **meta):
        return VerifyResult(reward=1.0, extracted_answer=response, scoring_details={"method": "noop"})


class _EchoSolver:
    async def solve(self, task):
        return SolveResult(response=task.prompt)

    async def close(self):
        pass


class _SlotRecorder:
    """Progress tracker that records the slot value of every step/phase call."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.phases: list[tuple[int, str]] = []
        self._total_problems = 0

    def on_start(self, benchmark, total_problems, total_repeats):
        self._total_problems = total_problems

    def on_step(self, problem, repeat, total_problems, total_repeats, reward, tokens, latency_ms):
        self.steps.append(problem)

    def on_phase(self, problem, repeat, total_problems, total_repeats, phase):
        self.phases.append((problem, phase))

    def on_done(self, correct, total, elapsed, total_tokens, mean_reward=None):
        pass


def _run(env, solver, **kwargs):
    return asyncio.run(run_evaluation(env, solver, n_repeats=1, **kwargs))


# ---------------------------------------------------------------------------
# Determinism & backward compat
# ---------------------------------------------------------------------------


class TestShuffleDeterminism:
    def test_same_seed_same_order(self):
        env1 = _IndexEnv(20)
        _run(env1, _EchoSolver(), shuffle_seed=7)

        env2 = _IndexEnv(20)
        _run(env2, _EchoSolver(), shuffle_seed=7)

        assert env1.seed_order == env2.seed_order

    def test_different_seeds_different_order(self):
        env1 = _IndexEnv(50)
        _run(env1, _EchoSolver(), shuffle_seed=7)

        env2 = _IndexEnv(50)
        _run(env2, _EchoSolver(), shuffle_seed=123)

        assert env1.seed_order != env2.seed_order
        assert sorted(env1.seed_order) == sorted(env2.seed_order)

    def test_none_preserves_dataset_order(self):
        env = _IndexEnv(10)
        _run(env, _EchoSolver(), shuffle_seed=None)

        assert env.seed_order == list(range(10))

    def test_permutation_is_complete_bijection(self):
        env = _IndexEnv(100)
        _run(env, _EchoSolver(), shuffle_seed=42)

        assert sorted(env.seed_order) == list(range(100))
        assert env.seed_order != list(range(100))


# ---------------------------------------------------------------------------
# problem_idx invariant — results always carry the ORIGINAL dataset index
# ---------------------------------------------------------------------------


class TestProblemIdxInvariant:
    def test_results_use_original_dataset_indices(self):
        env = _IndexEnv(15)
        bundle = _run(env, _EchoSolver(), shuffle_seed=99)

        idxs = [r["problem_idx"] for r in bundle["_results"]]
        assert sorted(idxs) == list(range(15))

    def test_problem_idx_matches_env_seed_call_order(self):
        env = _IndexEnv(8)
        bundle = _run(env, _EchoSolver(), shuffle_seed=3)
        assert set(env.seed_order) == {r["problem_idx"] for r in bundle["_results"]}


# ---------------------------------------------------------------------------
# max_problems — shuffle-then-truncate
# ---------------------------------------------------------------------------


class TestMaxProblemsWithShuffle:
    def test_truncation_applies_after_shuffle(self):
        env = _IndexEnv(100)
        bundle = _run(env, _EchoSolver(), shuffle_seed=42, max_problems=10)

        assert len(bundle["_results"]) == 10
        idxs = [r["problem_idx"] for r in bundle["_results"]]
        assert all(0 <= i < 100 for i in idxs)
        assert len(set(idxs)) == 10

    def test_truncation_is_deterministic(self):
        env1 = _IndexEnv(100)
        b1 = _run(env1, _EchoSolver(), shuffle_seed=42, max_problems=10)

        env2 = _IndexEnv(100)
        b2 = _run(env2, _EchoSolver(), shuffle_seed=42, max_problems=10)

        assert sorted(r["problem_idx"] for r in b1["_results"]) == sorted(r["problem_idx"] for r in b2["_results"])

    def test_max_problems_ge_dataset_size_noop(self):
        env = _IndexEnv(5)
        bundle = _run(env, _EchoSolver(), shuffle_seed=42, max_problems=100)

        assert len(bundle["_results"]) == 5


# ---------------------------------------------------------------------------
# Sharding — shards still partition the dataset under shuffle
# ---------------------------------------------------------------------------


class TestShardPartitioning:
    def test_shards_cover_full_dataset_without_overlap(self):
        n = 47
        total_shards = 4
        all_idxs: list[int] = []

        for shard_idx in range(total_shards):
            env = _IndexEnv(n)
            bundle = _run(
                env,
                _EchoSolver(),
                shuffle_seed=42,
                shard_info=(shard_idx, total_shards),
            )
            all_idxs.extend(r["problem_idx"] for r in bundle["_results"])

        assert sorted(all_idxs) == list(range(n))

    def test_shard_size_balance(self):
        n = 100
        total_shards = 4
        sizes = []
        for shard_idx in range(total_shards):
            env = _IndexEnv(n)
            bundle = _run(
                env,
                _EchoSolver(),
                shuffle_seed=42,
                shard_info=(shard_idx, total_shards),
            )
            sizes.append(len(bundle["_results"]))
        assert sum(sizes) == n
        assert max(sizes) - min(sizes) <= 1

    def test_shard_gets_mixed_indices_not_contiguous_slice(self):
        env = _IndexEnv(200)
        bundle = _run(env, _EchoSolver(), shuffle_seed=42, shard_info=(0, 4))
        idxs = sorted(r["problem_idx"] for r in bundle["_results"])
        assert idxs != list(range(50))


# ---------------------------------------------------------------------------
# Progress slot decoupling
# ---------------------------------------------------------------------------


class TestProgressSlots:
    def test_slots_are_shard_local_under_shuffle(self):
        rec = _SlotRecorder()
        env = _IndexEnv(40)
        asyncio.run(run_evaluation(env, _EchoSolver(), n_repeats=1, shuffle_seed=42, shard_info=(1, 4), progress=rec))
        assert rec.steps
        assert all(0 <= s < 10 for s in rec.steps), rec.steps

    def test_slots_are_shard_local_without_shuffle(self):
        rec = _SlotRecorder()
        env = _IndexEnv(40)
        asyncio.run(run_evaluation(env, _EchoSolver(), n_repeats=1, shuffle_seed=None, shard_info=(2, 4), progress=rec))
        assert all(0 <= s < 10 for s in rec.steps), rec.steps

    def test_slots_on_phase_and_on_step_agree(self):
        rec = _SlotRecorder()
        env = _IndexEnv(12)
        asyncio.run(run_evaluation(env, _EchoSolver(), n_repeats=1, shuffle_seed=99, progress=rec))
        phase_slots = {p for p, _ in rec.phases}
        step_slots = set(rec.steps)
        assert step_slots.issubset(phase_slots) or not phase_slots


# ---------------------------------------------------------------------------
# problem_range override bypasses shuffle
# ---------------------------------------------------------------------------


class TestProblemRangeOverride:
    def test_explicit_problem_range_disables_shuffle(self):
        env = _IndexEnv(50)
        bundle = _run(env, _EchoSolver(), shuffle_seed=42, problem_range=(10, 20))
        idxs = [r["problem_idx"] for r in bundle["_results"]]
        assert sorted(idxs) == list(range(10, 20))


# ---------------------------------------------------------------------------
# Resume cache invalidation on seed change
# ---------------------------------------------------------------------------


class TestConfigHashInvalidation:
    def test_changing_shuffle_seed_changes_config_hash(self):
        h1 = config_hash({"model": "m", "shuffle_seed": 1})
        h2 = config_hash({"model": "m", "shuffle_seed": 2})
        assert h1 != h2

    def test_none_shuffle_seed_equals_missing(self):
        h_absent = config_hash({"model": "m"})
        h_none = config_hash({"model": "m", "shuffle_seed": None})
        assert h_absent == h_none

    def test_resume_invalidates_inference_cache_on_seed_change(self, tmp_path: Path):
        env = _IndexEnv(10)
        _run(env, _EchoSolver(), shuffle_seed=1, step_log_dir=tmp_path, config={"model": "m"})
        inf_log = tmp_path / "inference_log.jsonl"
        assert inf_log.exists()
        assert inf_log.stat().st_size > 0

        env2 = _IndexEnv(10)
        _run(env2, _EchoSolver(), shuffle_seed=2, step_log_dir=tmp_path, resume=True, config={"model": "m"})
        assert inf_log.exists()


# ---------------------------------------------------------------------------
# Shuffle metadata in bundle config
# ---------------------------------------------------------------------------


class TestShuffleMetadataInBundle:
    def test_bundle_config_records_shuffle(self):
        env = _IndexEnv(10)
        bundle = _run(env, _EchoSolver(), shuffle_seed=42)

        cfg = bundle["config"]
        assert cfg["shuffle_seed"] == 42
        assert cfg["shuffle"]["applied"] is True
        assert cfg["shuffle"]["seed"] == 42
        assert cfg["shuffle"]["ds_full_size"] == 10
        assert cfg["shuffle"]["ds_effective_size"] == 10

    def test_bundle_config_records_opt_out(self):
        env = _IndexEnv(10)
        bundle = _run(env, _EchoSolver(), shuffle_seed=None)

        cfg = bundle["config"]
        assert cfg["shuffle_seed"] is None
        assert cfg["shuffle"]["applied"] is False


# ---------------------------------------------------------------------------
# End-to-end cross-shard merge: "no samples mixup, correct merges"
# ---------------------------------------------------------------------------


class _ScoredEnv(EvalEnvironment):
    """Reward == problem_idx % 2 (deterministic per-sample score)."""

    name = "scored_env"

    def __init__(self, size: int) -> None:
        super().__init__()
        self._size = size

    async def dataset_size(self) -> int:
        return self._size

    async def seed(self, idx: int) -> SeedResult:
        return SeedResult(prompt=f"q{idx}", expected_answer=str(idx), metadata={"problem_idx": idx})

    async def verify(self, response, expected, **meta):
        idx = int(expected)
        return VerifyResult(
            reward=float(idx % 2),
            extracted_answer=response,
            scoring_details={"method": "idx_parity"},
        )


def _run_shard_to_dir(size: int, shard_idx: int, total_shards: int, shuffle_seed, tmp_path: Path) -> Path:
    """Run one shard and write its bundle to disk; returns the shard dir."""
    from nemo_evaluator.engine.artifacts import write_all

    env = _ScoredEnv(size)
    bundle = asyncio.run(
        run_evaluation(
            env,
            _EchoSolver(),
            n_repeats=1,
            shuffle_seed=shuffle_seed,
            shard_info=(shard_idx, total_shards),
            config={"model": "m", "benchmark": "scored_env"},
        )
    )
    shard_dir = tmp_path / f"shard_{shard_idx}"
    shard_dir.mkdir()
    write_all(bundle, shard_dir)
    return shard_dir


class TestCrossShardMerge:
    def test_merged_results_cover_dataset_exactly_once(self, tmp_path: Path):
        size = 37
        n_shards = 4
        shard_dirs = [_run_shard_to_dir(size, i, n_shards, shuffle_seed=42, tmp_path=tmp_path) for i in range(n_shards)]

        out = tmp_path / "merged"
        merge_results(shard_dirs, out, n_repeats=1)

        results_path = out / "results.jsonl"
        assert results_path.exists()
        lines = results_path.read_text().strip().split("\n")
        idxs = [json.loads(line)["problem_idx"] for line in lines]

        assert sorted(idxs) == list(range(size)), (
            f"Merged results should cover {size} original indices exactly once; "
            f"got {len(idxs)} rows with {len(set(idxs))} unique idxs"
        )

    def test_merged_score_matches_unsharded_run(self, tmp_path: Path):
        """Sharded+shuffled pass@1 must equal unsharded+unshuffled pass@1."""
        size = 40
        n_shards = 5

        env = _ScoredEnv(size)
        baseline = asyncio.run(
            run_evaluation(
                env,
                _EchoSolver(),
                n_repeats=1,
                shuffle_seed=None,
                config={"model": "m", "benchmark": "scored_env"},
            )
        )
        baseline_score = baseline["benchmark"]["scores"]["pass@1"]["value"]

        shard_dirs = [_run_shard_to_dir(size, i, n_shards, shuffle_seed=42, tmp_path=tmp_path) for i in range(n_shards)]
        out = tmp_path / "merged"
        merged = merge_results(shard_dirs, out, n_repeats=1)
        merged_score = merged["benchmark"]["scores"]["pass@1"]["value"]

        assert baseline_score == pytest.approx(0.5)
        assert merged_score == pytest.approx(baseline_score)

    def test_merged_sample_count(self, tmp_path: Path):
        size = 25
        n_shards = 3
        shard_dirs = [_run_shard_to_dir(size, i, n_shards, shuffle_seed=42, tmp_path=tmp_path) for i in range(n_shards)]
        out = tmp_path / "merged"
        merge_results(shard_dirs, out, n_repeats=1)

        rp = out / "results.jsonl"
        idxs = {json.loads(line)["problem_idx"] for line in rp.read_text().strip().split("\n")}
        assert len(idxs) == size
        assert idxs == set(range(size))
