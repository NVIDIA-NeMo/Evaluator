"""Tests for checkpointing and resume."""
import pytest
from nemo_evaluator.runner.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_mark_completed_and_skip(self, tmp_path):
        ckpt = CheckpointManager(tmp_path)
        assert not ckpt.is_completed("gsm8k")

        ckpt.mark_completed("gsm8k", "/results/gsm8k")
        assert ckpt.is_completed("gsm8k")
        assert ckpt.summary["completed"] == 1

    def test_pending_excludes_completed(self, tmp_path):
        ckpt = CheckpointManager(tmp_path)
        ckpt.mark_completed("gsm8k", "/results/gsm8k")
        pending = ckpt.pending_benchmarks(["gsm8k", "triviaqa", "bench_c"])
        assert pending == ["triviaqa", "bench_c"]

    def test_survives_restart(self, tmp_path):
        ckpt1 = CheckpointManager(tmp_path)
        ckpt1.mark_completed("gsm8k", "/results/gsm8k")

        ckpt2 = CheckpointManager(tmp_path)
        assert ckpt2.is_completed("gsm8k")

    def test_failed_does_not_count_as_completed(self, tmp_path):
        ckpt = CheckpointManager(tmp_path)
        ckpt.mark_failed("mmlu", "timeout")
        assert not ckpt.is_completed("mmlu")
        assert ckpt.summary["failed"] == 1

    def test_clear_resets_state(self, tmp_path):
        ckpt = CheckpointManager(tmp_path)
        ckpt.mark_completed("gsm8k", "/p")
        ckpt.clear()
        assert not ckpt.is_completed("gsm8k")
