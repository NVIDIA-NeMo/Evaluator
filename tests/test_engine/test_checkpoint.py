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
"""Tests for checkpointing and resume."""

from nemo_evaluator.engine.checkpoint import CheckpointManager


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
