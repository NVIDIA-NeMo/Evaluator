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
"""Tests for StepLog trajectory overflow sidecars and capture markers (FEP-1295).

Before the fix, StepLog.append() rewrote any record whose trajectory JSON
exceeded the cap as ``{"trajectory": None, "_truncated": True}`` — silently
dropping the only copy of the trajectory a resumed run could recover.
"""

import logging

from nemo_evaluator.engine.step_log import (
    CAPTURE_MARKER_DIR,
    INFERENCE_LOG,
    TRAJECTORY_OVERFLOW_DIR,
    StepLog,
    has_capture_marker,
    reset_checkpoint_sidecars,
    write_capture_marker,
)

LOGGER_NAME = "nemo_evaluator.engine.step_log"

BIG_TRAJECTORY = [{"step": 1, "action": "think", "output": "x" * 4096}]


def _record(problem_idx=0, repeat=0, **extra):
    return {"problem_idx": problem_idx, "repeat": repeat, "response": "ok", **extra}


def _open_log(tmp_path, **kwargs) -> StepLog:
    log = StepLog(tmp_path / INFERENCE_LOG, **kwargs)
    log.open(truncate=True)
    return log


class TestTrajectoryOverflow:
    async def test_small_trajectory_kept_inline(self, tmp_path):
        log = _open_log(tmp_path)
        await log.append(_record(trajectory=[{"step": 1}]))
        log.close()

        rec = log.load()[(0, 0)]
        assert rec["trajectory"] == [{"step": 1}]
        assert "trajectory_ref" not in rec
        assert not (tmp_path / TRAJECTORY_OVERFLOW_DIR).exists()

    async def test_oversized_trajectory_spilled_to_sidecar(self, tmp_path, caplog):
        log = _open_log(tmp_path, max_trajectory_bytes=100)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            await log.append(_record(problem_idx=2, repeat=1, trajectory=BIG_TRAJECTORY))
        log.close()

        rec = log.load()[(2, 1)]
        assert rec["trajectory"] is None
        assert rec["trajectory_ref"] == f"{TRAJECTORY_OVERFLOW_DIR}/p2_r1.json.gz"
        assert rec["trajectory_bytes"] > 100
        assert (tmp_path / TRAJECTORY_OVERFLOW_DIR / "p2_r1.json.gz").is_file()
        assert log.resolve_trajectory(rec) == BIG_TRAJECTORY
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    async def test_env_var_cap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEL_MAX_TRAJECTORY_BYTES", "100")
        log = _open_log(tmp_path)
        await log.append(_record(trajectory=BIG_TRAJECTORY))
        log.close()

        rec = log.load()[(0, 0)]
        assert rec["trajectory"] is None
        assert rec["trajectory_ref"] == f"{TRAJECTORY_OVERFLOW_DIR}/p0_r0.json.gz"

    async def test_ctor_cap_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEL_MAX_TRAJECTORY_BYTES", "100")
        log = _open_log(tmp_path, max_trajectory_bytes=10_000_000)
        await log.append(_record(trajectory=BIG_TRAJECTORY))
        log.close()

        rec = log.load()[(0, 0)]
        assert rec["trajectory"] == BIG_TRAJECTORY
        assert "trajectory_ref" not in rec

    async def test_spill_failure_falls_back_to_truncation(self, tmp_path, caplog):
        (tmp_path / TRAJECTORY_OVERFLOW_DIR).write_text("not a directory")
        log = _open_log(tmp_path, max_trajectory_bytes=100)
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            await log.append(_record(trajectory=BIG_TRAJECTORY))
        log.close()

        rec = log.load()[(0, 0)]
        assert rec["trajectory"] is None
        assert rec["_truncated"] is True
        assert "trajectory_ref" not in rec
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_resolve_legacy_truncated_warns_and_returns_none(self, tmp_path, caplog):
        log = StepLog(tmp_path / INFERENCE_LOG)
        rec = _record(trajectory=None, _truncated=True)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert log.resolve_trajectory(rec) is None
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_resolve_missing_sidecar_warns_and_returns_none(self, tmp_path, caplog):
        log = StepLog(tmp_path / INFERENCE_LOG)
        rec = _record(trajectory=None, trajectory_ref=f"{TRAJECTORY_OVERFLOW_DIR}/p9_r9.json.gz")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert log.resolve_trajectory(rec) is None
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_resolve_inline_trajectory_passthrough(self, tmp_path):
        log = StepLog(tmp_path / INFERENCE_LOG)
        assert log.resolve_trajectory(_record(trajectory=[{"step": 1}])) == [{"step": 1}]
        assert log.resolve_trajectory(_record()) is None

    async def test_compact_preserves_trajectory_ref(self, tmp_path):
        log = _open_log(tmp_path, max_trajectory_bytes=100)
        await log.append(_record(trajectory=BIG_TRAJECTORY))
        log.close()

        cache = log.load()
        log.compact(cache, meta={"config_hash": "sha256:abc"})

        rec = log.load()[(0, 0)]
        assert rec["trajectory_ref"] == f"{TRAJECTORY_OVERFLOW_DIR}/p0_r0.json.gz"
        assert log.resolve_trajectory(rec) == BIG_TRAJECTORY


class TestCaptureMarkers:
    def test_write_and_has(self, tmp_path):
        assert not has_capture_marker(tmp_path, 3, 1)
        write_capture_marker(tmp_path, 3, 1)
        assert has_capture_marker(tmp_path, 3, 1)
        assert not has_capture_marker(tmp_path, 3, 2)
        assert (tmp_path / CAPTURE_MARKER_DIR / "p3_r1.captured").is_file()

    def test_write_swallows_oserror(self, tmp_path):
        (tmp_path / CAPTURE_MARKER_DIR).write_text("not a directory")
        write_capture_marker(tmp_path, 0, 0)
        assert not has_capture_marker(tmp_path, 0, 0)

    def test_reset_checkpoint_sidecars(self, tmp_path):
        write_capture_marker(tmp_path, 0, 0)
        (tmp_path / TRAJECTORY_OVERFLOW_DIR).mkdir()
        (tmp_path / TRAJECTORY_OVERFLOW_DIR / "p0_r0.json.gz").write_bytes(b"x")

        reset_checkpoint_sidecars(tmp_path)

        assert not (tmp_path / CAPTURE_MARKER_DIR).exists()
        assert not (tmp_path / TRAJECTORY_OVERFLOW_DIR).exists()

    def test_reset_noop_when_absent(self, tmp_path):
        reset_checkpoint_sidecars(tmp_path)
