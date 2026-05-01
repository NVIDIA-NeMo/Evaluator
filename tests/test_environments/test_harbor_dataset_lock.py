# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dataset-directory file lock that serializes
``download_harbor_tasks`` callers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from nemo_evaluator.environments.harbor import _dataset_dir_lock


def test_lock_creates_sibling_lockfile(tmp_path: Path) -> None:
    output_dir = tmp_path / "harbor_datasets" / "terminal-bench@2.0"
    with _dataset_dir_lock(output_dir):
        assert (output_dir.parent / ".terminal-bench@2.0.lock").exists()


def test_lock_releases_after_block(tmp_path: Path) -> None:
    output_dir = tmp_path / "ds"
    with _dataset_dir_lock(output_dir):
        pass
    # Re-acquiring must not block.
    with _dataset_dir_lock(output_dir):
        pass


def test_lock_propagates_exceptions(tmp_path: Path) -> None:
    output_dir = tmp_path / "ds"
    with pytest.raises(RuntimeError, match="boom"):
        with _dataset_dir_lock(output_dir):
            raise RuntimeError("boom")
    # Lock must have been released even though the body raised.
    with _dataset_dir_lock(output_dir):
        pass


def _hold_lock(output_dir: str, hold_seconds: float, started_path: str, finished_path: str) -> None:
    with _dataset_dir_lock(Path(output_dir)):
        Path(started_path).touch()
        time.sleep(hold_seconds)
        Path(finished_path).touch()


def test_lock_serializes_concurrent_callers(tmp_path: Path) -> None:
    output_dir = tmp_path / "ds"

    a_started = tmp_path / "a_started"
    a_finished = tmp_path / "a_finished"
    b_started = tmp_path / "b_started"
    b_finished = tmp_path / "b_finished"

    ctx = mp.get_context("spawn")
    proc_a = ctx.Process(target=_hold_lock, args=(str(output_dir), 0.6, str(a_started), str(a_finished)))
    proc_b = ctx.Process(target=_hold_lock, args=(str(output_dir), 0.0, str(b_started), str(b_finished)))

    proc_a.start()
    while not a_started.exists():
        time.sleep(0.02)
    # A holds the lock for ~0.6s; start B which must wait until A releases.
    proc_b.start()

    proc_a.join(timeout=10)
    proc_b.join(timeout=10)
    assert proc_a.exitcode == 0
    assert proc_b.exitcode == 0

    a_finish_t = os.path.getmtime(a_finished)
    b_start_t = os.path.getmtime(b_started)
    assert b_start_t >= a_finish_t, "B entered the critical section before A released the lock"
