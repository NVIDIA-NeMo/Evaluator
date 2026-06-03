# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nemo_evaluator.run_store import RunMeta, load_run_meta


def test_run_meta_save_does_not_require_central_store(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    central_store_file = tmp_path / "not-a-directory"
    central_store_file.write_text("blocks directory creation")

    meta = RunMeta(
        run_id="run-1",
        executor="local",
        output_dir=str(output_dir),
        started_at="2026-05-30T00:00:00+00:00",
        config_summary="gsm8k",
    )

    with patch("nemo_evaluator.run_store._runs_store", return_value=central_store_file):
        meta.save()

    loaded = load_run_meta(output_dir)
    assert loaded is not None
    assert loaded.run_id == "run-1"
