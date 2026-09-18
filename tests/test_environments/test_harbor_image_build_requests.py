# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Harbor build planning and runtime image selection."""

from pathlib import Path

import pytest

from nemo_evaluator.environments.harbor import HarborEnvironment, _built_image_tag
from nemo_evaluator.sandbox.base import ImageSpec


@pytest.mark.parametrize(
    ("task_toml", "needs_build"),
    [
        pytest.param('[environment]\ndocker_image = "example/prebuilt:1.0"\n', False, id="declared-image"),
        pytest.param("[environment]\n", True, id="no-declared-image"),
        pytest.param(None, True, id="no-task-toml"),
    ],
)
async def test_image_build_requests_respects_declared_image(
    tmp_path: Path, task_toml: str | None, needs_build: bool
) -> None:
    dataset = tmp_path / "dataset"
    task_dir = dataset / "task"
    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Solve the task.\n", encoding="utf-8")
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\nRUN touch /prepared\n", encoding="utf-8")
    if task_toml is not None:
        (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")

    env = HarborEnvironment(dataset_path=dataset)
    requests = await env.image_build_requests()

    if not needs_build:
        assert requests is None
        assert env._resolve_image(task_dir) == "example/prebuilt:1.0"
    else:
        assert requests is not None
        assert len(requests) == 1
        expected_image = _built_image_tag(dataset.name, task_dir.name, env_dir)
        assert requests[0].specs == [ImageSpec(image=expected_image, source={"task_dir": str(task_dir)})]
        assert callable(requests[0].docker_build_fn)
        assert env._resolve_image(task_dir) == expected_image
