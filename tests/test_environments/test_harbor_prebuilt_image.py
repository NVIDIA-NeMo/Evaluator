# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prebuilt task images — task.toml [environment] docker_image handling.

A task that declares ``docker_image`` ships a published image built from its own
Dockerfile.  Rebuilding that Dockerfile re-resolves every unpinned apt/pip/curl
reference against today's internet, so the declared image is what should run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_evaluator.environments.harbor import HarborEnvironment

_DOCKERFILE_WITH_LAYERS = "FROM python:3.12-slim\nRUN echo build-step\n"


def _make_task_dir(root: Path, name: str, *, docker_image: str | None = None) -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text(f"do {name}")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (task_dir / "environment" / "Dockerfile").write_text(_DOCKERFILE_WITH_LAYERS)
    body = "[verifier]\ntimeout_sec = 600\n"
    if docker_image is not None:
        body += f'\n[environment]\ndocker_image = "{docker_image}"\n'
    (task_dir / "task.toml").write_text(body)
    return task_dir


@pytest.mark.asyncio
async def test_image_build_requests_skips_tasks_with_prebuilt_image(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "prebuilt", docker_image="acme/prebuilt:20251031")
    _make_task_dir(dataset, "needs-build")

    env = HarborEnvironment(dataset_path=str(dataset))
    reqs = await env.image_build_requests()

    assert reqs is not None
    images = [spec.image for spec in reqs[0].specs]
    assert len(images) == 1, f"only the task without docker_image should build, got {images}"
    assert "needs-build" in images[0]


@pytest.mark.asyncio
async def test_image_build_requests_none_when_every_task_is_prebuilt(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "a", docker_image="acme/a:1")
    _make_task_dir(dataset, "b", docker_image="acme/b:1")

    env = HarborEnvironment(dataset_path=str(dataset))
    assert await env.image_build_requests() is None


@pytest.mark.asyncio
async def test_seed_marks_prebuilt_image_as_mirror_source(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t", docker_image="acme/prebuilt:20251031")

    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)

    assert seed.sandbox_spec.image == "acme/prebuilt:20251031"
    assert seed.sandbox_spec.source_image == "acme/prebuilt:20251031"
    assert seed.verify_sandbox_spec.source_image == "acme/prebuilt:20251031"


@pytest.mark.asyncio
async def test_seed_leaves_source_image_unset_when_image_is_built_locally(tmp_path):
    dataset = tmp_path / "ds"
    _make_task_dir(dataset, "t")

    env = HarborEnvironment(dataset_path=str(dataset))
    seed = await env.seed(0)

    assert seed.sandbox_spec.source_image is None
