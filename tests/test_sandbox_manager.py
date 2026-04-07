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
"""Tests for SandboxManager: resolve_spec merge logic, image template rendering."""

from __future__ import annotations


from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.sandbox.base import SandboxSpec, VolumeMount


def _make_seed(**overrides) -> SeedResult:
    defaults = dict(prompt="test", expected_answer="42")
    defaults.update(overrides)
    return SeedResult(**defaults)


# ---------------------------------------------------------------------------
# resolve_spec merge logic
# ---------------------------------------------------------------------------


class TestResolveSpec:
    def _make_manager(self, image_template=None, default_image=None):
        from nemo_evaluator.sandbox.manager import SandboxManager

        return SandboxManager(
            backend="local",
            image_template=image_template,
            default_image=default_image,
        )

    def test_returns_none_when_no_spec(self):
        mgr = self._make_manager()
        seed = _make_seed()
        assert mgr.resolve_spec(seed) is None

    def test_returns_seed_spec_verbatim(self):
        mgr = self._make_manager()
        spec = SandboxSpec(image="custom:v1", workdir="/app", env={"FOO": "bar"})
        seed = _make_seed(sandbox_spec=spec)

        result = mgr.resolve_spec(seed)
        assert result is not None
        assert result.image == "custom:v1"
        assert result.workdir == "/app"
        assert result.env == {"FOO": "bar"}

    def test_image_template_overrides_image(self):
        mgr = self._make_manager(image_template="registry.io/{task_id}")
        spec = SandboxSpec(image="python:3.12-slim", workdir="/ws")
        seed = _make_seed(sandbox_spec=spec, metadata={"task_id": "my-task"})

        result = mgr.resolve_spec(seed)
        assert result is not None
        assert result.image == "registry.io/my-task"
        assert result.workdir == "/ws"

    def test_image_template_missing_key_falls_back(self):
        """When template key is missing from metadata, falls back to seed image."""
        mgr = self._make_manager(image_template="registry.io/{missing_key}")
        spec = SandboxSpec(image="python:3.12-slim")
        seed = _make_seed(sandbox_spec=spec)

        result = mgr.resolve_spec(seed)
        assert result is not None
        assert result.image == "python:3.12-slim"

    def test_default_image_used_when_no_spec(self):
        mgr = self._make_manager(default_image="fallback:latest")
        seed = _make_seed()

        result = mgr.resolve_spec(seed)
        if result is not None:
            assert result.image == "fallback:latest"

    def test_volumes_appended(self):
        mgr = self._make_manager()
        vol = VolumeMount(host_path="/tmp/shared", container_path="/data")
        spec = SandboxSpec(image="img:v1", volumes=[vol])
        extra = [VolumeMount(host_path="/tmp/extra", container_path="/extra")]

        seed = _make_seed(sandbox_spec=spec)
        result = mgr.resolve_spec(seed, extra_volumes=extra)
        if result is not None:
            container_paths = [v.container_path for v in result.volumes]
            assert "/data" in container_paths
            assert "/extra" in container_paths


# ---------------------------------------------------------------------------
# SandboxSpec fields
# ---------------------------------------------------------------------------


class TestSandboxSpec:
    def test_defaults(self):
        spec = SandboxSpec(image="python:3.12")
        assert spec.workdir == "/workspace"
        assert spec.env == {}
        assert spec.files == {}
        assert spec.entrypoint is None
        assert spec.volumes == []

    def test_volume_mount_readonly(self):
        vol = VolumeMount(host_path="/h", container_path="/c", readonly=True)
        assert vol.readonly is True
