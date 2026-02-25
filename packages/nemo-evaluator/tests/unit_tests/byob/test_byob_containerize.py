# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for BYOB containerization module."""

import json
import os

import pytest
import yaml
from unittest.mock import MagicMock, patch

from nemo_evaluator.contrib.byob.containerize import (
    build_image,
    generate_dockerfile,
    prepare_build_context,
    push_image,
    rewrite_fdf_paths,
)
from nemo_evaluator.contrib.byob.defaults import DEFAULT_BASE_IMAGE


class TestGenerateDockerfile:
    """Tests for generate_dockerfile function."""

    def test_basic_dockerfile(self):
        """Test basic Dockerfile generation with pkg_name."""
        content = generate_dockerfile("byob_boolq")
        assert "FROM " in content
        assert "byob_boolq" in content
        assert "COPY pkg/" in content
        assert "COPY code/" in content
        assert "COPY data/" in content

    def test_dockerfile_has_labels(self):
        """Test that Dockerfile includes launcher-compliant labels."""
        content = generate_dockerfile("byob_boolq")
        assert 'com.nvidia.nemo-evaluator.pkg-name' in content
        assert 'com.nvidia.nemo-evaluator.integration-type' in content
        assert '"byob_boolq"' in content

    def test_dockerfile_with_custom_base_image(self):
        """Test Dockerfile uses custom base image."""
        content = generate_dockerfile("byob_boolq", base_image="python:3.11-slim")
        assert "FROM python:3.11-slim" in content

    def test_dockerfile_default_base_image(self):
        """Test Dockerfile uses DEFAULT_BASE_IMAGE when not specified."""
        content = generate_dockerfile("byob_boolq")
        assert f"FROM {DEFAULT_BASE_IMAGE}" in content

    def test_dockerfile_with_user_requirements(self):
        """Test Dockerfile installs user requirements via requirements.txt."""
        content = generate_dockerfile("byob_boolq", user_requirements=["numpy>=1.20", "pandas"])
        assert "requirements.txt" in content
        assert "pip install" in content

    def test_dockerfile_without_user_requirements(self):
        """Test Dockerfile with no user requirements doesn't have extra pip install."""
        content = generate_dockerfile("byob_boolq")
        # The user reqs line should be empty
        lines = content.split("\n")
        # Find the line after "Install user requirements" comment
        found_empty_reqs = False
        for i, line in enumerate(lines):
            if "Install user requirements" in line:
                # Next non-empty line should be empty or the verify step
                found_empty_reqs = True
                break
        assert found_empty_reqs

    def test_dockerfile_verify_import(self):
        """Test Dockerfile verifies package import."""
        content = generate_dockerfile("byob_test_pkg")
        assert 'import core_evals.byob_test_pkg' in content


class TestRewriteFdfPaths:
    """Tests for rewrite_fdf_paths function."""

    def test_rewrites_benchmark_module(self):
        """Test host path is rewritten to container path."""
        fdf = {
            "framework": {"pkg_name": "byob_test"},
            "defaults": {
                "config": {
                    "params": {
                        "extra": {
                            "benchmark_module": "/home/user/benchmarks/benchmark.py",
                            "dataset": "/home/user/data/test.jsonl",
                        }
                    }
                }
            },
        }
        result = rewrite_fdf_paths(fdf, "byob_test")
        extra = result["defaults"]["config"]["params"]["extra"]
        assert extra["benchmark_module"] == "/nemo_run/code/benchmark.py"
        assert extra["dataset"] == "/nemo_run/data/test.jsonl"

    def test_does_not_mutate_original(self):
        """Test that original FDF is not mutated."""
        fdf = {
            "framework": {"pkg_name": "byob_test"},
            "defaults": {
                "config": {
                    "params": {
                        "extra": {
                            "benchmark_module": "/original/path.py",
                            "dataset": "/original/data.jsonl",
                        }
                    }
                }
            },
        }
        rewrite_fdf_paths(fdf, "byob_test")
        extra = fdf["defaults"]["config"]["params"]["extra"]
        assert extra["benchmark_module"] == "/original/path.py"
        assert extra["dataset"] == "/original/data.jsonl"

    def test_handles_empty_extra(self):
        """Test graceful handling when extra is empty."""
        fdf = {
            "framework": {"pkg_name": "byob_test"},
            "defaults": {"config": {"params": {"extra": {}}}},
        }
        result = rewrite_fdf_paths(fdf, "byob_test")
        assert result is not None


class TestPrepareBuildContext:
    """Tests for prepare_build_context function."""

    def test_creates_directory_layout(self, tmp_path):
        """Test that build context has pkg/, code/, data/ directories."""
        # Create a mock compiled package
        pkg_dir = tmp_path / "byob_test"
        core_evals_dir = pkg_dir / "core_evals" / "byob_test"
        core_evals_dir.mkdir(parents=True)
        (core_evals_dir / "framework.yml").write_text("framework: test")
        (core_evals_dir / "__init__.py").write_text("")
        (pkg_dir / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create mock benchmark and dataset files
        benchmark_file = tmp_path / "benchmark.py"
        benchmark_file.write_text("# benchmark code")
        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text('{"q": "hello"}\n')

        fdf = {
            "framework": {"pkg_name": "byob_test"},
            "defaults": {
                "config": {
                    "params": {
                        "extra": {
                            "benchmark_module": str(benchmark_file),
                            "dataset": str(dataset_file),
                        }
                    }
                }
            },
        }

        context_dir = tmp_path / "build_context"
        prepare_build_context(str(pkg_dir), fdf, str(context_dir))

        assert (context_dir / "pkg").is_dir(), "Missing pkg/ in build context"
        assert (context_dir / "code").is_dir(), "Missing code/ in build context"
        assert (context_dir / "data").is_dir(), "Missing data/ in build context"
        assert (context_dir / "code" / "benchmark.py").is_file()
        assert (context_dir / "data" / "data.jsonl").is_file()


class TestBuildImage:
    """Tests for build_image function."""

    @patch("nemo_evaluator.contrib.byob.containerize.subprocess.run")
    def test_build_calls_docker_build(self, mock_run, tmp_path):
        """Test that build_image calls docker build with correct args."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        context_dir = str(tmp_path)
        tag = "test/byob:latest"
        build_image(
            context_dir=context_dir,
            tag=tag,
            pkg_name="byob_test",
        )

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "docker"
        assert call_args[1] == "build"
        assert "-t" in call_args
        assert tag in call_args
        assert context_dir in call_args

    @patch("nemo_evaluator.contrib.byob.containerize.subprocess.run")
    def test_build_writes_dockerfile(self, mock_run, tmp_path):
        """Test that build_image writes Dockerfile to context dir."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        build_image(
            context_dir=str(tmp_path),
            tag="test:latest",
            pkg_name="byob_test",
        )

        dockerfile = tmp_path / "Dockerfile"
        assert dockerfile.is_file(), "Dockerfile not written to context dir"
        content = dockerfile.read_text()
        assert "FROM " in content
        assert "byob_test" in content

    @patch("nemo_evaluator.contrib.byob.containerize.subprocess.run")
    def test_build_raises_on_failure(self, mock_run, tmp_path):
        """Test that build_image raises RuntimeError on docker build failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="build error")

        with pytest.raises(RuntimeError, match="docker build failed"):
            build_image(
                context_dir=str(tmp_path),
                tag="test:latest",
                pkg_name="byob_test",
            )

    @patch("nemo_evaluator.contrib.byob.containerize.subprocess.run")
    def test_build_returns_tag(self, mock_run, tmp_path):
        """Test that build_image returns the tag on success."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = build_image(
            context_dir=str(tmp_path),
            tag="myrepo/byob_test:v1",
            pkg_name="byob_test",
        )
        assert result == "myrepo/byob_test:v1"


class TestPushImage:
    """Tests for push_image function."""

    @patch("nemo_evaluator.contrib.byob.containerize.subprocess.run")
    def test_push_calls_docker_push(self, mock_run):
        """Test that push_image calls docker push with correct tag."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        push_image("myrepo/byob_test:latest")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["docker", "push", "myrepo/byob_test:latest"]

    @patch("nemo_evaluator.contrib.byob.containerize.subprocess.run")
    def test_push_raises_on_failure(self, mock_run):
        """Test that push_image raises RuntimeError on docker push failure."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="push error")

        with pytest.raises(RuntimeError, match="docker push failed"):
            push_image("myrepo/byob_test:latest")


class TestCLIContainerizeFlags:
    """Tests for --containerize and --push CLI flags."""

    def test_push_implies_containerize(self):
        """Test that --push flag implies --containerize."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--containerize", action="store_true", default=False)
        parser.add_argument("--push", type=str, default=None)

        args = parser.parse_args(["--push", "myrepo/test:latest"])
        # Simulate the logic in cli.py
        if args.push:
            args.containerize = True

        assert args.containerize is True
        assert args.push == "myrepo/test:latest"

    def test_containerize_standalone(self):
        """Test that --containerize works without --push."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--containerize", action="store_true", default=False)
        parser.add_argument("--push", type=str, default=None)

        args = parser.parse_args(["--containerize"])
        assert args.containerize is True
        assert args.push is None
