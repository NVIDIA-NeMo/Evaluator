# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from argparse import Namespace

from nemo_evaluator.rasb_26h1 import runner


def test_strip_endpoint_path():
    assert (
        runner._strip_endpoint_path("https://example.test/v1/chat/completions")
        == "https://example.test/v1"
    )
    assert (
        runner._strip_endpoint_path("https://example.test/v1/completions/")
        == "https://example.test/v1"
    )
    assert runner._strip_endpoint_path("https://example.test/v1") == (
        "https://example.test/v1"
    )


def test_select_callable_prefers_anthropic_for_claude(tmp_path):
    root = tmp_path
    callable_dir = root / "callables" / "anthropic_lm"
    callable_dir.mkdir(parents=True)

    selected = runner._select_callable(
        root,
        explicit_callable=None,
        model_id="azure/anthropic/claude-sonnet-4-5",
    )

    assert selected == callable_dir


def test_select_callable_uses_openai_default(tmp_path):
    root = tmp_path
    callable_dir = root / "callables" / "openai_lm"
    callable_dir.mkdir(parents=True)

    selected = runner._select_callable(
        root,
        explicit_callable=None,
        model_id="gpt-4o",
    )

    assert selected == callable_dir


def test_select_callable_allows_explicit_relative_path(tmp_path):
    root = tmp_path
    callable_dir = root / "custom_callable"
    callable_dir.mkdir()

    selected = runner._select_callable(
        root,
        explicit_callable="custom_callable",
        model_id="gpt-4o",
    )

    assert selected == callable_dir


def test_build_extra_env_maps_endpoint_and_api_key(monkeypatch):
    monkeypatch.setenv("NVAPI_KEY", "secret")
    args = Namespace(
        model_id="azure/anthropic/claude-opus-4-5",
        model_url="https://inference-api.nvidia.com",
        temperature=0.2,
        judge_model="azure/anthropic/claude-opus-4-5",
    )

    extra_env = runner._build_extra_env(args, os.environ["NVAPI_KEY"])

    assert extra_env["TARGET_MODEL"] == "azure/anthropic/claude-opus-4-5"
    assert extra_env["NVAPI_KEY"] == "secret"
    assert extra_env["OPENAI_API_KEY"] == "secret"
    assert extra_env["ANTHROPIC_API_KEY"] == "secret"
    assert extra_env["NVIDIA_BASE_URL"] == "https://inference-api.nvidia.com"
    assert extra_env["OPENAI_BASE_URL"] == "https://inference-api.nvidia.com"
    assert extra_env["RASB_TEMPERATURE"] == "0.2"
    assert extra_env["JUDGE_MODEL"] == "azure/anthropic/claude-opus-4-5"


def test_discover_env_dirs_filters_and_slices(tmp_path):
    env_root = tmp_path / "26h1"
    env_root.mkdir()
    for name in ("alpha", "beta", "gamma"):
        env_dir = env_root / name
        env_dir.mkdir()
        (env_dir / "metadata.json").write_text("{}")

    envs = runner._discover_env_dirs(env_root, only="alpha", slice_spec="0:")

    assert [env_id for env_id, _ in envs] == ["alpha"]


def test_env_file_for_run_uses_empty_temp_file_by_default(tmp_path):
    default_env = tmp_path / ".env"
    default_env.write_text("API_KEY=secret\n", encoding="utf-8")

    env_file = runner._env_file_for_run(tmp_path, requested_env_file=None)

    assert env_file != default_env
    assert env_file.read_text(encoding="utf-8") == ""


def test_env_file_for_run_allows_explicit_env_file(tmp_path):
    default_env = tmp_path / ".env"
    default_env.write_text("API_KEY=secret\n", encoding="utf-8")

    env_file = runner._env_file_for_run(tmp_path, requested_env_file=".env")

    assert env_file == default_env.resolve()
