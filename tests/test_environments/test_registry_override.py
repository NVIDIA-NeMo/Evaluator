# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Harbor registry override layer.

Override shards under ``HARBOR_REGISTRY_OVERRIDE_DIR`` are merged into the
base registry so locally-vendored dataset entries can shadow or extend
whatever upstream serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_evaluator.environments import harbor
from nemo_evaluator.environments.registry import _make_harbor


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    harbor._registry_cache = None
    yield
    harbor._registry_cache = None


def _write_base(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(entries))
    return path


def _write_override(dir_path: Path, name: str, entries: list[dict]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{name}.json"
    path.write_text(json.dumps(entries))
    return path


_BASE_TB_2_0 = {
    "name": "terminal-bench",
    "version": "2.0",
    "description": "Upstream TB2.0",
    "tasks": [
        {
            "name": "task-a",
            "git_url": "https://github.com/laude-institute/terminal-bench-2.git",
            "git_commit_id": "69671fba",
            "path": "task-a",
        }
    ],
}


_OVERRIDE_TB_2_1 = {
    "name": "terminal-bench/terminal-bench-2-1",
    "version": "1.0",
    "description": "Vendored TB2.1",
    "tasks": [
        {
            "name": "task-a",
            "git_url": "https://github.com/harbor-framework/terminal-bench-2-1.git",
            "git_commit_id": "c5ee500c",
            "path": "tasks/task-a",
        }
    ],
}


def test_override_adds_new_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "overrides"))
    _write_override(tmp_path / "overrides", "tb21", [_OVERRIDE_TB_2_1])

    registry = harbor.get_registry()
    names = sorted((d.name, d.version) for d in registry)
    assert ("terminal-bench", "2.0") in names
    assert ("terminal-bench/terminal-bench-2-1", "1.0") in names


def test_override_wins_on_name_version_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "overrides"))
    patched = dict(_BASE_TB_2_0)
    patched["description"] = "Patched upstream entry"
    patched["tasks"] = [
        {**_BASE_TB_2_0["tasks"][0], "git_commit_id": "deadbeef"},
    ]
    _write_override(tmp_path / "overrides", "patch", [patched])

    dataset = harbor.find_dataset("terminal-bench", "2.0")
    assert dataset.description == "Patched upstream entry"
    assert dataset.tasks[0].git_commit_id == "deadbeef"


def test_missing_override_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "does-not-exist"))

    registry = harbor.get_registry()
    assert len(registry) == 1
    assert registry[0].name == "terminal-bench"


def test_empty_override_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    overrides = tmp_path / "overrides"
    overrides.mkdir()
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(overrides))

    registry = harbor.get_registry()
    assert len(registry) == 1


def test_find_dataset_returns_harbor_hub_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "overrides"))
    _write_override(tmp_path / "overrides", "tb21", [_OVERRIDE_TB_2_1])

    dataset = harbor.find_dataset("terminal-bench/terminal-bench-2-1")
    assert dataset.version == "1.0"
    assert dataset.tasks[0].path == "tasks/task-a"
    assert "harbor-framework/terminal-bench-2-1" in dataset.tasks[0].git_url


@pytest.mark.parametrize(
    "uri, expected_cache_subdir",
    [
        ("terminal-bench@2.0", "terminal-bench@2.0"),
        (
            "terminal-bench/terminal-bench-2-1",
            "terminal-bench__terminal-bench-2-1",
        ),
    ],
    ids=["legacy_at_version", "harbor_hub_slash"],
)
def test_make_harbor_dispatch_pre_staged(tmp_path, monkeypatch, uri, expected_cache_subdir):
    """`_make_harbor` resolves both URI conventions to the right cache dir
    and reuses a pre-staged dataset without hitting the registry."""
    datasets_dir = tmp_path / "harbor_datasets"
    pre_staged = datasets_dir / expected_cache_subdir
    pre_staged.mkdir(parents=True)
    (pre_staged / "task-a").mkdir()
    (pre_staged / "task-a" / "instruction.md").write_text("Do something")

    monkeypatch.setenv("HARBOR_DATASETS_DIR", str(datasets_dir))
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "no-overrides"))

    env = _make_harbor(uri)
    assert env.name == expected_cache_subdir


@pytest.mark.parametrize(
    "params, expected_keep_hints",
    [
        (None, True),
        ({}, True),
        ({"keep_swebench_multilingual_hints": False}, False),
        ({"keep_swebench_multilingual_hints": True}, True),
    ],
    ids=["no_params", "empty_params", "explicit_false", "explicit_true"],
)
def test_make_harbor_forwards_keep_hints_param(tmp_path, monkeypatch, params, expected_keep_hints):
    """`_make_harbor` threads the ``keep_swebench_multilingual_hints`` param into the env."""
    datasets_dir = tmp_path / "harbor_datasets"
    pre_staged = datasets_dir / "swebench_multilingual@1.0"
    pre_staged.mkdir(parents=True)
    (pre_staged / "task-a").mkdir()
    (pre_staged / "task-a" / "instruction.md").write_text("Do something")

    monkeypatch.setenv("HARBOR_DATASETS_DIR", str(datasets_dir))
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "no-overrides"))

    kwargs = {} if params is None else {"params": params}
    env = _make_harbor("swebench_multilingual@1.0", **kwargs)
    assert env._keep_swebench_multilingual_hints is expected_keep_hints


@pytest.mark.parametrize(
    "bad_value", ["false", "true", 1, 0, None], ids=["str_false", "str_true", "int_1", "int_0", "none"]
)
def test_make_harbor_rejects_non_bool_keep_hints(tmp_path, monkeypatch, bad_value):
    """A non-bool ``keep_swebench_multilingual_hints`` fails fast (no truthiness coercion)."""
    datasets_dir = tmp_path / "harbor_datasets"
    pre_staged = datasets_dir / "swebench_multilingual@1.0"
    pre_staged.mkdir(parents=True)
    (pre_staged / "task-a").mkdir()
    (pre_staged / "task-a" / "instruction.md").write_text("Do something")

    monkeypatch.setenv("HARBOR_DATASETS_DIR", str(datasets_dir))
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [])))
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(tmp_path / "no-overrides"))

    with pytest.raises(TypeError, match="keep_swebench_multilingual_hints must be a bool"):
        _make_harbor("swebench_multilingual@1.0", params={"keep_swebench_multilingual_hints": bad_value})


def test_local_override_dir_prefers_in_package_over_repo_root(tmp_path, monkeypatch):
    """Wheel installs ship the override dir as a package sibling.

    When ``nemo_evaluator/_registry_overrides`` exists it must win over the
    repo-root fallback so non-editable installs resolve correctly.
    """
    from nemo_evaluator import paths

    fake_pkg = tmp_path / "nemo_evaluator"
    fake_pkg.mkdir()
    (fake_pkg / "_registry_overrides").mkdir()
    fake_paths_py = fake_pkg / "paths.py"
    fake_paths_py.touch()

    monkeypatch.setattr(paths, "__file__", str(fake_paths_py))
    assert paths.local_registry_override_dir() == fake_pkg / "_registry_overrides"

    (fake_pkg / "_registry_overrides").rmdir()
    assert paths.local_registry_override_dir() == paths.REPO_ROOT / "harbor_datasets" / "registry_overrides"


def test_env_var_pointing_at_non_dir_logs_warning_and_is_noop(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    bogus = tmp_path / "does-not-exist"
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(bogus))

    import logging

    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.environments.harbor"):
        registry = harbor.get_registry()

    assert len(registry) == 1
    assert any("HARBOR_REGISTRY_OVERRIDE_DIR" in rec.message and "ignoring" in rec.message for rec in caplog.records)


def test_malformed_shard_is_skipped_not_fatal(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    overrides = tmp_path / "overrides"
    overrides.mkdir()
    (overrides / "bad_schema.json").write_text(
        json.dumps([{"name": "only-name-no-version-no-tasks"}])  # missing required keys
    )
    _write_override(overrides, "good", [_OVERRIDE_TB_2_1])
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(overrides))

    import logging

    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.environments.harbor"):
        registry = harbor.get_registry()

    names = {d.name for d in registry}
    assert "terminal-bench/terminal-bench-2-1" in names
    assert "terminal-bench" in names
    assert any("bad_schema.json" in rec.message and "malformed" in rec.message for rec in caplog.records)


def test_multiple_shards_all_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("HARBOR_REGISTRY", str(_write_base(tmp_path, [_BASE_TB_2_0])))
    overrides = tmp_path / "overrides"
    monkeypatch.setenv("HARBOR_REGISTRY_OVERRIDE_DIR", str(overrides))
    _write_override(overrides, "tb21", [_OVERRIDE_TB_2_1])
    _write_override(
        overrides,
        "extra",
        [
            {
                "name": "another-bench",
                "version": "1.0",
                "description": "Extra shard",
                "tasks": [],
            }
        ],
    )

    registry = harbor.get_registry()
    names = {d.name for d in registry}
    assert names == {
        "terminal-bench",
        "terminal-bench/terminal-bench-2-1",
        "another-bench",
    }
