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

import importlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

from nemo_evaluator.core.entrypoint import run_eval


@pytest.fixture
def n():
    # Reuse dummy framework package from installed_modules/1
    return 1


@pytest.fixture(autouse=True)
def installed_modules(n: int, monkeypatch):
    if not n:
        monkeypatch.setitem(sys.modules, "core_evals", None)
        yield
        return

    pkg = "core_evals"
    # Prefer a sibling installed_modules/<n> directory; fallback to functional_tests location.
    search_paths = [
        os.path.join(Path(__file__).parent.absolute(), f"installed_modules/{n}"),
        os.path.join(
            Path(__file__).parents[2].absolute(),
            "functional_tests",
            "installed_modules",
            str(n),
        ),
    ]
    spec = None
    for sp in search_paths:
        spec = importlib.machinery.PathFinder().find_spec(pkg, [sp])
        if spec is not None:
            break
    if spec is None:
        raise RuntimeError(
            f"Could not locate dummy package for {pkg} in {search_paths}"
        )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    sys.modules[pkg] = importlib.import_module(pkg)
    yield
    sys.modules.pop(pkg)


def _basic_run_config(tmp_path: Path) -> dict:
    return {
        "config": {"type": "dummy_task", "output_dir": str(tmp_path)},
        "target": {
            "api_endpoint": {
                "type": "chat",
                "url": "https://nonexistent.com",
                "model_id": "dummy/model",
            }
        },
    }


def test_metadata_full(monkeypatch, tmp_path):
    monkeypatch.setenv("CORE_EVALS_GIT_HASH", "abc123")
    from nemo_evaluator import __version__ as nemo_evaluator_version

    cfg = _basic_run_config(tmp_path)
    cfg["metadata"] = {
        "versioning": {"foo": "1.2.3"},
        "data": {"b": 2},
    }

    cfg_path = tmp_path / "run.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(
        sys, "argv", ["prog", "run_eval", "--run_config", str(cfg_path)]
    )
    run_eval()

    # Files persisted
    md_file = tmp_path / "metadata.yaml"
    results_path = tmp_path / "results.yml"

    assert md_file.exists()
    assert results_path.exists()

    expected_versioning = {
        "foo": "1.2.3",
        "git-hash": "abc123",
        "nemo_evaluator": nemo_evaluator_version,
    }
    expected_metadata_full = {
        "versioning": expected_versioning,
        "data": {"b": 2},
    }
    assert yaml.safe_load(md_file.read_text()) == expected_metadata_full

    results = yaml.safe_load(results_path.read_text())
    assert results["git_hash"] == "abc123"
    assert results["metadata"]["versioning"] == expected_versioning


def test_no_metadata(monkeypatch, tmp_path):
    # No metadata in run_config; ensure no metadata block and no launcher files
    cfg = _basic_run_config(tmp_path)
    cfg_path = tmp_path / "run.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(
        sys, "argv", ["prog", "run_eval", "--run_config", str(cfg_path)]
    )
    run_eval()

    results = yaml.safe_load((tmp_path / "results.yml").read_text())
    assert "metadata" not in results
    assert not (tmp_path / "metadata.yaml").exists()


def test_env_versioning_and_resolved_only(monkeypatch, tmp_path):
    monkeypatch.setenv("CORE_EVALS_GIT_HASH", "deadbeef")
    from nemo_evaluator import __version__ as nemo_evaluator_version

    cfg = _basic_run_config(tmp_path)
    cfg["metadata"] = {"data": {"x": 7}}
    cfg_path = tmp_path / "run.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(
        sys, "argv", ["prog", "run_eval", "--run_config", str(cfg_path)]
    )
    run_eval()

    assert (tmp_path / "metadata.yaml").exists()

    results = yaml.safe_load((tmp_path / "results.yml").read_text())
    mv = results["metadata"]["versioning"]
    assert mv["git-hash"] == "deadbeef"
    assert mv["nemo_evaluator"] == nemo_evaluator_version


def test_only_versioning_no_env(monkeypatch, tmp_path):
    # Ensure nemo_evaluator_version is present and git-hash is absent when no env set
    from nemo_evaluator import __version__ as nemo_evaluator_version

    monkeypatch.delenv("CORE_EVALS_GIT_HASH", raising=False)

    cfg = _basic_run_config(tmp_path)
    cfg["metadata"] = {"versioning": {"bar": "9.9.9"}}
    cfg_path = tmp_path / "run.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(
        sys, "argv", ["prog", "run_eval", "--run_config", str(cfg_path)]
    )
    run_eval()

    mv = yaml.safe_load((tmp_path / "results.yml").read_text())["metadata"][
        "versioning"
    ]
    assert mv["bar"] == "9.9.9"

    assert mv["nemo_evaluator"] == nemo_evaluator_version
    assert "git-hash" not in mv
