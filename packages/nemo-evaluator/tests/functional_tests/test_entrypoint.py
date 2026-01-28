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
    return 1


@pytest.fixture(autouse=True)
def installed_modules(n: int, monkeypatch):
    if not n:
        monkeypatch.setitem(sys.modules, "core_evals", None)
        yield
        return

    pkg = "core_evals"

    spec = importlib.machinery.PathFinder().find_spec(
        pkg, [os.path.join(Path(__file__).parent.absolute(), f"installed_modules/{n}")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    sys.modules[pkg] = importlib.import_module(pkg)
    yield
    sys.modules.pop(pkg)


@pytest.mark.parametrize(
    "n,expected_output",
    [
        (0, "NO evaluation packages are installed.\n"),
        (
            1,
            "dummy-framework: \n  * dummy_default_task\n  * dummy_task\n  * dummy_chat_task\n",
        ),
    ],
)
def test_ls(monkeypatch, capsys, expected_output):
    monkeypatch.setattr(sys, "argv", ["prog", "ls"])
    run_eval()
    captured = capsys.readouterr()
    assert expected_output in captured.out


@pytest.mark.parametrize(
    "eval_type,score", [("dummy_task", 42), ("dummy_default_task", 123)]
)
@pytest.mark.parametrize("model_type", ["chat", "completions", "vlm", "embedding"])
def test_run(monkeypatch, tmp_path, eval_type, score, model_type):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run_eval",
            "--eval_type",
            eval_type,
            "--model_id",
            "dummy-provider/dummy-endpoint",
            "--model_url",
            "https://nonexistent.com",
            "--model_type",
            model_type,
            "--output_dir",
            str(tmp_path),
            "--model_url",
            "https://nonexistent.com",
        ],
    )
    run_eval()
    results_path = tmp_path / "results.yml"
    assert results_path.exists()
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    assert (
        results["results"]["tasks"]["dummy_task"]["metrics"]["dummy_metric"]["scores"][
            "dummy_score"
        ]["value"]
        == score
    )


@pytest.mark.parametrize(
    "model_type,should_fail",
    [("chat", False), ("completions", True), ("vlm", True), ("embedding", True)],
)
def test_run_chat(monkeypatch, tmp_path, model_type, should_fail):
    import sys

    from nemo_evaluator.core.utils import MisconfigurationError

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run_eval",
            "--eval_type",
            "dummy_chat_task",
            "--model_id",
            "dummy-provider/dummy-endpoint",
            "--model_url",
            "https://nonexistent.com",
            "--model_type",
            model_type,
            "--output_dir",
            str(tmp_path),
            "--model_url",
            "https://nonexistent.com",
        ],
    )
    results_path = tmp_path / "results.yml"
    if should_fail:
        with pytest.raises(MisconfigurationError):
            run_eval()
        assert not results_path.exists()
    else:
        run_eval()
        assert results_path.exists()


@pytest.mark.parametrize(
    "task_name,matching_msg",
    [
        ("nonexistent_task", "Unknown evaluation nonexistent_task.*"),
        (
            "nonexistent_framework.dummy_task",
            "Unknown framework nonexistent_framework.*",
        ),
        (
            "framework_name.task_name.another_level_shouldnotbehere",
            "eval_type must follow 'framework_name.evaluation_name'. No additional dots are allowed.",
        ),
    ],
)
def test_run_framework_nonexistent(monkeypatch, tmp_path, task_name, matching_msg):
    import sys

    from nemo_evaluator.core.utils import MisconfigurationError

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run_eval",
            "--eval_type",
            task_name,
            "--model_id",
            "dummy-provider/dummy-endpoint",
            "--model_url",
            "https://nonexistent.com",
            "--model_type",
            "chat",
            "--output_dir",
            str(tmp_path),
            "--model_url",
            "https://nonexistent.com",
        ],
    )
    results_path = tmp_path / "results.yml"

    with pytest.raises(MisconfigurationError, match=matching_msg):
        run_eval()
    assert not results_path.exists()


def test_run_framework_not_listed_eval(monkeypatch, tmp_path):
    """Test that framework.task invocation works for tasks not explicitly listed in framework.yml."""
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "run_eval",
            "--eval_type",
            "dummy-framework.not_listed_eval",
            "--model_id",
            "dummy-provider/dummy-endpoint",
            "--model_url",
            "https://nonexistent.com",
            "--model_type",
            "chat",
            "--output_dir",
            str(tmp_path),
        ],
    )
    run_eval()
    results_path = tmp_path / "results.yml"
    assert results_path.exists()
    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    # Verify score is 123 (the default score from framework defaults)
    assert (
        results["results"]["tasks"]["dummy_task"]["metrics"]["dummy_metric"]["scores"][
            "dummy_score"
        ]["value"]
        == 123
    )

    # Verify config.type maintains the "framework.task" structure
    assert results["config"]["type"] == "dummy-framework.not_listed_eval"
