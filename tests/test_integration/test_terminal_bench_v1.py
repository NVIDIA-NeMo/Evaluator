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
"""Tests for the terminal-bench-v1 BYOB benchmark."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_evaluator.benchmarks.terminal_bench_hard import (
    _TB_HARD_TASKS,
    TerminalBenchHard,
)
from nemo_evaluator.benchmarks.terminal_bench_v1 import (
    _ensure_dataset,
    _find_tasks_dir,
    TerminalBenchV1,
)

MINIMAL_TASK_YAML = """\
instruction: "Do something useful"
author_name: "Test Author"
difficulty: unknown
max_agent_timeout_sec: 120
max_test_timeout_sec: 30
"""

MINIMAL_DOCKER_COMPOSE = """\
services:
  client:
    build: .
    command: ["sh", "-c", "sleep infinity"]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
"""

MINIMAL_DOCKERFILE = "FROM ubuntu:22.04\nRUN echo hello\n"

INSTALL_WINDOWS_XP_DOCKERFILE = """\
FROM ubuntu:24.04
COPY etc/supervisor/supervisord.conf /etc/supervisor/supervisord.conf
ENTRYPOINT [ "supervisord", "-c", "/etc/supervisor/supervisord.conf" ]
"""

MINIMAL_RUN_TESTS = "#!/bin/bash\necho test\nexit 0\n"


def _create_v1_task(parent: Path, name: str, dockerfile: str = MINIMAL_DOCKERFILE) -> Path:
    """Create a minimal Terminal-Bench v1 task directory."""
    task_dir = parent / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(MINIMAL_TASK_YAML)
    (task_dir / "docker-compose.yaml").write_text(MINIMAL_DOCKER_COMPOSE)
    (task_dir / "Dockerfile").write_text(dockerfile)
    (task_dir / "run-tests.sh").write_text(MINIMAL_RUN_TESTS)
    return task_dir


class TestFindTasksDir:
    def test_tasks_at_root(self, tmp_path):
        _create_v1_task(tmp_path, "task-a")
        _create_v1_task(tmp_path, "task-b")
        assert _find_tasks_dir(tmp_path) == tmp_path

    def test_tasks_in_subdirectory(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _create_v1_task(tasks_dir, "task-a")
        assert _find_tasks_dir(tmp_path) == tasks_dir

    def test_no_tasks(self, tmp_path):
        (tmp_path / "readme.md").write_text("nothing here")
        assert _find_tasks_dir(tmp_path) is None

    def test_ignores_non_task_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "docs").mkdir()
        assert _find_tasks_dir(tmp_path) is None


class TestEnsureDataset:
    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._git")
    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._find_tasks_dir")
    def test_maps_tasks_to_harbor_format(self, mock_find, mock_git, tmp_path):
        """Clones repo, maps tasks, produces instruction.md in output."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _create_v1_task(source_dir, "task-alpha")
        _create_v1_task(source_dir, "task-beta")

        mock_find.return_value = source_dir

        output = _ensure_dataset(datasets_dir=str(tmp_path))

        assert (output / "task-alpha" / "instruction.md").exists()
        assert (output / "task-beta" / "instruction.md").exists()
        assert (output / ".tbv1_mapped").exists()

    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._git")
    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._find_tasks_dir")
    def test_uses_cache_on_second_call(self, mock_find, mock_git, tmp_path):
        """Second call reuses cache, doesn't clone again."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _create_v1_task(source_dir, "task-one")
        mock_find.return_value = source_dir

        _ensure_dataset(datasets_dir=str(tmp_path))
        assert mock_git.call_count == 1

        _ensure_dataset(datasets_dir=str(tmp_path))
        # No additional clone call
        assert mock_git.call_count == 1

    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._git")
    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._find_tasks_dir")
    def test_raises_when_no_tasks(self, mock_find, mock_git, tmp_path):
        mock_find.return_value = None
        with pytest.raises(RuntimeError, match="No terminal-bench v1"):
            _ensure_dataset(datasets_dir=str(tmp_path))

    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._git")
    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._find_tasks_dir")
    def test_patches_install_windows_xp_entrypoint_after_mapping(self, mock_find, mock_git, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        _create_v1_task(source_dir, "install-windows-xp", dockerfile=INSTALL_WINDOWS_XP_DOCKERFILE)
        mock_find.return_value = source_dir

        output = _ensure_dataset(datasets_dir=str(tmp_path))

        dockerfile = output / "install-windows-xp" / "environment" / "Dockerfile"
        wrapper = output / "install-windows-xp" / "environment" / "nel-entrypoint.sh"
        dockerfile_text = dockerfile.read_text()
        wrapper_text = wrapper.read_text()
        assert 'ENTRYPOINT ["/nel-entrypoint.sh"]' in dockerfile_text
        assert "upstream compose passes no command" in dockerfile_text
        assert "supervisord -c /etc/supervisor/supervisord.conf &" in wrapper_text
        assert 'kill -0 "$pid"' in wrapper_text
        assert 'exec "$@"' in wrapper_text

    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._git")
    def test_patches_existing_cached_install_windows_xp_entrypoint(self, mock_git, tmp_path):
        output_dir = tmp_path / "terminal-bench@1.0"
        env_dir = output_dir / "install-windows-xp" / "environment"
        env_dir.mkdir(parents=True)
        (output_dir / "install-windows-xp" / "instruction.md").write_text("Install XP")
        (output_dir / ".tbv1_mapped").touch()
        (env_dir / "Dockerfile").write_text(INSTALL_WINDOWS_XP_DOCKERFILE)

        output = _ensure_dataset(datasets_dir=str(tmp_path))

        assert output == output_dir
        assert mock_git.call_count == 0
        assert 'ENTRYPOINT ["/nel-entrypoint.sh"]' in (env_dir / "Dockerfile").read_text()
        assert 'exec "$@"' in (env_dir / "nel-entrypoint.sh").read_text()

    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._git")
    def test_skips_install_windows_xp_patch_when_entrypoint_changes(self, mock_git, tmp_path):
        output_dir = tmp_path / "terminal-bench@1.0"
        env_dir = output_dir / "install-windows-xp" / "environment"
        env_dir.mkdir(parents=True)
        (output_dir / "install-windows-xp" / "instruction.md").write_text("Install XP")
        (output_dir / ".tbv1_mapped").touch()
        (env_dir / "Dockerfile").write_text('FROM ubuntu:24.04\nENTRYPOINT ["/custom-entrypoint.sh"]\n')

        output = _ensure_dataset(datasets_dir=str(tmp_path))

        assert output == output_dir
        assert mock_git.call_count == 0
        assert not (env_dir / "nel-entrypoint.sh").exists()
        assert 'ENTRYPOINT ["/custom-entrypoint.sh"]' in (env_dir / "Dockerfile").read_text()


class TestRegistration:
    def test_registered_as_builtin(self):
        from nemo_evaluator.environments.registry import _REGISTRY

        assert "terminal-bench-v1" in _REGISTRY
        assert _REGISTRY["terminal-bench-v1"] is TerminalBenchV1

    @patch("nemo_evaluator.benchmarks.terminal_bench_v1._ensure_dataset")
    def test_init_delegates_to_harbor(self, mock_ensure, tmp_path):
        """TerminalBenchV1 is a HarborEnvironment subclass."""
        task_dir = tmp_path / "mapped-task"
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text("Do something")

        mock_ensure.return_value = tmp_path

        env = TerminalBenchV1(num_examples=1)
        assert env.name == tmp_path.name
        mock_ensure.assert_called_once()


class TestTerminalBenchHard:
    def test_registered_as_builtin(self):
        from nemo_evaluator.environments.registry import _REGISTRY

        assert "terminal-bench-hard" in _REGISTRY
        assert _REGISTRY["terminal-bench-hard"] is TerminalBenchHard

    def test_task_list_has_47_entries(self):
        assert len(_TB_HARD_TASKS) == 47

    @patch("nemo_evaluator.benchmarks.terminal_bench_hard._ensure_dataset")
    def test_init_calls_ensure_dataset(self, mock_ensure, tmp_path):
        for name in ["task-a", "task-b"]:
            task_dir = tmp_path / name
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text("Do something")

        mock_ensure.return_value = tmp_path

        TerminalBenchHard()
        mock_ensure.assert_called_once()

    @patch("nemo_evaluator.benchmarks.terminal_bench_hard._ensure_dataset")
    def test_filters_to_known_tasks(self, mock_ensure, tmp_path):
        for name in ["aimo-airline-departures", "unknown-task", "blind-maze-explorer-5x5"]:
            task_dir = tmp_path / name
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text("Do something")

        mock_ensure.return_value = tmp_path

        env = TerminalBenchHard()
        task_names = {t.name for t in env._tasks}
        assert "aimo-airline-departures" in task_names
        assert "blind-maze-explorer-5x5" in task_names
        assert "unknown-task" not in task_names
