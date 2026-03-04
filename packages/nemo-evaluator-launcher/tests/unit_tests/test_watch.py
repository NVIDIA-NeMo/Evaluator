# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the watch mode (eval-and-sleep) feature."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.watch import (
    DEFAULT_CHECKPOINT_PATTERNS,
    DEFAULT_READY_MARKERS,
    SubmittedCheckpoint,
    WatchDirConfig,
    WatchState,
    _load_watch_config,
    _natural_sort_key,
    discover_checkpoints,
    watch_checkpoints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(
    tmp_path: Path, name: str, marker: str = "metadata.json"
) -> Path:
    """Create a checkpoint subdirectory with a ready marker file."""
    cp = tmp_path / name
    cp.mkdir(parents=True, exist_ok=True)
    (cp / marker).write_text("{}")
    return cp


def _sample_config() -> OmegaConf:
    """Return a minimal OmegaConf config for testing."""
    return OmegaConf.create(
        {
            "deployment": {"hf_model_handle": "original-model", "type": "none"},
            "evaluation": {"tasks": [{"name": "test_task"}]},
            "execution": {"type": "local", "output_dir": "/tmp/test_output"},
            "target": {"api_endpoint": {}},
        }
    )


# ---------------------------------------------------------------------------
# Natural sort
# ---------------------------------------------------------------------------


class TestNaturalSort:
    def test_basic_numeric(self):
        names = ["step-10", "step-2", "step-1", "step-20"]
        sorted_names = sorted(names, key=_natural_sort_key)
        assert sorted_names == ["step-1", "step-2", "step-10", "step-20"]

    def test_global_step_format(self):
        names = ["global_step1000", "global_step200", "global_step50"]
        sorted_names = sorted(names, key=_natural_sort_key)
        assert sorted_names == ["global_step50", "global_step200", "global_step1000"]

    def test_mixed_formats(self):
        names = ["checkpoint-100", "checkpoint-20", "checkpoint-3"]
        sorted_names = sorted(names, key=_natural_sort_key)
        assert sorted_names == ["checkpoint-3", "checkpoint-20", "checkpoint-100"]

    def test_no_numbers(self):
        names = ["beta", "alpha", "gamma"]
        sorted_names = sorted(names, key=_natural_sort_key)
        assert sorted_names == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoints:
    def test_finds_ready_checkpoints(self, tmp_path):
        _make_checkpoint(tmp_path, "step_1")
        _make_checkpoint(tmp_path, "step_2")
        # Not ready — no marker
        (tmp_path / "step_3").mkdir()

        result = discover_checkpoints(tmp_path, DEFAULT_READY_MARKERS)
        assert len(result) == 2
        assert result[0].name == "step_1"
        assert result[1].name == "step_2"

    def test_returns_natural_sorted_order(self, tmp_path):
        _make_checkpoint(tmp_path, "step_10")
        _make_checkpoint(tmp_path, "step_2")
        _make_checkpoint(tmp_path, "step_1")

        result = discover_checkpoints(tmp_path, DEFAULT_READY_MARKERS)
        assert [p.name for p in result] == ["step_1", "step_2", "step_10"]

    def test_custom_ready_marker(self, tmp_path):
        cp = tmp_path / "step_1"
        cp.mkdir()
        (cp / "done.txt").write_text("done")

        assert discover_checkpoints(tmp_path, ["metadata.json"]) == []
        assert len(discover_checkpoints(tmp_path, ["done.txt"])) == 1

    def test_ignores_files_in_watch_dir(self, tmp_path):
        (tmp_path / "some_file.txt").write_text("not a checkpoint")
        _make_checkpoint(tmp_path, "step_1")

        result = discover_checkpoints(tmp_path, DEFAULT_READY_MARKERS)
        assert len(result) == 1

    def test_nonexistent_watch_dir(self, tmp_path):
        result = discover_checkpoints(
            tmp_path / "does_not_exist", DEFAULT_READY_MARKERS
        )
        assert result == []

    def test_multi_marker_any_match(self, tmp_path):
        """Checkpoint is ready if ANY marker exists."""
        cp1 = tmp_path / "step_1"
        cp1.mkdir()
        (cp1 / "metadata.json").write_text("{}")

        cp2 = tmp_path / "step_2"
        cp2.mkdir()
        (cp2 / "config.yaml").write_text("")

        cp3 = tmp_path / "step_3"
        cp3.mkdir()
        # No markers

        result = discover_checkpoints(
            tmp_path, ["metadata.json", "config.yaml"]
        )
        assert len(result) == 2
        assert {p.name for p in result} == {"step_1", "step_2"}

    def test_checkpoint_patterns_filter(self, tmp_path):
        """Only dirs matching checkpoint patterns are considered."""
        _make_checkpoint(tmp_path, "iter_100")
        _make_checkpoint(tmp_path, "step_200")
        _make_checkpoint(tmp_path, "random_dir")

        result = discover_checkpoints(
            tmp_path,
            DEFAULT_READY_MARKERS,
            checkpoint_patterns=["iter_*", "step_*"],
        )
        assert len(result) == 2
        assert {p.name for p in result} == {"iter_100", "step_200"}

    def test_checkpoint_patterns_wildcard(self, tmp_path):
        """Pattern '*' matches all subdirectories."""
        _make_checkpoint(tmp_path, "iter_100")
        _make_checkpoint(tmp_path, "random_dir")

        result = discover_checkpoints(
            tmp_path, DEFAULT_READY_MARKERS, checkpoint_patterns=["*"]
        )
        assert len(result) == 2

    def test_checkpoint_patterns_none_matches_all(self, tmp_path):
        """When checkpoint_patterns is None, all subdirs are candidates."""
        _make_checkpoint(tmp_path, "iter_100")
        _make_checkpoint(tmp_path, "any_name")

        result = discover_checkpoints(tmp_path, DEFAULT_READY_MARKERS, checkpoint_patterns=None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# WatchState
# ---------------------------------------------------------------------------


class TestWatchState:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / ".watch_state.yaml"
        state = WatchState(
            submitted=[
                SubmittedCheckpoint(
                    checkpoint="/path/step_1",
                    invocation_id="abc123",
                    timestamp="2026-03-04T21:00:00+00:00",
                    watch_dir="/path/checkpoints",
                ),
                SubmittedCheckpoint(
                    checkpoint="/path/step_2",
                    invocation_id="def456",
                    timestamp="2026-03-04T21:05:00+00:00",
                    watch_dir="/path/checkpoints",
                ),
            ]
        )
        state.save(state_file)

        loaded = WatchState.load(state_file)
        assert len(loaded.submitted) == 2
        assert loaded.submitted[0].checkpoint == "/path/step_1"
        assert loaded.submitted[0].invocation_id == "abc123"
        assert loaded.submitted[0].watch_dir == "/path/checkpoints"
        assert loaded.submitted[1].checkpoint == "/path/step_2"

    def test_load_nonexistent(self, tmp_path):
        state = WatchState.load(tmp_path / "nonexistent.yaml")
        assert state.submitted == []

    def test_load_corrupt_file(self, tmp_path):
        state_file = tmp_path / "corrupt.yaml"
        state_file.write_text(":::: not valid yaml {{{{")
        state = WatchState.load(state_file)
        assert state.submitted == []

    def test_submitted_paths(self):
        state = WatchState(
            submitted=[
                SubmittedCheckpoint(
                    checkpoint="/a", invocation_id="x", timestamp=""
                ),
                SubmittedCheckpoint(
                    checkpoint="/b", invocation_id="y", timestamp=""
                ),
            ]
        )
        assert state.submitted_paths() == {"/a", "/b"}

    def test_save_creates_parent_dirs(self, tmp_path):
        state_file = tmp_path / "deep" / "nested" / "state.yaml"
        state = WatchState()
        state.save(state_file)
        assert state_file.exists()

    def test_save_omits_none_watch_dir(self, tmp_path):
        """When watch_dir is None, it should not appear in saved YAML."""
        state_file = tmp_path / "state.yaml"
        state = WatchState(
            submitted=[
                SubmittedCheckpoint(
                    checkpoint="/a", invocation_id="x", timestamp="t"
                ),
            ]
        )
        state.save(state_file)
        data = yaml.safe_load(state_file.read_text())
        assert "watch_dir" not in data["submitted"][0]


# ---------------------------------------------------------------------------
# watch_checkpoints — once mode
# ---------------------------------------------------------------------------


class TestWatchCheckpointsOnce:
    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_submits_new_checkpoints(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-001"
        _make_checkpoint(tmp_path / "checkpoints", "step_1")
        _make_checkpoint(tmp_path / "checkpoints", "step_2")

        config = _sample_config()
        state_file = tmp_path / "state.yaml"

        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            once=True,
            state_file=state_file,
        )

        assert len(submissions) == 2
        assert mock_run_eval.call_count == 2
        assert submissions[0].invocation_id == "inv-001"

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_skips_already_submitted(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-002"
        cp1 = _make_checkpoint(tmp_path / "checkpoints", "step_1")
        _make_checkpoint(tmp_path / "checkpoints", "step_2")

        state_file = tmp_path / "state.yaml"
        # Pre-populate state with step_1
        state = WatchState(
            submitted=[
                SubmittedCheckpoint(
                    checkpoint=str(cp1),
                    invocation_id="old-inv",
                    timestamp="2026-01-01T00:00:00+00:00",
                )
            ]
        )
        state.save(state_file)

        config = _sample_config()
        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            once=True,
            state_file=state_file,
        )

        assert len(submissions) == 1
        assert "step_2" in submissions[0].checkpoint
        assert mock_run_eval.call_count == 1

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_overrides_checkpoint_field(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-003"
        cp = _make_checkpoint(tmp_path / "checkpoints", "step_1")

        config = _sample_config()
        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            checkpoint_field="deployment.hf_model_handle",
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        assert len(submissions) == 1
        call_cfg = mock_run_eval.call_args[0][0]
        assert call_cfg.deployment.hf_model_handle == str(cp)

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_persists_state_after_submission(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-004"
        _make_checkpoint(tmp_path / "checkpoints", "step_1")

        state_file = tmp_path / "state.yaml"
        config = _sample_config()

        watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            once=True,
            state_file=state_file,
        )

        loaded = WatchState.load(state_file)
        assert len(loaded.submitted) == 1
        assert loaded.submitted[0].invocation_id == "inv-004"

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_no_new_checkpoints_returns_empty(self, mock_run_eval, tmp_path):
        config = _sample_config()
        watch_dir = tmp_path / "checkpoints"
        watch_dir.mkdir()

        submissions = watch_checkpoints(
            config=config,
            watch_dir=watch_dir,
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        assert submissions == []
        mock_run_eval.assert_not_called()

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_handles_submission_failure(self, mock_run_eval, tmp_path):
        mock_run_eval.side_effect = RuntimeError("API error")
        _make_checkpoint(tmp_path / "checkpoints", "step_1")
        _make_checkpoint(tmp_path / "checkpoints", "step_2")

        config = _sample_config()
        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        assert len(submissions) == 0
        assert mock_run_eval.call_count == 2

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_state_includes_watch_dir(self, mock_run_eval, tmp_path):
        """State records include the watch_dir field."""
        mock_run_eval.return_value = "inv-005"
        _make_checkpoint(tmp_path / "checkpoints", "step_1")

        config = _sample_config()
        state_file = tmp_path / "state.yaml"

        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            once=True,
            state_file=state_file,
        )

        assert len(submissions) == 1
        assert submissions[0].watch_dir == str(tmp_path / "checkpoints")

        loaded = WatchState.load(state_file)
        assert loaded.submitted[0].watch_dir == str(tmp_path / "checkpoints")


# ---------------------------------------------------------------------------
# watch_checkpoints — dry-run mode
# ---------------------------------------------------------------------------


class TestWatchCheckpointsDryRun:
    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_dry_run_does_not_call_run_eval(self, mock_run_eval, tmp_path):
        _make_checkpoint(tmp_path / "checkpoints", "step_1")

        config = _sample_config()
        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            dry_run=True,
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        assert len(submissions) == 1
        assert submissions[0].invocation_id is None
        mock_run_eval.assert_not_called()

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_dry_run_does_not_persist_state(self, mock_run_eval, tmp_path):
        _make_checkpoint(tmp_path / "checkpoints", "step_1")

        state_file = tmp_path / "state.yaml"
        config = _sample_config()

        watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            dry_run=True,
            once=True,
            state_file=state_file,
        )

        assert not state_file.exists()


# ---------------------------------------------------------------------------
# Processing order
# ---------------------------------------------------------------------------


class TestProcessingOrder:
    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_newest_first(self, mock_run_eval, tmp_path):
        """With order='newest', highest step numbers are processed first."""
        mock_run_eval.return_value = "inv"
        _make_checkpoint(tmp_path / "checkpoints", "step_1")
        _make_checkpoint(tmp_path / "checkpoints", "step_10")
        _make_checkpoint(tmp_path / "checkpoints", "step_2")

        config = _sample_config()
        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            order="newest",
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        names = [Path(s.checkpoint).name for s in submissions]
        assert names == ["step_10", "step_2", "step_1"]

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_oldest_first(self, mock_run_eval, tmp_path):
        """With order='oldest', lowest step numbers are processed first."""
        mock_run_eval.return_value = "inv"
        _make_checkpoint(tmp_path / "checkpoints", "step_1")
        _make_checkpoint(tmp_path / "checkpoints", "step_10")
        _make_checkpoint(tmp_path / "checkpoints", "step_2")

        config = _sample_config()
        submissions = watch_checkpoints(
            config=config,
            watch_dir=tmp_path / "checkpoints",
            order="oldest",
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        names = [Path(s.checkpoint).name for s in submissions]
        assert names == ["step_1", "step_2", "step_10"]


# ---------------------------------------------------------------------------
# Watch config file (multi-dir)
# ---------------------------------------------------------------------------


class TestWatchConfig:
    def test_load_watch_config(self, tmp_path):
        config_file = tmp_path / "watch.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "watch_dirs": [
                        {
                            "checkpoint_dir": "/lustre/run-42/checkpoints",
                            "output_dir": "/lustre/run-42/evals",
                            "checkpoint_field": "deployment.hf_model_handle",
                        },
                        {
                            "checkpoint_dir": "/lustre/run-43/checkpoints",
                            "output_dir": "/lustre/run-43/evals",
                        },
                    ],
                    "checkpoint_field": "deployment.hf_model_handle",
                }
            )
        )

        watch_dirs, global_field = _load_watch_config(config_file)
        assert len(watch_dirs) == 2
        assert watch_dirs[0].checkpoint_dir == Path("/lustre/run-42/checkpoints")
        assert watch_dirs[0].output_dir == Path("/lustre/run-42/evals")
        assert watch_dirs[0].checkpoint_field == "deployment.hf_model_handle"
        assert watch_dirs[1].checkpoint_dir == Path("/lustre/run-43/checkpoints")
        assert watch_dirs[1].checkpoint_field is None
        assert global_field == "deployment.hf_model_handle"

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_multi_dir_watches_all_dirs(self, mock_run_eval, tmp_path):
        """Multi-dir mode discovers checkpoints from all dirs."""
        mock_run_eval.return_value = "inv-multi"

        dir1 = tmp_path / "run1" / "checkpoints"
        dir2 = tmp_path / "run2" / "checkpoints"
        out1 = tmp_path / "run1" / "evals"
        out2 = tmp_path / "run2" / "evals"
        out1.mkdir(parents=True)
        out2.mkdir(parents=True)

        _make_checkpoint(dir1, "step_100")
        _make_checkpoint(dir2, "step_200")

        config = _sample_config()
        state_file = tmp_path / "state.yaml"

        submissions = watch_checkpoints(
            config=config,
            watch_dirs=[
                WatchDirConfig(checkpoint_dir=dir1, output_dir=out1),
                WatchDirConfig(checkpoint_dir=dir2, output_dir=out2),
            ],
            once=True,
            state_file=state_file,
        )

        assert len(submissions) == 2
        assert mock_run_eval.call_count == 2
        checkpoint_names = {Path(s.checkpoint).name for s in submissions}
        assert checkpoint_names == {"step_100", "step_200"}

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_multi_dir_per_dir_checkpoint_field(self, mock_run_eval, tmp_path):
        """Per-dir checkpoint_field overrides the global one."""
        mock_run_eval.return_value = "inv"

        dir1 = tmp_path / "checkpoints1"
        _make_checkpoint(dir1, "step_1")

        config = _sample_config()

        submissions = watch_checkpoints(
            config=config,
            watch_dirs=[
                WatchDirConfig(
                    checkpoint_dir=dir1,
                    checkpoint_field="deployment.type",
                ),
            ],
            checkpoint_field="deployment.hf_model_handle",
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        assert len(submissions) == 1
        call_cfg = mock_run_eval.call_args[0][0]
        # The per-dir field should have been used
        assert call_cfg.deployment.type == str(dir1 / "step_1")

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_multi_dir_state_tracks_watch_dir(self, mock_run_eval, tmp_path):
        """State records track which watch_dir each submission came from."""
        mock_run_eval.return_value = "inv"

        dir1 = tmp_path / "run1" / "checkpoints"
        dir2 = tmp_path / "run2" / "checkpoints"
        _make_checkpoint(dir1, "step_1")
        _make_checkpoint(dir2, "step_2")

        config = _sample_config()
        state_file = tmp_path / "state.yaml"

        submissions = watch_checkpoints(
            config=config,
            watch_dirs=[
                WatchDirConfig(checkpoint_dir=dir1),
                WatchDirConfig(checkpoint_dir=dir2),
            ],
            once=True,
            state_file=state_file,
        )

        assert submissions[0].watch_dir == str(dir1)
        assert submissions[1].watch_dir == str(dir2)

    @patch("nemo_evaluator_launcher.api.watch.run_eval")
    def test_multi_dir_output_dir_override(self, mock_run_eval, tmp_path):
        """When a watch_dir entry has output_dir, it overrides config output_dir."""
        mock_run_eval.return_value = "inv"

        dir1 = tmp_path / "checkpoints"
        out1 = tmp_path / "custom_output"
        _make_checkpoint(dir1, "step_1")

        config = _sample_config()

        watch_checkpoints(
            config=config,
            watch_dirs=[
                WatchDirConfig(checkpoint_dir=dir1, output_dir=out1),
            ],
            once=True,
            state_file=tmp_path / "state.yaml",
        )

        call_cfg = mock_run_eval.call_args[0][0]
        assert call_cfg.execution.output_dir == str(out1)


# ---------------------------------------------------------------------------
# Tmux support
# ---------------------------------------------------------------------------


class TestTmuxSupport:
    def test_build_watch_command(self):
        """Verify command construction for tmux wrapping."""
        from nemo_evaluator_launcher.cli.watch import Cmd

        cmd = Cmd(
            config="/path/to/config.yaml",
            watch_dir="/path/to/checkpoints",
            interval=300,
            ready_markers="metadata.json,config.yaml",
            checkpoint_patterns="iter_*,step_*",
            checkpoint_field="deployment.hf_model_handle",
            order="newest",
            dry_run=True,
            once=False,
        )
        result = cmd._build_watch_command()
        assert "--config" in result
        assert "/path/to/config.yaml" in result
        assert "--watch-dir" in result
        assert "/path/to/checkpoints" in result
        assert "--dry-run" in result
        assert "--tmux" not in result
        assert "--order" in result
        assert "newest" in result

    @patch("nemo_evaluator_launcher.cli.watch.subprocess.run")
    @patch("nemo_evaluator_launcher.cli.watch.shutil.which", return_value="/usr/bin/tmux")
    def test_launch_in_tmux(self, mock_which, mock_run, tmp_path):
        """Verify tmux session launch."""
        from nemo_evaluator_launcher.cli.watch import Cmd

        # First call: has-session (session doesn't exist, returns 1)
        # Second call: new-session
        mock_run.side_effect = [
            MagicMock(returncode=1),  # has-session fails = no existing session
            MagicMock(returncode=0),  # new-session succeeds
        ]

        cmd = Cmd(
            config="/path/config.yaml",
            watch_dir="/path/checkpoints",
            tmux="my-session",
        )
        cmd._launch_in_tmux()

        assert mock_run.call_count == 2
        # Check has-session call
        has_session_call = mock_run.call_args_list[0]
        assert has_session_call[0][0] == ["tmux", "has-session", "-t", "my-session"]
        # Check new-session call
        new_session_call = mock_run.call_args_list[1]
        assert new_session_call[0][0][0:4] == [
            "tmux",
            "new-session",
            "-d",
            "-s",
        ]
        assert new_session_call[0][0][4] == "my-session"

    @patch("nemo_evaluator_launcher.cli.watch.shutil.which", return_value=None)
    def test_tmux_not_installed(self, mock_which):
        """Error when tmux is not available."""
        from nemo_evaluator_launcher.cli.watch import Cmd

        cmd = Cmd(config="/path/config.yaml", watch_dir="/path/checkpoints", tmux="s")
        with pytest.raises(RuntimeError, match="tmux is not installed"):
            cmd._launch_in_tmux()

    @patch("nemo_evaluator_launcher.cli.watch.subprocess.run")
    @patch("nemo_evaluator_launcher.cli.watch.shutil.which", return_value="/usr/bin/tmux")
    def test_tmux_session_already_exists(self, mock_which, mock_run):
        """Warn when tmux session already exists."""
        from nemo_evaluator_launcher.cli.watch import Cmd

        mock_run.return_value = MagicMock(returncode=0)  # has-session succeeds

        cmd = Cmd(
            config="/path/config.yaml",
            watch_dir="/path/checkpoints",
            tmux="existing",
        )
        cmd._launch_in_tmux()

        # Should only call has-session, not new-session
        assert mock_run.call_count == 1

    def test_tmux_auto_session_name_from_config(self):
        """Auto-generated session name uses config stem."""
        from nemo_evaluator_launcher.cli.watch import Cmd

        cmd = Cmd(
            config="/path/to/my_eval.yaml",
            watch_dir="/path/checkpoints",
            tmux="",  # empty string = auto-generate
        )
        # When tmux is falsy but not None, auto-generate from config
        # The actual name generation happens in _launch_in_tmux
        # We test that the stem-based name is generated
        with patch("nemo_evaluator_launcher.cli.watch.shutil.which", return_value="/usr/bin/tmux"), \
             patch("nemo_evaluator_launcher.cli.watch.subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=1),
                MagicMock(returncode=0),
            ]
            cmd._launch_in_tmux()
            new_session_call = mock_run.call_args_list[1]
            session_name = new_session_call[0][0][4]
            assert session_name == "nel-watch-my_eval"


# ---------------------------------------------------------------------------
# Checkpoint patterns with discover_checkpoints
# ---------------------------------------------------------------------------


class TestCheckpointPatterns:
    def test_default_patterns(self, tmp_path):
        """Default patterns match iter_* and step_*."""
        _make_checkpoint(tmp_path, "iter_100")
        _make_checkpoint(tmp_path, "step_200")
        _make_checkpoint(tmp_path, "checkpoint_300")

        result = discover_checkpoints(
            tmp_path,
            DEFAULT_READY_MARKERS,
            checkpoint_patterns=DEFAULT_CHECKPOINT_PATTERNS,
        )
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"iter_100", "step_200"}

    def test_custom_pattern(self, tmp_path):
        """Custom pattern filters appropriately."""
        _make_checkpoint(tmp_path, "ckpt-100")
        _make_checkpoint(tmp_path, "ckpt-200")
        _make_checkpoint(tmp_path, "step_300")

        result = discover_checkpoints(
            tmp_path,
            DEFAULT_READY_MARKERS,
            checkpoint_patterns=["ckpt-*"],
        )
        assert len(result) == 2
        assert all("ckpt-" in p.name for p in result)

    def test_multiple_patterns(self, tmp_path):
        """Multiple patterns match different naming conventions."""
        _make_checkpoint(tmp_path, "iter_100")
        _make_checkpoint(tmp_path, "ckpt-200")
        _make_checkpoint(tmp_path, "other_300")

        result = discover_checkpoints(
            tmp_path,
            DEFAULT_READY_MARKERS,
            checkpoint_patterns=["iter_*", "ckpt-*"],
        )
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"iter_100", "ckpt-200"}
