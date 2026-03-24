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
"""Unit tests for the watcher module (nel-watch feature)."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.watcher.configs import (
    CHECKPOINT_FIELD,
    DEFAULT_CHECKPOINT_PATTERNS,
    DEFAULT_READY_MARKERS,
    ClusterConfig,
    ConversionConfig,
    MonitoringConfig,
    WatchConfig,
)
from nemo_evaluator_launcher.watcher.run import (
    _natural_sort_key,
    discover_checkpoints,
    watch_and_evaluate,
)
from nemo_evaluator_launcher.watcher.watchdb import SubmittedCheckpoint, WatchStateDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster_config(**kwargs) -> ClusterConfig:
    defaults = {
        "username": "testuser",
        "account": "test-account",
        "output_dir": "/tmp/out",
    }
    return ClusterConfig(**{**defaults, **kwargs})


def _make_eval_config(**kwargs) -> RunConfig:
    """Create a minimal RunConfig with SLURM execution for WatchConfig tests."""
    data = {
        "deployment": {"checkpoint_path": None},
        "execution": {"type": "slurm"},
        **kwargs,
    }
    return RunConfig(OmegaConf.create(data))


def _make_watch_config(
    directories: list[str] | None = None,
    output_dir: str = "/tmp/out",
) -> WatchConfig:
    """Return a minimal valid WatchConfig."""
    return WatchConfig.model_construct(
        cluster_config=_make_cluster_config(output_dir=output_dir),
        monitoring_config=MonitoringConfig(
            directories=directories or ["/checkpoints"], interval=None
        ),
        evaluation_configs=[_make_eval_config()],
        conversion_config=None,
    )


def _make_submitted(
    checkpoint: str = "/path/step_1", invocation_id: str = "inv-001"
) -> SubmittedCheckpoint:
    return SubmittedCheckpoint(
        checkpoint=checkpoint,
        session_id="session-xyz",
        invocation_id=invocation_id,
        timestamp="2026-03-04T21:00:00+00:00",
        watch_config={},
        eval_config={},
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
# discover_checkpoints (SSH-based)
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoints:
    @patch("nemo_evaluator_launcher.watcher.run.run_remote_command")
    def test_returns_sorted_paths(self, mock_cmd):
        mock_cmd.return_value = (
            "/checkpoints/step_10\n/checkpoints/step_2\n/checkpoints/step_1\n"
        )
        cluster = _make_cluster_config()
        result = discover_checkpoints(
            Path("/checkpoints"),
            cluster,
            DEFAULT_READY_MARKERS,
            DEFAULT_CHECKPOINT_PATTERNS,
        )
        assert [p.name for p in result] == ["step_1", "step_2", "step_10"]

    @patch("nemo_evaluator_launcher.watcher.run.run_remote_command")
    def test_returns_empty_on_failure(self, mock_cmd):
        mock_cmd.side_effect = RuntimeError("SSH error")
        cluster = _make_cluster_config()
        result = discover_checkpoints(
            Path("/checkpoints"),
            cluster,
            DEFAULT_READY_MARKERS,
            DEFAULT_CHECKPOINT_PATTERNS,
        )
        assert result == []

    @patch("nemo_evaluator_launcher.watcher.run.run_remote_command")
    def test_returns_empty_when_no_output(self, mock_cmd):
        mock_cmd.return_value = ""
        cluster = _make_cluster_config()
        result = discover_checkpoints(
            Path("/checkpoints"),
            cluster,
            DEFAULT_READY_MARKERS,
            DEFAULT_CHECKPOINT_PATTERNS,
        )
        assert result == []

    @patch("nemo_evaluator_launcher.watcher.run.run_remote_command")
    def test_passes_socket_to_remote_command(self, mock_cmd):
        mock_cmd.return_value = ""
        cluster = _make_cluster_config()
        discover_checkpoints(
            Path("/checkpoints"),
            cluster,
            DEFAULT_READY_MARKERS,
            DEFAULT_CHECKPOINT_PATTERNS,
            socket="/tmp/sock",
        )
        call_kwargs = mock_cmd.call_args[1]
        assert call_kwargs["socket"] == "/tmp/sock"

    @patch("nemo_evaluator_launcher.watcher.run.run_remote_command")
    def test_command_includes_checkpoint_patterns(self, mock_cmd):
        mock_cmd.return_value = ""
        cluster = _make_cluster_config()
        discover_checkpoints(
            Path("/ckpts"),
            cluster,
            DEFAULT_READY_MARKERS,
            ["iter_*", "step_*"],
        )
        cmd = mock_cmd.call_args[1]["command"]
        assert "iter_*" in cmd
        assert "step_*" in cmd

    @patch("nemo_evaluator_launcher.watcher.run.run_remote_command")
    def test_command_includes_ready_markers(self, mock_cmd):
        mock_cmd.return_value = ""
        cluster = _make_cluster_config()
        discover_checkpoints(
            Path("/ckpts"),
            cluster,
            ["metadata.json", "config.yaml"],
            DEFAULT_CHECKPOINT_PATTERNS,
        )
        cmd = mock_cmd.call_args[1]["command"]
        assert "metadata.json" in cmd
        assert "config.yaml" in cmd


# ---------------------------------------------------------------------------
# WatchStateDB
# ---------------------------------------------------------------------------


class TestWatchStateDB:
    def test_append_and_submitted_paths(self, tmp_path, monkeypatch):
        state_file = tmp_path / "watch-state.v1.jsonl"
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db = WatchStateDB()
        db.append(_make_submitted("/path/step_1", "inv-001"))
        db.append(_make_submitted("/path/step_2", "inv-002"))

        assert db.submitted_paths() == {"/path/step_1", "/path/step_2"}

    def test_persists_across_instances(self, tmp_path, monkeypatch):
        state_file = tmp_path / "watch-state.v1.jsonl"
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db1 = WatchStateDB()
        db1.append(_make_submitted("/path/step_1", "inv-001"))

        db2 = WatchStateDB()
        assert "/path/step_1" in db2.submitted_paths()

    def test_submitted_paths_returns_empty_when_no_records(self, tmp_path, monkeypatch):
        state_file = tmp_path / "watch-state.v1.jsonl"
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db = WatchStateDB()
        assert db.submitted_paths() == set()

    def test_submitted_paths_filters_by_session_id(self, tmp_path, monkeypatch):
        state_file = tmp_path / "watch-state.v1.jsonl"
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db = WatchStateDB()
        rec1 = SubmittedCheckpoint(
            checkpoint="/a",
            session_id="session-A",
            invocation_id="inv-1",
            timestamp="t",
            watch_config={},
            eval_config={},
        )
        rec2 = SubmittedCheckpoint(
            checkpoint="/b",
            session_id="session-B",
            invocation_id="inv-2",
            timestamp="t",
            watch_config={},
            eval_config={},
        )
        db.append(rec1)
        db.append(rec2)

        assert db.submitted_paths(session_ids=["session-A"]) == {"/a"}
        assert db.submitted_paths(session_ids=["session-B"]) == {"/b"}
        assert db.submitted_paths(session_ids=None) == {"/a", "/b"}

    def test_submitted_paths_unfiltered_returns_all_sessions(
        self, tmp_path, monkeypatch
    ):
        state_file = tmp_path / "watch-state.v1.jsonl"
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db = WatchStateDB()
        db.append(_make_submitted("/a", "inv-1"))
        db.append(_make_submitted("/b", "inv-2"))

        assert db.submitted_paths() == {"/a", "/b"}

    def test_tolerates_corrupt_lines(self, tmp_path, monkeypatch):
        state_file = tmp_path / "watch-state.v1.jsonl"
        state_file.write_text("not-json\n")
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db = WatchStateDB()
        assert db.submitted_paths() == set()

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        state_file = tmp_path / "deep" / "nested" / "watch-state.v1.jsonl"
        monkeypatch.setattr(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        )

        db = WatchStateDB()
        db.append(_make_submitted())
        assert state_file.exists()


# ---------------------------------------------------------------------------
# SubmittedCheckpoint
# ---------------------------------------------------------------------------


class TestSubmittedCheckpoint:
    def test_valid_construction(self):
        rec = _make_submitted()
        assert rec.checkpoint == "/path/step_1"
        assert rec.invocation_id == "inv-001"
        assert rec.session_id == "session-xyz"

    def test_rejects_extra_fields(self):
        with pytest.raises(Exception):
            SubmittedCheckpoint(
                checkpoint="/a",
                session_id="s",
                invocation_id="i",
                timestamp="t",
                watch_config={},
                eval_config={},
                unknown_field="oops",
            )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestMonitoringConfig:
    def test_default_values(self):
        cfg = MonitoringConfig(directories=["/checkpoints"])
        assert cfg.interval == 300
        assert cfg.ready_markers == list(DEFAULT_READY_MARKERS)
        assert cfg.checkpoint_patterns == list(DEFAULT_CHECKPOINT_PATTERNS)
        assert cfg.order == "last"

    def test_interval_none_is_valid(self):
        cfg = MonitoringConfig(directories=["/checkpoints"], interval=None)
        assert cfg.interval is None

    def test_interval_zero_is_invalid(self):
        with pytest.raises(Exception):
            MonitoringConfig(directories=["/checkpoints"], interval=0)

    def test_interval_negative_is_invalid(self):
        with pytest.raises(Exception):
            MonitoringConfig(directories=["/checkpoints"], interval=-1)

    def test_empty_directories_is_invalid(self):
        with pytest.raises(Exception):
            MonitoringConfig(directories=[])

    def test_empty_ready_markers_is_invalid(self):
        with pytest.raises(Exception):
            MonitoringConfig(directories=["/checkpoints"], ready_markers=[])

    def test_empty_checkpoint_patterns_is_invalid(self):
        with pytest.raises(Exception):
            MonitoringConfig(directories=["/checkpoints"], checkpoint_patterns=[])


class TestConversionConfig:
    def test_render_command_basic(self):
        cfg = ConversionConfig(
            container="nvcr.io/nemo:latest",
            command_pattern="convert {{ input_path }} {{ output_path }}",
        )
        result = cfg.render_command("/in/ckpt", "/out/ckpt")
        assert result == "convert /in/ckpt /out/ckpt"

    def test_render_command_with_extra_params(self):
        cfg = ConversionConfig(
            container="nvcr.io/nemo:latest",
            command_pattern="convert {{ input_path }} {{ output_path }} --dtype {{ dtype }}",
            command_params={"dtype": "bfloat16"},
        )
        result = cfg.render_command("/in", "/out")
        assert "--dtype bfloat16" in result

    def test_missing_input_placeholder_raises(self):
        with pytest.raises(Exception, match="input_path"):
            ConversionConfig(
                container="img",
                command_pattern="convert {{ output_path }}",
            )

    def test_missing_output_placeholder_raises(self):
        with pytest.raises(Exception, match="output_path"):
            ConversionConfig(
                container="img",
                command_pattern="convert {{ input_path }}",
            )


class TestWatchConfigValidation:
    def test_valid_config(self):
        cfg = WatchConfig(
            cluster_config=_make_cluster_config(),
            monitoring_config=MonitoringConfig(directories=["/checkpoints"]),
            evaluation_configs=[_make_eval_config()],
        )
        assert cfg.conversion_config is None

    def test_missing_deployment_raises(self):
        eval_cfg = RunConfig(OmegaConf.create({"execution": {"type": "slurm"}}))
        with pytest.raises(Exception, match="deployment"):
            WatchConfig(
                cluster_config=_make_cluster_config(),
                monitoring_config=MonitoringConfig(directories=["/checkpoints"]),
                evaluation_configs=[eval_cfg],
            )

    def test_missing_execution_raises(self):
        eval_cfg = RunConfig(OmegaConf.create({"deployment": {}}))
        with pytest.raises(Exception, match="execution"):
            WatchConfig(
                cluster_config=_make_cluster_config(),
                monitoring_config=MonitoringConfig(directories=["/checkpoints"]),
                evaluation_configs=[eval_cfg],
            )

    def test_non_slurm_execution_raises(self):
        eval_cfg = RunConfig(
            OmegaConf.create({"deployment": {}, "execution": {"type": "local"}})
        )
        with pytest.raises(Exception, match="Only SLURM execution is supported"):
            WatchConfig(
                cluster_config=_make_cluster_config(),
                monitoring_config=MonitoringConfig(directories=["/checkpoints"]),
                evaluation_configs=[eval_cfg],
            )


# ---------------------------------------------------------------------------
# watch_and_evaluate
# ---------------------------------------------------------------------------


@contextmanager
def _noop_master_connection(**kwargs):
    yield "fake-socket"


class TestWatchAndEvaluate:
    def _patch_all(self, mock_run_eval, tmp_state_file, discovered_checkpoints):
        return [
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=discovered_checkpoints,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_state_file,
            ),
        ]

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_dry_run_does_not_call_run_eval(self, mock_run_eval, tmp_path):
        checkpoint = Path("/checkpoints/step_1")
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=[checkpoint],
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config, dry_run=True)

        assert mock_run_eval.called
        assert any(
            call.kwargs.get("dry_run") is True for call in mock_run_eval.call_args_list
        )
        assert result == []

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_submits_new_checkpoints(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-001"
        checkpoints = [Path("/checkpoints/step_1"), Path("/checkpoints/step_2")]
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=checkpoints,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config)

        assert mock_run_eval.call_count == 2
        assert len(result) == 2

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_skips_already_submitted_checkpoints(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-new"
        state_file = tmp_path / "state.jsonl"

        # Pre-populate state with step_1 already submitted
        from nemo_evaluator_launcher.watcher.watchdb import WatchStateDB

        with patch(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        ):
            db = WatchStateDB()
            db.append(
                SubmittedCheckpoint(
                    checkpoint="/checkpoints/step_1",
                    session_id="old-session",
                    invocation_id="inv-old",
                    timestamp="t",
                    watch_config={},
                    eval_config={},
                )
            )

        checkpoints = [Path("/checkpoints/step_1"), Path("/checkpoints/step_2")]
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=checkpoints,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
            ),
        ):
            result = watch_and_evaluate(watch_config)

        assert mock_run_eval.call_count == 1
        assert result[0].checkpoint == "/checkpoints/step_2"

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_no_checkpoints_returns_empty(self, mock_run_eval, tmp_path):
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=[],
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config)

        assert result == []
        mock_run_eval.assert_not_called()

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_run_eval_failure_does_not_add_to_results(self, mock_run_eval, tmp_path):
        mock_run_eval.side_effect = RuntimeError("API error")
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=[Path("/checkpoints/step_1")],
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config)

        assert result == []

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_order_last_processes_newest_first(self, mock_run_eval, tmp_path):
        """With order='last', highest step name is processed first."""
        mock_run_eval.return_value = "inv"
        # discover_checkpoints returns paths in natural (ascending) order
        checkpoints = [
            Path("/checkpoints/step_1"),
            Path("/checkpoints/step_2"),
            Path("/checkpoints/step_10"),
        ]
        watch_config = _make_watch_config(directories=["/checkpoints"])
        watch_config.monitoring_config.order = "last"

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=checkpoints,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config)

        submitted_names = [Path(r.checkpoint).name for r in result]
        assert submitted_names == ["step_10", "step_2", "step_1"]

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_order_first_processes_oldest_first(self, mock_run_eval, tmp_path):
        """With order='first', lowest step name is processed first."""
        mock_run_eval.return_value = "inv"
        checkpoints = [
            Path("/checkpoints/step_1"),
            Path("/checkpoints/step_2"),
            Path("/checkpoints/step_10"),
        ]
        watch_config = _make_watch_config(directories=["/checkpoints"])
        watch_config.monitoring_config.order = "first"

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=checkpoints,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config)

        submitted_names = [Path(r.checkpoint).name for r in result]
        assert submitted_names == ["step_1", "step_2", "step_10"]

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_checkpoint_field_is_set_in_eval_config(self, mock_run_eval, tmp_path):
        """The discovered checkpoint path is set in evaluation config before submission."""
        mock_run_eval.return_value = "inv-001"
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=[Path("/checkpoints/step_1")],
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            watch_and_evaluate(watch_config)

        call_cfg = mock_run_eval.call_args[0][0]
        assert OmegaConf.select(call_cfg, CHECKPOINT_FIELD) == "/checkpoints/step_1"

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_persists_state_after_submission(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv-persist"
        state_file = tmp_path / "state.jsonl"
        watch_config = _make_watch_config(directories=["/checkpoints"])

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                return_value=[Path("/checkpoints/step_1")],
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
            ),
        ):
            watch_and_evaluate(watch_config)

        assert state_file.exists()
        with patch(
            "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE", state_file
        ):
            db = WatchStateDB()
        assert "/checkpoints/step_1" in db.submitted_paths()

    @patch("nemo_evaluator_launcher.watcher.run.run_eval")
    def test_watches_multiple_directories(self, mock_run_eval, tmp_path):
        mock_run_eval.return_value = "inv"
        watch_config = _make_watch_config(
            directories=["/run1/checkpoints", "/run2/checkpoints"]
        )

        call_count = [0]

        def fake_discover(watch_dir, *args, **kwargs):
            call_count[0] += 1
            if "run1" in str(watch_dir):
                return [Path("/run1/checkpoints/step_1")]
            return [Path("/run2/checkpoints/step_2")]

        with (
            patch(
                "nemo_evaluator_launcher.watcher.run.master_connection",
                side_effect=_noop_master_connection,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.run.discover_checkpoints",
                side_effect=fake_discover,
            ),
            patch(
                "nemo_evaluator_launcher.watcher.watchdb.WATCH_STATE_FILE",
                tmp_path / "state.jsonl",
            ),
        ):
            result = watch_and_evaluate(watch_config)

        assert call_count[0] == 2
        assert mock_run_eval.call_count == 2
        checkpoint_names = {Path(r.checkpoint).name for r in result}
        assert checkpoint_names == {"step_1", "step_2"}
