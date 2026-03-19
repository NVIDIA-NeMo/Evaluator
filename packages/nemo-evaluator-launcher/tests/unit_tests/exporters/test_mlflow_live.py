# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Tests for the standalone MLflow live export script (exporters/mlflow_live.py).

All mlflow calls are mocked — no network, no real MLflow server.
"""

import sys
import unittest.mock
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure mlflow is mocked before importing export module
if "mlflow" not in sys.modules:
    sys.modules["mlflow"] = MagicMock()

import nemo_evaluator_launcher.exporters.mlflow_live as export_mod
from nemo_evaluator_launcher.exporters.mlflow_live import (
    _cancel_mlflow_run,
    _check_existing_run,
    _extract_metrics_from_results_yml,
    _extract_task_metrics,
    _finalize_mlflow_run,
    _init_mlflow_run,
    _load_config,
    _log_artifacts,
    _log_logs,
    _sanitize_metric_key,
    _should_exclude,
    _update_mlflow_run,
)

SAMPLE_CONFIG = {
    "experiment_name": "test-experiment",
    "run_name": "test-run",
    "description": "test description",
    "tags": {
        "invocation_id": "abc123",
        "job_id": "abc123.0",
        "task_name": "mteb.FiQA2018",
        "benchmark": "mteb.FiQA2018",
        "harness": "mteb",
        "executor": "slurm",
        "cluster": "test-cluster",
    },
    "skip_existing": True,
    "log_artifacts": True,
    "log_logs": True,
}

SAMPLE_RESULTS_YML = {
    "results": {
        "tasks": {
            "nq": {
                "metrics": {
                    "ndcg_at_10": {
                        "scores": {
                            "ndcg_at_10": {"value": 0.85},
                            "recall": {"value": 0.72},
                        },
                        "stats": {"stderr": 0.02, "num_samples": 100},
                    },
                    "mrr": {"scores": {"mrr": {"value": 0.91}}},
                }
            }
        },
        "groups": {
            "MTEB": {
                "metrics": {
                    "avg": {"scores": {"avg": {"value": 0.80}}},
                },
                "groups": {
                    "retrieval": {
                        "metrics": {
                            "avg": {"scores": {"avg": {"value": 0.75}}},
                        }
                    }
                },
            }
        },
    }
}


@pytest.fixture()
def sample_results_dir(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "results.yml").write_text(
        yaml.dump(SAMPLE_RESULTS_YML), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def config_dir(tmp_path):
    """Write export_config.yml with SAMPLE_CONFIG."""
    (tmp_path / "export_config.yml").write_text(
        yaml.safe_dump(
            {
                "tracking_uri": "http://mlflow-test:5000",
                "experiment_name": "test-experiment",
                "description": "test description",
                "log_artifacts": True,
                "log_logs": True,
                "skip_existing": False,
                "tags": {
                    "invocation_id": "abc123",
                    "job_id": "abc123.0",
                    "task_name": "mteb.FiQA2018",
                    "benchmark": "mteb.FiQA2018",
                    "harness": "mteb",
                    "executor": "slurm",
                    "cluster": "test-cluster",
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


# --- _sanitize_metric_key ---


def test_sanitize_metric_key():
    assert _sanitize_metric_key("gpqa_majority@2") == "gpqa_majority_2"
    assert _sanitize_metric_key("normal_key") == "normal_key"
    assert _sanitize_metric_key("a/b:c-d.e f") == "a/b:c-d.e f"


# --- _should_exclude ---


def test_should_exclude_cache():
    assert _should_exclude("cache")
    assert _should_exclude("response_stats_cache")
    assert _should_exclude("lm_cache_rank0")


def test_should_exclude_extensions():
    assert _should_exclude("data.db")
    assert _should_exclude("file.lock")


def test_should_exclude_files():
    assert _should_exclude("debug.json")
    assert _should_exclude("synthetic")


def test_should_not_skip_artifact():
    assert not _should_exclude("results.yml")
    assert not _should_exclude("metadata.yaml")
    assert not _should_exclude("run_config.yml")


# --- _extract_task_metrics ---


def test_extract_task_metrics_simple():
    task_data = {"metrics": {"ndcg_at_10": {"scores": {"ndcg_at_10": {"value": 0.85}}}}}
    result = _extract_task_metrics("nq", task_data)
    assert result == {"nq_ndcg_at_10": 0.85}


def test_extract_task_metrics_different_score_type():
    task_data = {"metrics": {"ndcg_at_10": {"scores": {"recall": {"value": 0.72}}}}}
    result = _extract_task_metrics("nq", task_data)
    assert result == {"nq_ndcg_at_10_recall": 0.72}


def test_extract_task_metrics_with_stats():
    task_data = {
        "metrics": {
            "accuracy": {
                "scores": {"accuracy": {"value": 0.9}},
                "stats": {"stderr": 0.02, "num_samples": 100},
            }
        }
    }
    result = _extract_task_metrics("task", task_data)
    assert result["task_accuracy"] == 0.9
    assert result["task_accuracy_stderr"] == 0.02
    assert result["task_accuracy_num_samples"] == 100.0


def test_extract_task_metrics_nested_groups():
    task_data = {
        "metrics": {},
        "groups": {"sub": {"metrics": {"f1": {"scores": {"f1": {"value": 0.9}}}}}},
    }
    result = _extract_task_metrics("task", task_data)
    assert result == {"task_sub_f1": 0.9}


def test_extract_task_metrics_skips_invalid():
    task_data = {
        "metrics": {
            "m": {
                "scores": {
                    "a": "not_a_dict",
                    "b": {"no_value": 1},
                    "c": {"value": "not_a_number"},
                    "d": {"value": None},
                }
            }
        }
    }
    assert not _extract_task_metrics("t", task_data)


def test_extract_task_metrics_empty():
    assert not _extract_task_metrics("t", {})
    assert not _extract_task_metrics("t", {"metrics": None})


# --- _extract_metrics_from_results_yml ---


def test_extract_metrics_from_results_yml(sample_results_dir):
    metrics = _extract_metrics_from_results_yml(
        sample_results_dir / "artifacts" / "results.yml"
    )
    assert metrics["nq_ndcg_at_10"] == 0.85
    assert metrics["nq_ndcg_at_10_recall"] == 0.72
    assert metrics["nq_mrr"] == 0.91
    assert metrics["MTEB_avg"] == 0.80
    assert metrics["MTEB_retrieval_avg"] == 0.75
    # Stats extracted
    assert metrics["nq_ndcg_at_10_stderr"] == 0.02
    assert metrics["nq_ndcg_at_10_num_samples"] == 100.0


def test_extract_metrics_invalid_yaml(tmp_path):
    (tmp_path / "results.yml").write_text("not_a_dict", encoding="utf-8")
    assert not _extract_metrics_from_results_yml(tmp_path / "results.yml")


def test_extract_metrics_no_results_key(tmp_path):
    (tmp_path / "results.yml").write_text(
        yaml.dump({"no_results": {}}), encoding="utf-8"
    )
    assert not _extract_metrics_from_results_yml(tmp_path / "results.yml")


# --- _load_config ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_load_config(mock_mlflow, config_dir):
    config = _load_config(config_dir)
    assert config["experiment_name"] == "test-experiment"
    assert config["tags"]["invocation_id"] == "abc123"
    assert config["tags"]["task_name"] == "mteb.FiQA2018"
    assert config["log_artifacts"] is True
    assert config["log_logs"] is True
    mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow-test:5000")


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_load_config_defaults(mock_mlflow, tmp_path):
    (tmp_path / "export_config.yml").write_text(
        yaml.safe_dump({"tags": {"invocation_id": "x", "task_name": "t"}}),
        encoding="utf-8",
    )
    config = _load_config(tmp_path)
    assert config["experiment_name"] == "nemo-evaluator-launcher"
    assert config["description"] == ""
    assert config["skip_existing"] is False
    assert config["run_name"] == "eval-x-t"


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_load_config_copy_artifacts_compat(mock_mlflow, tmp_path):
    """copy_artifacts/copy_logs are mapped to log_artifacts/log_logs."""
    (tmp_path / "export_config.yml").write_text(
        yaml.safe_dump({"copy_artifacts": False, "copy_logs": False, "tags": {}}),
        encoding="utf-8",
    )
    config = _load_config(tmp_path)
    assert config["log_artifacts"] is False
    assert config["log_logs"] is False


# --- _check_existing_run ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_check_existing_run_found(mock_mlflow):
    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    mock_mlflow.get_experiment_by_name.return_value = mock_exp
    # Mock a non-empty DataFrame-like object
    mock_df = MagicMock()
    mock_df.empty = False
    mock_df.iloc.__getitem__ = MagicMock(return_value=MagicMock(run_id="abc"))
    mock_mlflow.search_runs.return_value = mock_df
    exists, run_id = _check_existing_run(SAMPLE_CONFIG)
    assert exists is True
    assert run_id == "abc"


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_check_existing_run_not_found(mock_mlflow):
    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    mock_mlflow.get_experiment_by_name.return_value = mock_exp
    mock_df = MagicMock()
    mock_df.empty = True
    mock_mlflow.search_runs.return_value = mock_df
    exists, run_id = _check_existing_run(SAMPLE_CONFIG)
    assert exists is False
    assert run_id is None


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_check_existing_run_no_experiment(mock_mlflow):
    mock_mlflow.get_experiment_by_name.return_value = None
    exists, run_id = _check_existing_run(SAMPLE_CONFIG)
    assert exists is False


# --- _log_artifacts ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
@patch("shutil.copytree")
def test_log_artifacts(mock_copytree, mock_mlflow, sample_results_dir):
    (sample_results_dir / "artifacts" / "cache").mkdir()
    (sample_results_dir / "artifacts" / "data.db").write_text("")
    (sample_results_dir / "artifacts" / "synthetic").mkdir()
    (sample_results_dir / "artifacts" / "debug.json").write_text("{}")

    _log_artifacts(sample_results_dir, SAMPLE_CONFIG)

    # copytree called with correct source
    mock_copytree.assert_called_once()
    src = mock_copytree.call_args[0][0]
    assert str(src).endswith("artifacts")
    # log_artifacts called with correct artifact_path
    mock_mlflow.log_artifacts.assert_called_once()
    artifact_path = mock_mlflow.log_artifacts.call_args[1]["artifact_path"]
    assert artifact_path == "mteb.mteb.FiQA2018/artifacts"

    # Verify the ignore function excludes the right files
    ignore_fn = mock_copytree.call_args[1]["ignore"]
    excluded = ignore_fn(
        "",
        ["results.yml", "cache", "data.db", "synthetic", "debug.json", "metadata.yaml"],
    )
    assert "cache" in excluded
    assert "data.db" in excluded
    assert "synthetic" in excluded
    assert "debug.json" in excluded
    assert "results.yml" not in excluded
    assert "metadata.yaml" not in excluded


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_log_artifacts_no_artifacts_dir(mock_mlflow, tmp_path):
    _log_artifacts(tmp_path, SAMPLE_CONFIG)
    mock_mlflow.log_artifacts.assert_not_called()


# --- _log_logs ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_log_logs(mock_mlflow, tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "slurm-123.log").write_text("log")
    (logs_dir / "subdir").mkdir()
    _log_logs(tmp_path, SAMPLE_CONFIG)
    assert mock_mlflow.log_artifact.call_count == 1


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_log_logs_no_dir(mock_mlflow, tmp_path):
    _log_logs(tmp_path, SAMPLE_CONFIG)
    mock_mlflow.log_artifact.assert_not_called()


# --- _init_mlflow_run ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_init_creates_run(mock_mlflow, tmp_path):
    mock_run = MagicMock()
    mock_run.info.run_id = "new-run-123"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    (tmp_path / "export_config.yml").write_text("config: true")

    run_id = _init_mlflow_run(tmp_path, SAMPLE_CONFIG)

    assert run_id == "new-run-123"
    mock_mlflow.set_tags.assert_called_once()
    tags = mock_mlflow.set_tags.call_args[0][0]
    assert tags["status"] == "deploying"
    assert (tmp_path / "mlflow_run_id.txt").read_text() == "new-run-123"
    # Params logged
    mock_mlflow.log_params.assert_called_once()
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["invocation_id"] == "abc123"
    assert params["executor"] == "slurm"


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_init_retry(mock_mlflow, tmp_path):
    (tmp_path / "mlflow_run_id.txt").write_text("existing-456")
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    run_id = _init_mlflow_run(tmp_path, SAMPLE_CONFIG)

    assert run_id == "existing-456"
    mock_mlflow.set_tag.assert_called_once_with("status", "retrying")


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_init_sets_run_name_and_description(mock_mlflow, tmp_path):
    mock_run = MagicMock()
    mock_run.info.run_id = "run-desc"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    _init_mlflow_run(tmp_path, SAMPLE_CONFIG)

    mock_mlflow.set_tag.assert_any_call("mlflow.runName", "test-run")
    mock_mlflow.set_tag.assert_any_call("mlflow.note.content", "test description")


# --- _finalize_mlflow_run ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_finalize_updates_existing(mock_mlflow, sample_results_dir):
    (sample_results_dir / "mlflow_run_id.txt").write_text("init-123")
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    run_id = _finalize_mlflow_run(
        sample_results_dir, {**SAMPLE_CONFIG, "skip_existing": False}
    )

    assert run_id == "init-123"
    mock_mlflow.set_tag.assert_any_call("status", "completed")
    mock_mlflow.log_metrics.assert_called_once()


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_finalize_fallback_creates_new(mock_mlflow, sample_results_dir):
    mock_run = MagicMock()
    mock_run.info.run_id = "fallback-run"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_mlflow.get_experiment_by_name.return_value = None

    run_id = _finalize_mlflow_run(
        sample_results_dir, {**SAMPLE_CONFIG, "skip_existing": False}
    )

    assert run_id == "fallback-run"
    tags = mock_mlflow.set_tags.call_args[0][0]
    assert tags["status"] == "completed"


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_finalize_no_results(mock_mlflow, tmp_path):
    (tmp_path / "mlflow_run_id.txt").write_text("run-noresults")
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    config = {
        **SAMPLE_CONFIG,
        "skip_existing": False,
        "log_artifacts": False,
        "log_logs": False,
    }
    run_id = _finalize_mlflow_run(tmp_path, config)

    assert run_id == "run-noresults"
    mock_mlflow.log_metrics.assert_not_called()


# --- _update_mlflow_run ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_update_no_run_id(mock_mlflow, tmp_path):
    _update_mlflow_run(tmp_path, SAMPLE_CONFIG)
    mock_mlflow.start_run.assert_not_called()


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_update_heartbeat_only(mock_mlflow, tmp_path):
    (tmp_path / "mlflow_run_id.txt").write_text("run-1")
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    _update_mlflow_run(tmp_path, SAMPLE_CONFIG)

    mock_mlflow.set_tag.assert_any_call("heartbeat", unittest.mock.ANY)
    status_calls = [
        c for c in mock_mlflow.set_tag.call_args_list if c[0][0] == "status"
    ]
    assert not status_calls


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_update_with_status(mock_mlflow, tmp_path):
    (tmp_path / "mlflow_run_id.txt").write_text("run-1")
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    _update_mlflow_run(tmp_path, SAMPLE_CONFIG, "evaluating")

    mock_mlflow.set_tag.assert_any_call("status", "evaluating")
    mock_mlflow.set_tag.assert_any_call("heartbeat", unittest.mock.ANY)


# --- _cancel_mlflow_run ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_cancel_no_run_id(mock_mlflow, tmp_path):
    _cancel_mlflow_run(tmp_path, SAMPLE_CONFIG, "cancelled")
    mock_mlflow.start_run.assert_not_called()


@patch("nemo_evaluator_launcher.exporters.mlflow_live.mlflow")
def test_cancel_sets_status(mock_mlflow, tmp_path):
    (tmp_path / "mlflow_run_id.txt").write_text("run-cancel")
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    _cancel_mlflow_run(tmp_path, SAMPLE_CONFIG, "timeout")

    mock_mlflow.set_tag.assert_any_call("status", "timeout")
    mock_mlflow.set_tag.assert_any_call("ended_at", unittest.mock.ANY)


# --- main() ---


@patch("nemo_evaluator_launcher.exporters.mlflow_live._finalize_mlflow_run")
@patch("nemo_evaluator_launcher.exporters.mlflow_live._load_config")
def test_main_finalize(mock_load, mock_finalize, tmp_path, monkeypatch):
    mock_load.return_value = SAMPLE_CONFIG
    mock_finalize.return_value = "run-main"
    monkeypatch.setattr(export_mod, "__file__", str(tmp_path / "mlflow_live.py"))
    monkeypatch.setattr("sys.argv", ["mlflow_live.py"])

    export_mod.main()

    mock_finalize.assert_called_once()


@patch("nemo_evaluator_launcher.exporters.mlflow_live._init_mlflow_run")
@patch("nemo_evaluator_launcher.exporters.mlflow_live._load_config")
def test_main_init(mock_load, mock_init, tmp_path, monkeypatch):
    mock_load.return_value = SAMPLE_CONFIG
    mock_init.return_value = "init-run"
    monkeypatch.setattr(export_mod, "__file__", str(tmp_path / "mlflow_live.py"))
    monkeypatch.setattr("sys.argv", ["mlflow_live.py", "--init"])

    export_mod.main()

    mock_init.assert_called_once()


@patch("nemo_evaluator_launcher.exporters.mlflow_live._update_mlflow_run")
@patch("nemo_evaluator_launcher.exporters.mlflow_live._load_config")
def test_main_update(mock_load, mock_update, tmp_path, monkeypatch):
    mock_load.return_value = SAMPLE_CONFIG
    monkeypatch.setattr(export_mod, "__file__", str(tmp_path / "mlflow_live.py"))
    monkeypatch.setattr("sys.argv", ["mlflow_live.py", "--update", "evaluating"])

    export_mod.main()

    mock_update.assert_called_once_with(tmp_path, SAMPLE_CONFIG, "evaluating")


@patch("nemo_evaluator_launcher.exporters.mlflow_live._cancel_mlflow_run")
@patch("nemo_evaluator_launcher.exporters.mlflow_live._load_config")
def test_main_cancel(mock_load, mock_cancel, tmp_path, monkeypatch):
    mock_load.return_value = SAMPLE_CONFIG
    monkeypatch.setattr(export_mod, "__file__", str(tmp_path / "mlflow_live.py"))
    monkeypatch.setattr("sys.argv", ["mlflow_live.py", "--cancel", "timeout"])

    export_mod.main()

    mock_cancel.assert_called_once_with(tmp_path, SAMPLE_CONFIG, "timeout")


@patch(
    "nemo_evaluator_launcher.exporters.mlflow_live._finalize_mlflow_run",
    side_effect=RuntimeError("fail"),
)
@patch("nemo_evaluator_launcher.exporters.mlflow_live._load_config")
def test_main_catches_errors(mock_load, mock_finalize, tmp_path, monkeypatch):
    mock_load.return_value = SAMPLE_CONFIG
    monkeypatch.setattr(export_mod, "__file__", str(tmp_path / "mlflow_live.py"))
    monkeypatch.setattr("sys.argv", ["mlflow_live.py"])

    export_mod.main()  # Should not raise

    mock_finalize.assert_called_once()
