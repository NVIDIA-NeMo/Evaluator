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
"""SLURM submission metadata, snapshot, and partial-failure semantics."""

import json
from unittest.mock import patch

import nemo_evaluator.executors.slurm_executor as slurm_executor_module
from nemo_evaluator.config import (
    BenchmarkConfig,
    EvalConfig,
    ExternalApiService,
    NodePool,
    OutputConfig,
    SimpleSolver,
    SlurmCluster,
)
from nemo_evaluator.executors.slurm_executor import SlurmExecutor


def _remote_config(tmp_path):
    config = EvalConfig(
        services={
            "m": ExternalApiService(
                type="api",
                url="http://localhost:8000/v1/chat/completions",
                protocol="chat_completions",
                model="test-model",
            )
        },
        benchmarks=[BenchmarkConfig(name="gsm8k", solver=SimpleSolver(type="simple", service="m"))],
        cluster=SlurmCluster(
            type="slurm",
            hostname="login.example.com",
            node_pools={"compute": NodePool(partition="batch")},
        ),
        output=OutputConfig(dir=str(tmp_path / "remote_out")),
    )
    # The CLI loader attaches the pre-expansion composed dict; simulate it.
    config._composed_raw = {"services": {"m": {"api_key": "${KEY}"}}}
    return config


def _local_config(tmp_path, *, shards=None):
    config = EvalConfig(
        services={
            "m": ExternalApiService(
                type="api",
                url="http://localhost:8000/v1/chat/completions",
                protocol="chat_completions",
                model="test-model",
            )
        },
        benchmarks=[BenchmarkConfig(name="gsm8k", solver=SimpleSolver(type="simple", service="m"))],
        cluster=SlurmCluster(
            type="slurm",
            shards=shards,
            node_pools={"compute": NodePool(partition="batch")},
        ),
        output=OutputConfig(dir=str(tmp_path / "local_out"), timestamped=False),
    )
    config._composed_raw = {"services": {"m": {"api_key": "${KEY}"}}}
    return config


def _run_remote(config, tmp_path, monkeypatch, copy_side_effect=None):
    """Run the executor with SSH mocked; track every mkdtemp'd directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    made = []

    def fake_mkdtemp(prefix=""):
        d = tmp_path / f"{prefix}{len(made)}"
        d.mkdir(parents=True)
        made.append(d)
        return str(d)

    monkeypatch.setattr(slurm_executor_module.tempfile, "mkdtemp", fake_mkdtemp)
    with (
        patch("nemo_evaluator.executors.ssh.copy_to_remote", side_effect=copy_side_effect) as copy_mock,
        patch("nemo_evaluator.executors.ssh.submit_eval", return_value={"job_id": "42"}) as submit_mock,
        patch("nemo_evaluator.run_store.RunMeta.save"),
    ):
        SlurmExecutor().run(config)
    return made, copy_mock, submit_mock


def test_remote_submission_uploads_snapshot_and_cleans_staging(tmp_path, monkeypatch, capsys):
    config = _remote_config(tmp_path)
    made, copy_mock, submit_mock = _run_remote(config, tmp_path, monkeypatch)

    # snapshot was uploaded to the resolved remote run dir
    copy_mock.assert_called_once()
    assert [p.name for p in copy_mock.call_args.args[1]] == ["full_config.yaml"]
    assert copy_mock.call_args.args[2] == config.output.dir

    submit_mock.assert_called_once()
    assert "SLURM job submitted: 42" in capsys.readouterr().out
    # staging dir (holding .secrets.env etc.) must not outlive the submission
    staging = next(d for d in made if "nel-slurm-" in d.name)
    assert not staging.exists()


def test_failed_upload_keeps_only_snapshot(tmp_path, monkeypatch, capsys):
    config = _remote_config(tmp_path)
    made, _, submit_mock = _run_remote(config, tmp_path, monkeypatch, copy_side_effect=RuntimeError("scp broke"))
    out = capsys.readouterr().out

    # the failed upload must not block the actual submission
    submit_mock.assert_called_once()
    assert "could not upload config snapshot" in out

    # staging purged; only the secret-free snapshot survives, with a retry hint
    staging = next(d for d in made if "nel-slurm-" in d.name)
    assert not staging.exists()
    kept = next(d for d in made if "nel-snapshot-" in d.name) / "full_config.yaml"
    assert kept.exists()
    assert "${KEY}" in kept.read_text()
    assert f"scp {kept} login.example.com:{config.output.dir}/" in out


def test_local_submission_persists_job_and_run_metadata(tmp_path, monkeypatch):
    config = _local_config(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    with (
        patch("nemo_evaluator.executors.ssh.submit_sbatch", return_value="42") as submit_mock,
        patch("nemo_evaluator.executors.ssh.submit_eval") as remote_submit,
        patch("nemo_evaluator.executors.ssh.copy_to_remote") as remote_copy,
    ):
        SlurmExecutor().run(config)

    script = tmp_path / "local_out" / "nel_eval.sbatch"
    submit_mock.assert_called_once_with(None, str(script))
    remote_submit.assert_not_called()
    remote_copy.assert_not_called()

    output_meta = json.loads((tmp_path / "local_out" / "slurm_job.json").read_text())
    stored_meta = json.loads((tmp_path / "xdg" / "nel" / "jobs" / "42" / "slurm_job.json").read_text())
    assert output_meta == stored_meta
    assert output_meta["job_id"] == "42"
    assert output_meta["hostname"] == ""
    assert output_meta["remote_dir"] == str(tmp_path / "local_out")

    run_meta = json.loads((tmp_path / "local_out" / "nel_run.json").read_text())
    assert run_meta["executor"] == "slurm"
    assert run_meta["details"]["job_ids"] == ["42"]
    assert run_meta["details"]["hostname"] == ""


def test_local_sharded_submission_persists_per_shard_and_aggregate_metadata(tmp_path, monkeypatch):
    config = _local_config(tmp_path, shards=2)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    with patch("nemo_evaluator.executors.ssh.submit_sbatch", side_effect=["42", "43"]):
        SlurmExecutor().run(config)

    output_dir = tmp_path / "local_out"
    for shard_idx, job_id in enumerate(("42", "43")):
        shard_meta = json.loads((output_dir / f"shard_{shard_idx}" / "slurm_job.json").read_text())
        assert shard_meta["job_id"] == job_id
        assert shard_meta["remote_dir"] == str(output_dir / f"shard_{shard_idx}")

    aggregate = json.loads((output_dir / "slurm_job.json").read_text())
    assert aggregate["job_id"] == "42"
    assert aggregate["job_ids"] == ["42", "43"]
    assert aggregate["is_sharded"] is True
    assert aggregate["num_shards"] == 2

    run_meta = json.loads((output_dir / "nel_run.json").read_text())
    assert run_meta["details"]["job_ids"] == ["42", "43"]
    assert run_meta["details"]["remote_dir"] == str(output_dir)


def test_local_relative_output_is_persisted_as_absolute(tmp_path, monkeypatch):
    config = _local_config(tmp_path)
    config.output.dir = "relative-output"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    with patch("nemo_evaluator.executors.ssh.submit_sbatch", return_value="42"):
        SlurmExecutor().run(config)

    expected = tmp_path / "relative-output"
    assert config.output.dir == str(expected)
    output_meta = json.loads((expected / "slurm_job.json").read_text())
    run_meta = json.loads((expected / "nel_run.json").read_text())
    assert output_meta["remote_dir"] == str(expected)
    assert run_meta["output_dir"] == str(expected)
    assert run_meta["details"]["remote_dir"] == str(expected)


def test_second_shard_submit_failure_keeps_submitted_job_and_warns(tmp_path, monkeypatch, capsys):
    config = _local_config(tmp_path, shards=2)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    with (
        patch(
            "nemo_evaluator.executors.ssh.submit_sbatch",
            side_effect=["42", RuntimeError("second sbatch failed")],
        ) as submit_mock,
        patch("nemo_evaluator.executors.ssh.cancel_job") as cancel_mock,
        patch("nemo_evaluator.executors.ssh.run_on_slurm_host") as run_on_host,
    ):
        SlurmExecutor().run(config)

    assert submit_mock.call_count == 2
    cancel_mock.assert_not_called()
    captured = capsys.readouterr()
    assert "SLURM job submitted: 42" in captured.out
    assert "run_id:" in captured.out
    assert "WARNING: Only 1/2 shards submitted" in captured.err
    output_dir = tmp_path / "local_out"
    aggregate = json.loads((output_dir / "slurm_job.json").read_text())
    assert aggregate["job_ids"] == ["42"]
    shard_meta = json.loads((output_dir / "shard_0" / "slurm_job.json").read_text())
    stored_meta = json.loads((tmp_path / "xdg" / "nel" / "jobs" / "42" / "slurm_job.json").read_text())
    run_meta = json.loads((output_dir / "nel_run.json").read_text())
    assert shard_meta["job_id"] == "42"
    assert stored_meta == shard_meta
    assert run_meta["details"]["job_ids"] == ["42"]
    assert not (output_dir / "shard_1" / "slurm_job.json").exists()
    assert all(not call.args[1].startswith("scancel ") for call in run_on_host.call_args_list)


def test_metadata_write_failure_keeps_job_id_visible_for_manual_recovery(tmp_path, monkeypatch, capsys):
    config = _local_config(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    blocked_jobs_store = tmp_path / "blocked-jobs-store"
    blocked_jobs_store.write_text("not a directory")
    monkeypatch.setattr(slurm_executor_module, "_jobs_store", lambda: blocked_jobs_store)

    with (
        patch("nemo_evaluator.executors.ssh.submit_sbatch", return_value="42"),
        patch("nemo_evaluator.executors.ssh.cancel_job") as cancel_mock,
        patch("nemo_evaluator.executors.ssh.run_on_slurm_host") as run_on_host,
    ):
        SlurmExecutor().run(config)

    cancel_mock.assert_not_called()
    captured = capsys.readouterr()
    assert "SLURM job submitted: 42" in captured.out
    assert "run_id:" in captured.out
    assert "WARNING: Only 1/1 shards submitted" in captured.err
    run_meta = json.loads((tmp_path / "local_out" / "nel_run.json").read_text())
    assert run_meta["details"]["job_id"] == "42"
    assert run_meta["details"]["job_ids"] == ["42"]
    assert not (tmp_path / "local_out" / "slurm_job.json").exists()
    assert all(not call.args[1].startswith("scancel ") for call in run_on_host.call_args_list)
