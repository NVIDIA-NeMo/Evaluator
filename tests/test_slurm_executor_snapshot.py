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
"""Remote SLURM submission: snapshot upload and staging cleanup semantics."""

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
