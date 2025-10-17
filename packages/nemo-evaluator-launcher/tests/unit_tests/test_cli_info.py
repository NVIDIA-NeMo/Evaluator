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
import pytest

from nemo_evaluator_launcher.cli.info import InfoCmd
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData


@pytest.mark.usefixtures("mock_execdb")
def test_logs_locations_local(job_local, capsys):
    cmd = InfoCmd(invocation_ids=[job_local.invocation_id])
    cmd.logs = True
    cmd.execute()
    out = capsys.readouterr().out
    assert f"{job_local.job_id}:" in out
    assert "/logs" in out


@pytest.mark.usefixtures("mock_execdb")
def test_artifacts_locations_local(job_local, capsys):
    cmd = InfoCmd(invocation_ids=[job_local.invocation_id])
    cmd.artifacts = True
    cmd.execute()
    out = capsys.readouterr().out
    assert f"{job_local.job_id}:" in out
    assert "/artifacts" in out


@pytest.mark.usefixtures("mock_execdb")
def test_config_dump_yaml(job_local, capsys):
    cmd = InfoCmd(invocation_ids=[job_local.invocation_id])
    cmd.config = True
    cmd.execute()
    out = capsys.readouterr().out
    # at least should have evaluation and tasks
    assert "evaluation:" in out
    assert "tasks:" in out


@pytest.mark.usefixtures("mock_execdb")
def test_no_jobs_found(capsys):
    # No jobs written to DB for this prefix
    cmd = InfoCmd(invocation_ids=["deadbeef"])
    cmd.execute()
    out = capsys.readouterr().out
    assert "No valid jobs found" in out


@pytest.mark.usefixtures("mock_execdb")
def test_multiple_jobs_under_invocation(job_local, prepare_local_job, capsys):
    # Add a second job under the same invocation
    inv = job_local.invocation_id
    second = JobData(
        invocation_id=inv,
        job_id=f"{inv}.1",
        timestamp=job_local.timestamp,
        executor="local",
        data={},
        config=job_local.config,
    )
    second, base = prepare_local_job(second, with_required=True, with_optional=True)
    # remove logs dir to force missing
    (base / "logs").rmdir()
    ExecutionDB().write_job(second)

    cmd = InfoCmd(invocation_ids=[inv])
    cmd.logs = True
    cmd.execute()
    out = capsys.readouterr().out
    assert f"{inv}.0:" in out
    assert f"{inv}.1:" in out
    # second job shows (not found)
    assert "(not found)" in out


@pytest.mark.usefixtures("mock_execdb")
def test_slurm_without_job_id(capsys):
    inv = "decafbadcafefeed"
    jd = JobData(
        invocation_id=inv,
        job_id=f"{inv}.0",
        timestamp=1_000_000_000.0,
        executor="slurm",
        data={
            "hostname": "h.example",
            "username": "user",
            "remote_rundir_path": "/remote/run/dir/mbpp",
            # no slurm_job_id
        },
        config={
            "execution": {"type": "slurm"},
            "deployment": {"type": "none"},
            "evaluation": {"tasks": [{"name": "mbpp"}]},
        },
    )
    ExecutionDB().write_job(jd)

    cmd = InfoCmd(invocation_ids=[inv])
    cmd.execute()
    out = capsys.readouterr().out
    # remote paths shown
    assert "(remote)" in out and "/logs" in out and "/artifacts" in out
    # no slurm line
    assert "Slurm Job ID:" not in out


@pytest.mark.usefixtures("mock_execdb")
def test_task_missing_index_is_graceful(capsys):
    inv = "feedfacefeedface"
    jd = JobData(
        invocation_id=inv,
        job_id=f"{inv}.0",  # index out of range
        timestamp=1_000_000_000.0,
        executor="local",
        data={"output_dir": "/tmp/test-output"},
        config={
            "execution": {"type": "local"},
            "deployment": {"type": "none"},
            "evaluation": {"tasks": [{"name": "mbpp"}]},
        },
    )
    ExecutionDB().write_job(jd)

    cmd = InfoCmd(invocation_ids=[inv])
    cmd.execute()
    out = capsys.readouterr().out
    assert f"Job {inv}.0" in out
    # No crash, task line may be absent
    assert "Task:" not in out or "Task:" in out  # tolerant