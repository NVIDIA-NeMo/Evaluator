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
"""Tests for the execdb module."""

import json
import time

# Access mocked EXEC_DB_DIR and EXEC_DB_FILE through the execdb module
import nemo_evaluator_launcher.common.execdb as execdb
from nemo_evaluator_launcher.common.execdb import (
    ExecutionDB,
    JobData,
    generate_invocation_id,
    generate_job_id,
)


def test_singleton_pattern(mock_execdb):
    """Test that ExecutionDB follows singleton pattern."""
    db1 = ExecutionDB()
    db2 = ExecutionDB()
    assert db1 is db2
    print(db1._jobs)


def test_ensure_db_dir_creation(mock_execdb):
    """Test that database directory is created if it doesn't exist."""
    _ = ExecutionDB()
    assert execdb.EXEC_DB_DIR.exists()
    assert execdb.EXEC_DB_DIR.is_dir()


def test_id_generation():
    """Test that ID generation functions work correctly."""
    inv_id = generate_invocation_id()
    job_id = generate_job_id(inv_id, 0)

    assert len(inv_id) == 16
    assert job_id == f"{inv_id}.0"

    # Test uniqueness
    inv_ids = [generate_invocation_id() for _ in range(10)]
    job_ids = [generate_job_id(inv_id, 0) for inv_id in inv_ids]

    assert len(set(inv_ids)) == 10
    assert len(set(job_ids)) == 10


def test_write_and_get_job(mock_execdb):
    """Test writing and retrieving a job."""
    invocation_id = "a1b2c3d4"
    job_id = generate_job_id(invocation_id, 0)
    executor = "local"
    data = {"model": "test-model", "tasks": ["task1", "task2"]}
    timestamp = time.time()
    job = JobData(
        invocation_id=invocation_id,
        job_id=job_id,
        timestamp=timestamp,
        executor=executor,
        data=data,
    )

    # Write job
    ExecutionDB().write_job(job)

    # Verify job was written to file
    assert execdb.EXEC_DB_FILE.exists()

    # Read and verify file content
    with open(execdb.EXEC_DB_FILE, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["invocation_id"] == invocation_id
        assert record["job_id"] == job_id
        assert record["executor"] == executor
        assert record["data"] == data
        assert "timestamp" in record

    # Get job and verify
    result = ExecutionDB().get_job(job_id)
    assert result is not None
    assert isinstance(result, JobData)
    assert result.invocation_id == invocation_id
    assert result.job_id == job_id
    assert result.executor == executor
    assert result.data == data


def test_write_job_with_null_job_id(mock_execdb):
    """Test writing a job with null job_id."""
    invocation_id = "inv-12345678"
    executor = "local"
    data = {"model": "test-model"}
    timestamp = time.time()
    job = JobData(
        invocation_id=invocation_id,
        job_id=None,
        timestamp=timestamp,
        executor=executor,
        data=data,
    )

    # Write job
    ExecutionDB().write_job(job)

    # Verify job was written to file
    with open(execdb.EXEC_DB_FILE, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["invocation_id"] == invocation_id
        assert record["job_id"] is None
        assert record["executor"] == executor


def test_get_nonexistent_job(mock_execdb):
    """Test getting a job that doesn't exist."""
    result = ExecutionDB().get_job("nonexist.0")
    assert result is None


def test_load_existing_jobs(mock_execdb):
    """Test loading existing jobs from file."""
    # Create existing job data
    existing_jobs = [
        {
            "invocation_id": "abcdef",
            "job_id": "abcdef.0",
            "timestamp": 1234567890,
            "executor": "local",
            "data": {"test": "data1"},
        },
        {
            "invocation_id": "fedcba",
            "job_id": "fedcba.0",
            "timestamp": 1234567891,
            "executor": "gitlab",
            "data": {"test": "data2"},
        },
    ]

    # Write existing jobs to file
    execdb.EXEC_DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(execdb.EXEC_DB_FILE, "w") as f:
        for job in existing_jobs:
            f.write(json.dumps(job) + "\n")

    _ = ExecutionDB()

    # Verify jobs are loaded
    result1 = ExecutionDB().get_job("abcdef.0")
    assert result1 is not None
    assert isinstance(result1, JobData)
    assert result1.invocation_id == "abcdef"
    assert result1.executor == "local"
    assert result1.data == {"test": "data1"}

    result2 = ExecutionDB().get_job("fedcba.0")
    assert result2 is not None
    assert isinstance(result2, JobData)
    assert result2.invocation_id == "fedcba"
    assert result2.executor == "gitlab"
    assert result2.data == {"test": "data2"}


def test_load_existing_jobs_with_invalid_json(mock_execdb):
    """Test loading existing jobs with invalid JSON lines."""
    execdb.EXEC_DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(execdb.EXEC_DB_FILE, "w") as f:
        f.write(
            '{"invocation_id": "abcdef", "job_id": "abcdef.0", "timestamp": 1234567890, "executor": "local", "data": {}}\n'
        )
        f.write("invalid json line\n")
        f.write(
            '{"invocation_id": "fedcba", "job_id": "fedcba.0", "timestamp": 1234567891, "executor": "gitlab", "data": {}}\n'
        )

    _ = ExecutionDB()

    assert ExecutionDB().get_job("abcdef.0") is not None
    assert ExecutionDB().get_job("fedcba.0") is not None


def test_multiple_jobs_same_id(mock_execdb):
    """Test that writing multiple jobs with same ID overwrites the previous one."""
    invocation_id = "abcdef"
    job_id = "abcdef.1"
    job1 = JobData(
        invocation_id=invocation_id,
        job_id=job_id,
        timestamp=time.time(),
        executor="local",
        data={"data": "first"},
    )
    job2 = JobData(
        invocation_id=invocation_id,
        job_id=job_id,
        timestamp=time.time(),
        executor="gitlab",
        data={"data": "second"},
    )

    ExecutionDB().write_job(job1)
    ExecutionDB().write_job(job2)

    result = ExecutionDB().get_job(job_id)
    assert result is not None
    assert isinstance(result, JobData)
    assert result.executor == "gitlab"
    assert result.data == {"data": "second"}

    with open(execdb.EXEC_DB_FILE, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2


def test_empty_data(mock_execdb):
    """Test writing job with empty data."""
    job = JobData(
        invocation_id="abcdef",
        job_id="abcdef.0",
        timestamp=time.time(),
        executor="local",
        data={},
    )
    ExecutionDB().write_job(job)

    result = ExecutionDB().get_job("abcdef.0")
    assert result is not None
    assert isinstance(result, JobData)
    assert result.data == {}


def test_get_jobs_by_invocation_id(mock_execdb):
    """Test getting all jobs for a specific invocation."""
    invocation_id = "a1b2c3d4"
    job1 = JobData(
        invocation_id=invocation_id,
        job_id=generate_job_id(invocation_id, 0),
        timestamp=time.time(),
        executor="local",
        data={"task": "task1"},
    )
    job2 = JobData(
        invocation_id=invocation_id,
        job_id=generate_job_id(invocation_id, 1),
        timestamp=time.time(),
        executor="local",
        data={"task": "task2"},
    )
    job3 = JobData(
        invocation_id="deadbeef",  # Different invocation
        job_id=generate_job_id("deadbeef", 0),
        timestamp=time.time(),
        executor="gitlab",
        data={"task": "task3"},
    )

    ExecutionDB().write_job(job1)
    ExecutionDB().write_job(job2)
    ExecutionDB().write_job(job3)

    # Get jobs for the specific invocation
    jobs = ExecutionDB().get_jobs(invocation_id)

    assert len(jobs) == 2
    assert generate_job_id(invocation_id, 0) in jobs
    assert generate_job_id(invocation_id, 1) in jobs
    assert generate_job_id("deadbeef", 0) not in jobs

    assert jobs[generate_job_id(invocation_id, 0)].data["task"] == "task1"
    assert jobs[generate_job_id(invocation_id, 1)].data["task"] == "task2"


def test_get_jobs_nonexistent_invocation(mock_execdb):
    """Test getting jobs for a nonexistent invocation."""
    jobs = ExecutionDB().get_jobs("inv-nonexistent")
    assert jobs == {}


def test_invocation_mapping_consistency(mock_execdb):
    """Test that invocation to job mapping is maintained correctly."""
    db = ExecutionDB()

    invocation_id = "a1b2c3d4"
    job1 = JobData(
        invocation_id=invocation_id,
        job_id=generate_job_id(invocation_id, 0),
        timestamp=time.time(),
        executor="local",
        data={"task": "task1"},
    )
    job2 = JobData(
        invocation_id=invocation_id,
        job_id=generate_job_id(invocation_id, 1),
        timestamp=time.time(),
        executor="local",
        data={"task": "task2"},
    )

    db.write_job(job1)
    db.write_job(job2)

    # Check invocation mapping
    job_ids = db.get_invocation_jobs(invocation_id)
    assert len(job_ids) == 2
    assert generate_job_id(invocation_id, 0) in job_ids
    assert generate_job_id(invocation_id, 1) in job_ids

    # Check that jobs can be retrieved
    jobs = ExecutionDB().get_jobs(invocation_id)
    assert len(jobs) == 2
    assert generate_job_id(invocation_id, 0) in jobs
    assert generate_job_id(invocation_id, 1) in jobs
