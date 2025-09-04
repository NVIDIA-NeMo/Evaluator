"""Tests for the execdb module."""

import json
import pathlib
import time
from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.common.execdb import (
    EXEC_DB_DIR,
    EXEC_DB_FILE,
    ExecutionDB,
    JobData,
    generate_invocation_id,
    generate_job_id,
    get_job,
    get_jobs,
    write_job,
)


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Clean up singleton between tests."""
    # Reset singleton instance
    ExecutionDB._instance = None

    # Clear any existing data
    if hasattr(ExecutionDB, "_jobs"):
        ExecutionDB._jobs = {}
    if hasattr(ExecutionDB, "_invocations"):
        ExecutionDB._invocations = {}

    yield

    # Clean up after test
    ExecutionDB._instance = None
    if hasattr(ExecutionDB, "_jobs"):
        ExecutionDB._jobs = {}
    if hasattr(ExecutionDB, "_invocations"):
        ExecutionDB._invocations = {}


def test_singleton_pattern():
    """Test that ExecutionDB follows singleton pattern."""
    db1 = ExecutionDB()
    db2 = ExecutionDB()
    assert db1 is db2


def test_ensure_db_dir_creation(tmpdir):
    """Test that database directory is created if it doesn't exist."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        db = ExecutionDB()
        assert EXEC_DB_DIR.exists()
        assert EXEC_DB_DIR.is_dir()


def test_id_generation():
    """Test that ID generation functions work correctly."""
    inv_id = generate_invocation_id()
    job_id = generate_job_id(inv_id, 0)

    assert len(inv_id) == 8  # 8 hex characters
    assert job_id == f"{inv_id}.0"

    # Test uniqueness
    inv_ids = [generate_invocation_id() for _ in range(10)]
    job_ids = [generate_job_id(inv_id, 0) for inv_id in inv_ids]

    assert len(set(inv_ids)) == 10
    assert len(set(job_ids)) == 10


def test_write_and_get_job(tmpdir):
    """Test writing and retrieving a job."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

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
        write_job(job)

        # Verify job was written to file
        assert EXEC_DB_FILE.exists()

        # Read and verify file content
        with open(EXEC_DB_FILE, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["invocation_id"] == invocation_id
            assert record["job_id"] == job_id
            assert record["executor"] == executor
            assert record["data"] == data
            assert "timestamp" in record

        # Get job and verify
        result = get_job(job_id)
        assert result is not None
        assert isinstance(result, JobData)
        assert result.invocation_id == invocation_id
        assert result.job_id == job_id
        assert result.executor == executor
        assert result.data == data


def test_write_job_with_null_job_id(tmpdir):
    """Test writing a job with null job_id."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

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
        write_job(job)

        # Verify job was written to file
        with open(EXEC_DB_FILE, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["invocation_id"] == invocation_id
            assert record["job_id"] is None
            assert record["executor"] == executor


def test_get_nonexistent_job(tmpdir):
    """Test getting a job that doesn't exist."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        result = get_job("job-nonexistent")
        assert result is None


def test_load_existing_jobs(tmpdir):
    """Test loading existing jobs from file."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

        # Create existing job data
        existing_jobs = [
            {
                "invocation_id": "inv-11111111",
                "job_id": "job-22222222",
                "timestamp": 1234567890,
                "executor": "local",
                "data": {"test": "data1"},
            },
            {
                "invocation_id": "inv-33333333",
                "job_id": "job-44444444",
                "timestamp": 1234567891,
                "executor": "gitlab",
                "data": {"test": "data2"},
            },
        ]

        # Write existing jobs to file
        EXEC_DB_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXEC_DB_FILE, "w") as f:
            for job in existing_jobs:
                f.write(json.dumps(job) + "\n")

        db = ExecutionDB()

        # Verify jobs are loaded
        result1 = get_job("job-22222222")
        assert result1 is not None
        assert isinstance(result1, JobData)
        assert result1.invocation_id == "inv-11111111"
        assert result1.executor == "local"
        assert result1.data == {"test": "data1"}

        result2 = get_job("job-44444444")
        assert result2 is not None
        assert isinstance(result2, JobData)
        assert result2.invocation_id == "inv-33333333"
        assert result2.executor == "gitlab"
        assert result2.data == {"test": "data2"}


def test_load_existing_jobs_with_invalid_json(tmpdir):
    """Test loading existing jobs with invalid JSON lines."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

        EXEC_DB_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXEC_DB_FILE, "w") as f:
            f.write(
                '{"invocation_id": "inv-11111111", "job_id": "job-22222222", "timestamp": 1234567890, "executor": "local", "data": {}}\n'
            )
            f.write("invalid json line\n")
            f.write(
                '{"invocation_id": "inv-33333333", "job_id": "job-44444444", "timestamp": 1234567891, "executor": "gitlab", "data": {}}\n'
            )

        db = ExecutionDB()

        assert get_job("job-22222222") is not None
        assert get_job("job-44444444") is not None


def test_multiple_jobs_same_id(tmpdir):
    """Test that writing multiple jobs with same ID overwrites the previous one."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

        invocation_id = "inv-12345678"
        job_id = "job-87654321"
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

        write_job(job1)
        write_job(job2)

        result = get_job(job_id)
        assert result is not None
        assert isinstance(result, JobData)
        assert result.executor == "gitlab"
        assert result.data == {"data": "second"}

        with open(EXEC_DB_FILE, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2


def test_empty_data(tmpdir):
    """Test writing job with empty data."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

        job = JobData(
            invocation_id="inv-12345678",
            job_id="job-87654321",
            timestamp=time.time(),
            executor="local",
            data={},
        )
        write_job(job)

        result = get_job("job-87654321")
        assert result is not None
        assert isinstance(result, JobData)
        assert result.data == {}


def test_get_jobs_by_invocation_id(tmpdir):
    """Test getting all jobs for a specific invocation."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

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

        write_job(job1)
        write_job(job2)
        write_job(job3)

        # Get jobs for the specific invocation
        jobs = get_jobs(invocation_id)

        assert len(jobs) == 2
        assert generate_job_id(invocation_id, 0) in jobs
        assert generate_job_id(invocation_id, 1) in jobs
        assert generate_job_id("deadbeef", 0) not in jobs

        assert jobs[generate_job_id(invocation_id, 0)].data["task"] == "task1"
        assert jobs[generate_job_id(invocation_id, 1)].data["task"] == "task2"


def test_get_jobs_nonexistent_invocation(tmpdir):
    """Test getting jobs for a nonexistent invocation."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        jobs = get_jobs("inv-nonexistent")
        assert jobs == {}


def test_invocation_mapping_consistency(tmpdir):
    """Test that invocation to job mapping is maintained correctly."""
    with patch(
        "nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR",
        pathlib.Path(tmpdir) / "test-db",
    ):
        ExecutionDB._instance = None
        if EXEC_DB_FILE.exists():
            EXEC_DB_FILE.unlink()

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
        jobs = get_jobs(invocation_id)
        assert len(jobs) == 2
        assert generate_job_id(invocation_id, 0) in jobs
        assert generate_job_id(invocation_id, 1) in jobs
