"""Test configuration and fixtures for nv-eval-platform tests."""

import json
import pathlib
import re
import time
from typing import Dict
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.execdb import (
    ExecutionDB,
    JobData,
    generate_invocation_id,
    generate_job_id,
)
from nemo_evaluator_launcher.executors.base import (
    BaseExecutor,
    ExecutionState,
    ExecutionStatus,
)
from nemo_evaluator_launcher.executors.registry import register_executor


@register_executor("dummy")
class DummyExecutor(BaseExecutor):
    """Dummy executor for testing purposes.

    This executor simulates backend execution without actually running any jobs.
    It maintains an in-memory state of job statuses for testing.
    """

    _job_statuses: Dict[str, ExecutionState] = {}
    _invocation_counter = 0

    @classmethod
    def reset_state(cls) -> None:
        """Reset the executor's internal state for testing."""
        cls._job_statuses = {}
        cls._invocation_counter = 0

    @classmethod
    def set_job_status(cls, job_id: str, status: ExecutionState) -> None:
        """Set the status of a specific job for testing."""
        cls._job_statuses[job_id] = status

    @classmethod
    def execute_eval(cls, cfg: DictConfig, dry_run: bool = False) -> str:
        """Simulate running an evaluation job.

        Args:
            cfg: The configuration object for the evaluation run.
            dry_run: If True, prepare job configurations without execution.

        Returns:
            str: The invocation ID for the evaluation run.
        """
        # Generate invocation ID
        invocation_id = generate_invocation_id()

        # Create job IDs for each evaluation task
        job_ids = []
        for idx, task in enumerate(cfg.evaluation):
            job_id = generate_job_id(invocation_id, idx)
            job_ids.append(job_id)

            # Set initial status as pending
            cls._job_statuses[job_id] = ExecutionState.PENDING

        if dry_run:
            print("🔍 DRY RUN: Dummy executor job configurations prepared")
            print(f"   - Tasks: {len(cfg.evaluation)}")
            for idx, task in enumerate(cfg.evaluation):
                print(f"   - Task {idx}: {task.name}")
            print(f"   - Invocation ID: {invocation_id}")
            print("\nTo execute, run the executor without --dry-run")
            return invocation_id

        # Save job metadata to database
        db = ExecutionDB()
        for job_id, task in zip(job_ids, cfg.evaluation):
            # Convert OmegaConf objects to regular Python dicts for JSON serialization
            overrides = task.get("overrides", {})
            if hasattr(overrides, "_content"):
                overrides = dict(overrides)

            db.write_job(
                job=JobData(
                    invocation_id=invocation_id,
                    job_id=job_id,
                    timestamp=time.time(),
                    executor="dummy",
                    data={
                        "task_name": task.name,
                        "overrides": overrides,
                    },
                )
            )

        return invocation_id

    @staticmethod
    def get_status(id: str) -> list[ExecutionStatus]:
        """Get the status of a job or invocation group.

        Args:
            id: Unique job identifier or invocation identifier.

        Returns:
            list[ExecutionStatus]: List of execution statuses for the job(s).
        """
        db = ExecutionDB()

        # If id looks like an invocation_id (8 hex digits, no dot), get all jobs for it
        if len(id) == 8 and "." not in id:
            jobs = db.get_jobs(id)
            statuses = []
            for job_id, job_data in jobs.items():
                if job_data.executor == "dummy":
                    state = DummyExecutor._job_statuses.get(
                        job_id, ExecutionState.FAILED
                    )
                    statuses.append(ExecutionStatus(id=job_id, state=state))
            return statuses

        # Otherwise, treat as job_id
        job_data = db.get_job(id)
        if job_data is None or job_data.executor != "dummy":
            return []

        state = DummyExecutor._job_statuses.get(id, ExecutionState.FAILED)
        return [ExecutionStatus(id=id, state=state)]


@pytest.fixture(autouse=True)
def reset_dummy_executor():
    """Reset the dummy executor state before each test."""
    DummyExecutor.reset_state()
    yield


@pytest.fixture
def tmp_execdb_dir(tmpdir):
    """Create a temporary directory for execution database."""
    return pathlib.Path(tmpdir) / "execdb"


@pytest.fixture
def mock_execdb(tmp_execdb_dir):
    """Mock the execution database to use a temporary directory."""
    with patch("nemo_evaluator_launcher.common.execdb.EXEC_DB_DIR", tmp_execdb_dir):
        # Reset singleton instance
        ExecutionDB._instance = None
        yield tmp_execdb_dir


@pytest.fixture
def sample_run_config():
    """Create a sample RunConfig for testing."""
    config_dict = {
        "deployment": {"type": "none"},
        "execution": {"type": "dummy", "output_dir": "/tmp/test_output"},
        "target": {
            "api_endpoint": {"api_key_name": "test_key", "model_id": "test_model"}
        },
        "evaluation": [
            {"name": "test_task_1", "overrides": {"param1": "value1"}},
            {"name": "test_task_2", "overrides": {"param2": "value2"}},
        ],
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_tasks_mapping():
    """Mock tasks mapping for testing."""
    return {
        ("lm-eval", "test_task_1"): {
            "task": "test_task_1",
            "endpoint_type": "openai",
            "harness": "lm-eval",
            "container": "test-container:latest",
        },
        ("helm", "test_task_2"): {
            "task": "test_task_2",
            "endpoint_type": "anthropic",
            "harness": "helm",
            "container": "test-container:latest",
        },
    }


@pytest.fixture
def mock_hydra_config():
    """Mock Hydra configuration loading."""
    config_dict = {
        "deployment": {"type": "none"},
        "execution": {"type": "dummy", "output_dir": "/tmp/test_output"},
        "target": {
            "api_endpoint": {"api_key_name": "test_key", "model_id": "test_model"}
        },
        "evaluation": [{"name": "test_task_1", "overrides": {"param1": "value1"}}],
    }

    with patch("nemo_evaluator_launcher.api.types.hydra.compose") as mock_compose:
        mock_compose.return_value = OmegaConf.create(config_dict)
        yield mock_compose


@pytest.fixture
def mock_api_endpoint_check():
    """Mock the API endpoint check function."""
    with patch(
        "nemo_evaluator_launcher.api.functional._check_api_endpoint_when_deployment_is_configured"
    ) as mock_check:
        yield mock_check


@pytest.fixture
def mock_tasks_mapping_load():
    """Mock the tasks mapping loading function."""
    with patch(
        "nemo_evaluator_launcher.api.functional.load_tasks_mapping"
    ) as mock_load:
        mock_load.return_value = {
            ("lm-eval", "test_task_1"): {
                "task": "test_task_1",
                "endpoint_type": "openai",
                "harness": "lm-eval",
                "container": "test-container:latest",
            },
            ("helm", "test_task_2"): {
                "task": "test_task_2",
                "endpoint_type": "anthropic",
                "harness": "helm",
                "container": "test-container:latest",
            },
        }
        yield mock_load


@pytest.fixture
def mock_print():
    """Mock the print function to capture output."""
    with patch("builtins.print") as mock_print:
        yield mock_print


@pytest.fixture
def mock_json_dumps():
    """Mock json.dumps to capture JSON output."""
    with patch("json.dumps") as mock_dumps:
        yield mock_dumps


@pytest.fixture
def mock_tabulate():
    """Mock tabulate to capture table output."""
    with patch("tabulate.tabulate") as mock_tabulate:
        yield mock_tabulate


@pytest.fixture
def mock_job_data():
    """Standard mock job data fixture with config for exporter tests."""
    return JobData(
        invocation_id="test1234",
        job_id="test1234.0",
        timestamp=1234567890.0,
        executor="local",
        data={"served_model_name": "qwen3-0.6b", "output_dir": "/tmp/test-output"},
        config={
            "execution": {
                "auto_export": {
                    "configs": {
                        "wandb": {
                            "entity": "test-entity",
                            "project": "test-project",
                            "name": "test-run",
                        }
                    }
                }
            },
            "deployment": {"type": "none"},
        },
    )


@pytest.fixture(autouse=True)
def isolate_exporter_writes(tmp_path, monkeypatch, request):
    # isolate cwd for the consolidated exporters suite
    if "test_exporters.py" in str(request.node.fspath):
        monkeypatch.chdir(tmp_path)


@pytest.fixture
def make_job_fs(tmp_path):
    """
    Create a temp job run directory structure with artifacts + job_metadata.json.
    Returns the job base dir: <tmp>/<invocation>/<job_id>/
    """

    def _make(job_data, *, with_required=True, with_optional=True) -> pathlib.Path:
        base = tmp_path / job_data.invocation_id / job_data.job_id
        artifacts = base / "artifacts"
        logs = base / "logs"
        artifacts.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)

        # job_metadata.json
        (base / "job_metadata.json").write_text(
            json.dumps(
                {
                    "invocation_id": job_data.invocation_id,
                    "job_id": job_data.job_id,
                    "executor": job_data.executor,
                    "timestamp": job_data.timestamp,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Required artifacts
        if with_required:
            (artifacts / "results.yml").write_text(
                "results: {tasks: {demo: {metrics: {accuracy: {value: 0.9}}}}}",
                encoding="utf-8",
            )
            (artifacts / "eval_factory_metrics.json").write_text(
                json.dumps({"total_time": 1.23}), encoding="utf-8"
            )

        # Optional artifacts
        if with_optional:
            (artifacts / "omni-info.json").write_text(
                json.dumps({"model_name": "test-model"}), encoding="utf-8"
            )

        return base

    return _make


@pytest.fixture
def prepare_local_job(make_job_fs):
    def _prep(job_data, **kwargs):
        job_dir = make_job_fs(job_data, **kwargs)
        job_data.data["output_dir"] = str(job_dir.parent)
        return job_data, job_dir

    return _prep


@pytest.fixture
def load_job_metadata():
    def _load(job_dir: pathlib.Path) -> dict:
        return json.loads((job_dir / "job_metadata.json").read_text(encoding="utf-8"))

    return _load


def extract_invocation_id(mock_print) -> str:
    for c in mock_print.mock_calls:
        if c.args:
            s = str(c.args[0])
            m = re.search(r"nv-eval status\s+([0-9a-f]{8})\b", s)
            if m:
                return m.group(1)
    for c in mock_print.mock_calls:
        if c.args:
            m = re.search(r"\b([0-9a-f]{8})\b", str(c.args[0]))
            if m:
                return m.group(1)
    raise AssertionError("invocation id not found in output")
