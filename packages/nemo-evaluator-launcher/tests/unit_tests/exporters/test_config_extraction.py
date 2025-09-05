"""Tests for config extraction and task name utilities."""

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.exporters.utils import (
    extract_exporter_config,
    get_task_name,
)


class TestConfigExtraction:
    def test_extract_exporter_config_from_job_config(self):
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.0",
            timestamp=123.0,
            executor="local",
            data={},
            config={
                "execution": {
                    "auto_export": {
                        "configs": {
                            "wandb": {
                                "entity": "test-entity",
                                "project": "test-project",
                            }
                        }
                    }
                }
            },
        )
        config = extract_exporter_config(job_data, "wandb")
        assert config["entity"] == "test-entity"
        assert config["project"] == "test-project"

    def test_get_task_name_local_job(self):
        job_data = JobData(
            invocation_id="test123",
            job_id="test123.1",
            timestamp=123.0,
            executor="local",
            data={},
            config={
                "evaluation": {
                    "tasks": [
                        {"name": "task_0"},
                        {"name": "task_1"},
                        {"name": "task_2"},
                    ]
                }
            },
        )
        task_name = get_task_name(job_data)
        assert task_name == "task_1"

    def test_get_task_name_gitlab_job(self):
        job_data = JobData(
            invocation_id="test123",
            job_id="test123",
            timestamp=123.0,
            executor="gitlab",
            data={},
        )
        task_name = get_task_name(job_data)
        assert task_name == "all_tasks"
