# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
