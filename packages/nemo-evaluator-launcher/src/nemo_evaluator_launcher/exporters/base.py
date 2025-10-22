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
"""Base exporter interface for nemo-evaluator-launcher results."""

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    dest: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExporter(ABC):
    """Base interface for result exporters."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db = ExecutionDB()

    def export_invocation(self, invocation_id: str) -> Dict[str, Any]:
        """Export all jobs in an invocation."""
        jobs = self.db.get_jobs(invocation_id)

        if not jobs:
            return {
                "success": False,
                "error": f"No jobs found for invocation {invocation_id}",
            }

        results = {}
        for job_id, job_data in jobs.items():
            result = self.export_job(job_data)
            results[job_id] = asdict(result)

        return {"success": True, "invocation_id": invocation_id, "jobs": results}

    @abstractmethod
    def export_job(self, job_data: JobData) -> ExportResult:
        """Export a single job's results."""
        pass

    @abstractmethod
    def supports_executor(self, executor_type: str) -> bool:
        """Check if this exporter supports the given executor type."""
        pass

    def get_job_paths(self, job_data: JobData) -> Dict[str, Any]:
        """Get result paths based on executor type from job metadata."""
        # Special case: remote executor artifacts accessed locally (remote auto-export)
        if job_data.data.get("storage_type") == "remote_local":
            output_dir = Path(job_data.data["output_dir"])
            return {
                "artifacts_dir": output_dir / "artifacts",
                "logs_dir": output_dir / "logs",
                "storage_type": "remote_local",
            }

        if job_data.executor == "local":
            output_dir = Path(job_data.data["output_dir"])
            return {
                "artifacts_dir": output_dir / "artifacts",
                "logs_dir": output_dir / "logs",
                "storage_type": "local_filesystem",
            }

        elif job_data.executor == "slurm":
            return {
                "remote_path": job_data.data["remote_rundir_path"],
                "hostname": job_data.data["hostname"],
                "username": job_data.data["username"],
                "storage_type": "remote_ssh",
            }

        elif job_data.executor == "gitlab":
            pipeline_id = job_data.data.get("pipeline_id")
            if pipeline_id and os.getenv("CI"):
                return {
                    "artifacts_dir": Path(f"artifacts/{pipeline_id}"),
                    "storage_type": "gitlab_ci_local",
                }
            else:
                return {
                    "pipeline_id": pipeline_id,
                    "project_id": job_data.data.get("project_id", 155749),
                    "storage_type": "gitlab_remote",
                }

        elif job_data.executor == "lepton":
            output_dir = Path(job_data.data["output_dir"])
            return {
                "artifacts_dir": output_dir / "artifacts",
                "logs_dir": output_dir / "logs",
                "storage_type": "local_filesystem",
            }

        else:
            raise ValueError(f"Unknown executor: {job_data.executor}")
