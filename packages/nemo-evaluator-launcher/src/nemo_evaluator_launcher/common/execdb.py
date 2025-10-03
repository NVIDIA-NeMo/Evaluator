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
"""Execution database module for tracking job executions."""

import json
import pathlib
import secrets
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from nemo_evaluator_launcher.common.logging_utils import logger

# Configuration constants
EXEC_DB_DIR = pathlib.Path.home() / ".nemo-evaluator" / "exec-db"
EXEC_DB_FILE = EXEC_DB_DIR / "exec.v1.jsonl"


def generate_invocation_id() -> str:
    """Generate a unique invocation ID as an 16-digit hex string."""
    return secrets.token_hex(8)


def generate_job_id(invocation_id: str, index: int) -> str:
    """Generate a job ID as <invocation_id>.<n>.

    Args:
        invocation_id: The invocation group ID (16-digit hex).
        index: The job index (0-based integer).
    Returns:
        The job ID string.
    """
    return f"{invocation_id}.{index}"


@dataclass
class JobData:
    """Data structure for job execution information.

    Attributes:
        invocation_id: 16-digit hex string.
        job_id: <invocation_id>.<n> string.
        timestamp: Unix timestamp when the job was created.
        executor: Name of the executor that handled this job.
        data: Additional job-specific data as a dictionary.
        config: Configuration used to setup a job.
    """

    invocation_id: str
    job_id: str
    timestamp: float
    executor: str
    data: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


class ExecutionDB:
    """Singleton class for managing execution database with invocation and job hierarchy."""

    _instance: Optional["ExecutionDB"] = None
    _jobs: Dict[str, JobData] = {}  # job_id -> JobData
    _invocations: Dict[str, List[str]] = {}  # invocation_id -> list of job_ids

    def __new__(cls) -> "ExecutionDB":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._ensure_db_dir()
            self._load_existing_jobs()
            self._initialized = True

    def _ensure_db_dir(self) -> None:
        EXEC_DB_DIR.mkdir(parents=True, exist_ok=True)

    def _load_existing_jobs(self) -> None:
        if not EXEC_DB_FILE.exists():
            return
        try:
            with open(EXEC_DB_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        invocation_id = record.get("invocation_id")
                        job_id = record.get("job_id")
                        executor = record.get("executor")
                        data = record.get("data", {})
                        config = record.get("config", {})
                        timestamp = record.get("timestamp", 0.0)
                        if invocation_id and job_id and executor:
                            job_data = JobData(
                                invocation_id=invocation_id,
                                job_id=job_id,
                                timestamp=timestamp,
                                executor=executor,
                                data=data,
                                config=config,
                            )
                            self._jobs[job_id] = job_data
                            if invocation_id not in self._invocations:
                                self._invocations[invocation_id] = []
                            self._invocations[invocation_id].append(job_id)
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSONL line", error=str(e))
        except OSError as e:
            logger.warning("Failed to load existing jobs", error=str(e))

    def write_job(self, job: JobData) -> None:
        if job.job_id:
            self._jobs[job.job_id] = job
        if job.invocation_id not in self._invocations:
            self._invocations[job.invocation_id] = []
        if job.job_id and job.job_id not in self._invocations[job.invocation_id]:
            self._invocations[job.invocation_id].append(job.job_id)
        record = asdict(job)
        try:
            with open(EXEC_DB_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(
                "Job written to execution database",
                invocation_id=job.invocation_id,
                job_id=job.job_id,
                executor=job.executor,
            )
        except OSError as e:
            logger.error(
                "Failed to write job to database",
                invocation_id=job.invocation_id,
                job_id=job.job_id,
                error=str(e),
            )
            raise

    def _resolve_invocation_id(self, short_id: str) -> Optional[str]:
        """Resolve a short invocation ID to the full one.

        Args:
            short_id: Partial or full invocation ID.

        Returns:
            Full invocation ID if found uniquely, None if not found.

        Raises:
            ValueError: If the short_id matches multiple invocation IDs.
        """
        if not short_id:
            return None

        short_id = short_id.lower()

        # NOTE(agronskiy): this is a non-optimized implementation that assumes small amount
        # of jobs in ExecDB(), a typical scenario. Speeding up would involve building a
        # prefix tree when loading invocations/jobs.
        matches = [
            inv_id
            for inv_id in self._invocations.keys()
            if inv_id.lower().startswith(short_id)
        ]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous invocation ID '{short_id}': matches {matches}")
        else:
            return None

    def _resolve_job_id(self, short_job_id: str) -> Optional[str]:
        """Resolve a short job ID to the full one.

        Args:
            short_job_id: Partial or full job ID.

        Returns:
            Full job ID if found uniquely, None if not found.

        Raises:
            ValueError: If the short_job_id matches multiple job IDs.
        """
        if not short_job_id:
            return None

        # Normalize to lowercase for case-insensitive matching
        short_job_id = short_job_id.lower()

        if "." in short_job_id:
            parts = short_job_id.split(".", 1)
            short_inv_id, job_index = parts[0], parts[1]

            # Resolve the invocation part
            full_inv_id = self._resolve_invocation_id(short_inv_id)
            if full_inv_id:
                candidate_job_id = f"{full_inv_id}.{job_index}"
                if candidate_job_id in self._jobs:
                    return candidate_job_id

        # NOTE(agronskiy): unfortunately, due to legacy, there exist usecases where
        # job_id is the same format as invocation_id
        candidate_job_id = self._resolve_invocation_id(short_job_id)
        if candidate_job_id and candidate_job_id in self._jobs:
            return candidate_job_id

        return None

    def get_job(self, job_id: str) -> Optional[JobData]:
        """Get job by full or partial job ID.

        Args:
            job_id: Full or partial job ID.

        Returns:
            JobData if found, None otherwise.

        Raises:
            ValueError: If the job_id matches multiple jobs.
        """
        resolved_id = self._resolve_job_id(job_id)
        if resolved_id:
            return self._jobs.get(resolved_id)

        return None

    def get_jobs(self, invocation_id: str) -> Dict[str, JobData]:
        """Get all jobs for a full or partial invocation ID.

        Args:
            invocation_id: Full or partial invocation ID.

        Returns:
            Dictionary mapping job_id to JobData for all jobs in the invocation.

        Raises:
            ValueError: If the invocation_id matches multiple invocations.
        """
        resolved_inv_id = self._resolve_invocation_id(invocation_id)
        if not resolved_inv_id:
            return {}

        job_ids = self._invocations.get(resolved_inv_id, [])
        return {
            job_id: self._jobs[job_id] for job_id in job_ids if job_id in self._jobs
        }

    def get_invocation_jobs(self, invocation_id: str) -> List[str]:
        """Get job IDs for a full or partial invocation ID.

        Args:
            invocation_id: Full or partial invocation ID.

        Returns:
            List of job IDs for the invocation.

        Raises:
            ValueError: If the invocation_id matches multiple invocations.
        """
        resolved_inv_id = self._resolve_invocation_id(invocation_id)
        if not resolved_inv_id:
            return []
        return self._invocations.get(resolved_inv_id, [])

    def get_all_jobs(self) -> Dict[str, JobData]:
        """Return a copy of all jobs in the execution DB."""
        return dict(self._jobs)


# Ensure all the paths
_DB = ExecutionDB()
