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

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData, generate_job_id
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.utils import (
    DataForExport,
    ExportResult,
    extract_accuracy_metrics,
    get_model_id,
    load_benchmark_info,
    load_config_from_metadata,
    ssh_cleanup_masters,
    ssh_download_artifacts,
    ssh_setup_masters,
)


class BaseExporter(ABC):
    """Base interface for result exporters."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.job_dirs = [Path(dir) for dir in self.config.get("job_dirs", [])]
        self.copy_logs = self.config.get("copy_logs", False)
        self.copy_artifacts = self.config.get("copy_artifacts", True)
        self.only_required = self.config.get("only_required", True)
        for job_dir in self.job_dirs:
            if not job_dir.exists():
                raise FileNotFoundError(f"Job directory {job_dir} not found")
        self.db = ExecutionDB()

    def export(self, invocation_or_job_ids: List[str]) -> ExportResult:
        failed_jobs = []

        # retrieve jobs data from execution db or job directories
        jobs_data, retrieved_failed_jobs = self._get_jobs_data(invocation_or_job_ids)
        failed_jobs.extend(retrieved_failed_jobs)

        # copy remote artifacts to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            jobs_data, copy_failed_jobs = self._copy_remote_artifacts(
                jobs_data, export_dir=temp_dir
            )
            failed_jobs.extend(copy_failed_jobs)

            # prepare data for export
            data_for_export_jobs = []
            for job_id, job_data in jobs_data.items():
                data_for_export = self.prepare_data_for_export(job_data)
                if data_for_export is None:
                    failed_jobs.append(job_id)
                    continue
                else:
                    data_for_export_jobs.append(data_for_export)

            # export jobs
            successful_jobs, export_failed_jobs, skipped_jobs = self.export_jobs(
                data_for_export_jobs
            )
            failed_jobs.extend(export_failed_jobs)

        # return result
        return ExportResult(
            successful_jobs=successful_jobs,
            failed_jobs=failed_jobs,
            skipped_jobs=skipped_jobs,
        )

    def _get_jobs_data(
        self, invocation_or_job_ids: List[str]
    ) -> Tuple[Dict[str, JobData], List[str]]:
        # TODO(martas): add test verifying that all ids are in the result,
        # including the ones not found in the db nor in job_dirs
        jobs_data = {}
        failed_jobs = []
        missing_ids = set()

        # first try to get jobs from db
        for id in invocation_or_job_ids:
            if "." in id:
                job_data = self.db.get_job(id)
                if job_data:
                    jobs_data[id] = job_data
                else:
                    missing_ids.add(id)
            else:
                inv_jobs = self.db.get_jobs(id)
                if inv_jobs:
                    jobs_data.update(inv_jobs)
                else:
                    missing_ids.add(id)

        # fallback to job_dirs if job not found in db
        invocations_to_search = {id.split(".")[0] for id in missing_ids}
        job_data_from_dirs = {}
        for invocation_id in invocations_to_search:
            found_dirs = []
            for job_dir in self.job_dirs:
                # we assume that the invocation ID is part of the job directory name
                # currently we don't have job ID in the name, so we need to infer jobs' order from config
                # we do it in _get_jobs_in_dir function
                found_dirs += job_dir.glob(f"*{invocation_id}*")
            if len(found_dirs) > 1:
                raise ValueError(
                    f"Multiple directories found for invocation {invocation_id}: {found_dirs}"
                )
            elif found_dirs:
                job_data_from_dirs[invocation_id] = self._get_jobs_in_dir(found_dirs[0])

        for id in missing_ids:
            if "." in id:
                # get data for single job
                invocation_id = id.split(".")[0]
                if (
                    invocation_id not in job_data_from_dirs
                    or id not in job_data_from_dirs[invocation_id]
                ):
                    logger.error(
                        f"Job {id} not found in ExecutionDB nor job directories"
                    )
                    failed_jobs.append(id)
                else:
                    jobs_data[id] = job_data_from_dirs[invocation_id][id]
            else:
                # get data for all jobs in invocation
                if id not in job_data_from_dirs:
                    logger.error(
                        f"Invocation {id} not found in ExecutionDB nor job directories"
                    )
                    failed_jobs.append(id)
                else:
                    jobs_data.update(job_data_from_dirs[id])

        return jobs_data, failed_jobs

    def _get_jobs_in_dir(self, invocation_dir: Path) -> Dict[str, JobData]:
        """Extract job data from an invocation directory.

        Args:
            invocation_dir: Path to invocation directory (e.g., nel-results/20251219_112732-f6926cc47f65e4ed/)

        Returns:
            Dictionary mapping job_id to JobData for all jobs in the invocation
        """
        jobs_data = {}

        # Extract invocation_id from directory name (format: YYYYMMDD_HHMMSS-<invocation_id>)
        dir_name = invocation_dir.name
        if "-" not in dir_name:
            return jobs_data

        invocation_id = dir_name.split("-", 1)[1]

        # Find all subdirectories with artifacts/metadata.yaml
        job_subdirs = []
        for subdir in invocation_dir.iterdir():
            if subdir.is_dir():
                metadata_file = subdir / "artifacts" / "metadata.yaml"
                if metadata_file.exists():
                    job_subdirs.append((subdir, metadata_file))

        if not job_subdirs:
            return jobs_data

        # Load config from first metadata file to get task order
        # All jobs in the same invocation share the same config
        first_metadata = job_subdirs[0][1]
        try:
            config = load_config_from_metadata(first_metadata.parent)
        except Exception as e:
            logger.warning(f"Failed to load config from {first_metadata}: {e}")
            return jobs_data

        # Get task names from config to determine job indices
        tasks = config.get("evaluation", {}).get("tasks", [])
        task_names = [
            task.get("name") if isinstance(task, dict) else task for task in tasks
        ]

        # Get executor type
        executor_type = config.get("execution", {}).get("type", "unknown")

        # Create JobData for each job subdirectory
        for job_subdir, metadata_file in job_subdirs:
            task_name = job_subdir.name

            # Find task index in config
            try:
                task_index = task_names.index(task_name)
            except ValueError:
                # Task not found in config, skip or use fallback
                logger.warning(f"Task {task_name} not found in config tasks list")
                continue

            job_id = generate_job_id(invocation_id, task_index)

            # Build job data dict
            job_data_dict = {
                "output_dir": str(job_subdir),
                "in_database": False,
            }

            # Create JobData object
            job_data = JobData(
                invocation_id=invocation_id,
                job_id=job_id,
                timestamp=invocation_dir.stat().st_mtime,  # Use directory modification time
                executor=executor_type,
                data=job_data_dict,
                config=config,
            )

            jobs_data[job_id] = job_data

        return jobs_data

    def _copy_remote_artifacts(
        self, jobs_data: Dict[str, JobData], export_dir: Path
    ) -> Tuple[Dict[str, JobData], List[str]]:
        """Copy artifacts to local filesystem. Returns list of failed jobs."""
        jobs_to_copy = []
        failed_jobs = []
        prepared_jobs = {}

        for job_id, job_data in jobs_data.items():
            if "output_dir" not in job_data.data:
                if "remote_rundir_path" in job_data.data:
                    if (
                        "username" not in job_data.data
                        or "hostname" not in job_data.data
                    ):
                        raise ValueError(
                            f"Username or hostname not found for remote job: {job_id}"
                        )
                    jobs_to_copy.append(job_data)
                else:
                    logger.warning(
                        f"Job {job_id} has no output directory and is not remote. Could not export job."
                    )
                    continue
            else:
                prepared_jobs[job_id] = job_data

        if not jobs_to_copy:
            return jobs_data, failed_jobs

        remotes = {
            (job_data.data["username"], job_data.data["hostname"])
            for job_data in jobs_to_copy
        }
        cp = ssh_setup_masters(list(remotes))
        try:
            for job_data in jobs_to_copy:
                job_local_dir = export_dir / job_data.job_id
                exported_files = ssh_download_artifacts(
                    job_data.data["username"],
                    job_data.data["hostname"],
                    job_data.data["remote_rundir_path"],
                    job_local_dir,
                    copy_logs=self.copy_logs,
                    copy_artifacts=self.copy_artifacts,
                    only_required=self.only_required,
                    control_paths=cp,
                )
                if len(exported_files) == 0:
                    logger.warning(
                        f"No artifacts copied for remote job {job_data.job_id}. Could not export job."
                    )
                    failed_jobs.append(job_data.job_id)
                    continue
                job_data.data["output_dir"] = str(job_local_dir)
                prepared_jobs[job_data.job_id] = job_data
        finally:
            ssh_cleanup_masters(cp)
        return prepared_jobs, failed_jobs

    @abstractmethod
    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export a prepared data for all requested jobs and invocations.
        It takes list of DataForExport objects as some exporters (e.g. local)
        combine jobs from different invocations into a single export.

        Returns list of successful jobs, list of failed jobs, list of skipped jobs."""
        raise NotImplementedError("Each exporter must implement this method")

    def prepare_data_for_export(self, job_data: JobData) -> DataForExport | None:
        """Prepare data for export from artifacts directory and logs directory.
        It assumes that the artifacts and logs are already copied to the local filesystem."""

        if "output_dir" not in job_data.data:
            logger.error(
                f"Output directory not found in job data for job {job_data.job_id}"
            )
            return None

        artifacts_dir = Path(job_data.data["output_dir"]) / "artifacts"
        if self.copy_logs:
            logs_dir = Path(job_data.data["output_dir"]) / "logs"
        else:
            logs_dir = None

        if not artifacts_dir.exists():
            logger.error(
                f"Artifacts directory {artifacts_dir} not found for job {job_data.job_id}"
            )
            return None

        if logs_dir and not logs_dir.exists():
            logger.error(
                f"Logs directory {logs_dir} not found for job {job_data.job_id}"
            )
            return None

        metrics = extract_accuracy_metrics(artifacts_dir)
        harness, task = load_benchmark_info(artifacts_dir)
        container = job_data.data.get("eval_image", None)
        model_id = get_model_id(artifacts_dir)

        return DataForExport(
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            config=job_data.config,
            model_id=model_id,
            metrics=metrics,
            harness=harness,
            task=task,
            container=container,
            executor=job_data.executor,
            invocation_id=job_data.invocation_id,
            job_id=job_data.job_id,
            timestamp=job_data.timestamp,
            job_data=job_data.data,
        )
