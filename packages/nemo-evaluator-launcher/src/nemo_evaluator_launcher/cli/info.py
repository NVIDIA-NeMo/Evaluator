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

"""Job information helper functionalities for nemo-evaluator-launcher."""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from simple_parsing import field

from nemo_evaluator_launcher.cli.version import Cmd as VersionCmd
from nemo_evaluator_launcher.common.execdb import EXEC_DB_FILE, ExecutionDB, JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.utils import copy_artifacts as copy_artifacts_fn


def get_job_paths(job_data: JobData) -> Dict[str, Any]:
    """Get result paths based on executor type from job metadata."""

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


@dataclass
class InfoCmd:
    """Job information functionalities for nemo-evaluator-launcher.

    Examples:
      nemo-evaluator-launcher info <inv>                 # Full job info
      nemo-evaluator-launcher info <inv> --config        # Show stored job config (YAML)
      nemo-evaluator-launcher info <inv> --artifacts     # Show artifact locations and key files
      nemo-evaluator-launcher info <inv> --logs          # Show log locations and key files
      nemo-evaluator-launcher info <inv> --copy-logs <DIR>       # Copy logs to <DIR>
      nemo-evaluator-launcher info <inv> --copy-artifacts <DIR>  # Copy artifacts to <DIR>

    Notes:
      - Supports invocation IDs and job IDs (space-separated)
      - Shows local or remote paths depending on executor (local/slurm/lepton)
      - Copy operations work for both local and remote jobs (expect longer time for remote jobs)
      - Copy operations are not supported for Lepton executor (yet).
    """

    invocation_ids: List[str] = field(
        positional=True,
        help="IDs to show info for (space-separated). Accepts invocation IDs or/and job IDs.",
    )

    # info modes
    config: bool = field(
        default=False, action="store_true", help="Show job configuration"
    )
    artifacts: bool = field(
        default=False, action="store_true", help="Show artifact locations and key files"
    )
    logs: bool = field(
        default=False, action="store_true", help="Show log locations and key files"
    )

    # copy operations - work for both local and remote jobs
    copy_logs: str | None = field(
        default=None,
        alias=["--copy-logs"],
        help="Copy logs to a local directory",
        metavar="DIR",
    )
    copy_artifacts: str | None = field(
        default=None,
        alias=["--copy-artifacts"],
        help="Copy artifacts to a local directory",
        metavar="DIR",
    )

    def execute(self) -> None:
        VersionCmd().execute()
        logger.debug("Info command started", invocation_ids=self.invocation_ids)

        if not self.invocation_ids:
            logger.error("No job or invocation IDs provided.")
            raise ValueError("No job or invocation IDs provided.")

        jobs = self._resolve_jobs()
        logger.debug(
            "Resolved jobs",
            total_ids=len(self.invocation_ids),
            valid_jobs=len(jobs),
            job_ids=[jid for jid, _ in jobs],
        )
        if not jobs:
            print(
                "No valid jobs found (jobs may have been deleted or IDs may be incorrect)."
            )
            return

        # show ops
        if self.config:
            self._show_config_info(jobs)
        if self.logs:
            self._show_logs_info(jobs)
        if self.artifacts:
            self._show_artifacts_info(jobs)

        # TODO(martas): why do we need to check this? we should just use args values
        if self.copy_logs:
            logger.info(
                "Copying logs to local directory",
                dest_dir=self.copy_logs,
                job_count=len(jobs),
            )
            self._copy_content(
                jobs, dest_dir=self.copy_logs, copy_logs=True, copy_artifacts=False
            )
        if self.copy_artifacts:
            logger.info(
                "Copying artifacts to local directory",
                dest_dir=self.copy_artifacts,
                job_count=len(jobs),
            )
            self._copy_content(
                jobs, dest_dir=self.copy_artifacts, copy_logs=False, copy_artifacts=True
            )

        # default view when no flags
        if not any(
            [
                self.config,
                self.logs,
                self.artifacts,
                self.copy_logs,
                self.copy_artifacts,
            ]
        ):
            self._show_invocation_info(jobs)

    def _resolve_jobs(self) -> List[Tuple[str, JobData]]:
        """Resolve jobs from ExecDB using IDs (job IDs and/or invocation IDs)."""
        db = ExecutionDB()
        found: list[tuple[str, JobData]] = []
        for id_or_prefix in self.invocation_ids:
            if "." in id_or_prefix:
                jd = db.get_job(id_or_prefix)
                if jd:
                    found.append((jd.job_id, jd))
            else:
                for jid, jd in db.get_jobs(id_or_prefix).items():
                    found.append((jid, jd))
        # deduplicate and stable sort
        seen: set[str] = set()
        uniq: list[tuple[str, JobData]] = []
        for jid, jd in found:
            if jid not in seen:
                seen.add(jid)
                uniq.append((jid, jd))
        return sorted(uniq, key=lambda p: p[0])

    def _show_invocation_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        inv = jobs[0][1].invocation_id if jobs else None
        print(
            f"Job information for {len(jobs)} job(s){f' under invocation {inv}' if inv else ''}:\n"
        )

        for job_id, job_data in jobs:
            self._show_job_info(job_id, job_data)
            print()

        # footer hint: where to find more metadata
        print(
            "For more details about this run, inspect the Execution DB under your home dir:"
        )
        print(f"Path: {EXEC_DB_FILE}")
        if inv:
            print(f"├── Lookup key: invocation_id={inv}")

        # Next steps hint
        print("\nNext steps:")
        print("  - Use --logs to show log locations.")
        print("  - Use --artifacts to show artifact locations.")
        print("  - Use --config to show stored job configuration (YAML).")
        print(
            "  - Use --copy-logs [DIR] to copy logs to a local directory (works for local and remote jobs)."
        )
        print(
            "  - Use --copy-artifacts [DIR] to copy artifacts to a local directory (works for local and remote jobs)."
        )

    def _show_job_info(self, job_id: str, job_data: JobData) -> None:
        print(f"Job {job_id}")

        # metadata
        try:
            when = datetime.fromtimestamp(job_data.timestamp).isoformat(
                timespec="seconds"
            )
        except Exception:
            when = str(job_data.timestamp)
        print(f"├── Executor: {job_data.executor}")
        print(f"├── Created: {when}")

        idx = int(job_id.split(".")[-1])
        tasks = (job_data.config or {}).get("evaluation", {}).get("tasks", [])
        if idx >= len(tasks):
            logger.error(
                f"Job task index {job_id} is larger than number of tasks {len(tasks)} in invocation"
            )
        else:
            task_name = tasks[idx].get("name")
            if task_name:
                print(f"├── Task: {task_name}")

        # Determine executor type for file descriptions
        cfg_exec_type = ((job_data.config or {}).get("execution") or {}).get("type")
        exec_type = (job_data.executor or cfg_exec_type or "").lower()

        # locations via exporter helper
        paths = get_job_paths(job_data)

        # Artifacts with file descriptions
        artifacts_list = _get_artifacts_file_list()
        if paths.get("storage_type") == "remote_ssh":
            artifacts_path = f"{paths['username']}@{paths['hostname']}:{paths['remote_path']}/artifacts"
            print(f"├── Artifacts: {artifacts_path} (remote)")
            print("│   └── Key files:")
            for filename, desc in artifacts_list:
                print(f"│       ├── {filename} - {desc}")
        else:
            ap = paths.get("artifacts_dir")
            if ap:
                exists = self._check_path_exists(paths, "artifacts")
                print(f"├── Artifacts: {ap} {exists} (local)")
                print("│   └── Key files:")
                for filename, desc in artifacts_list:
                    print(f"│       ├── {filename} - {desc}")

        # Logs with file descriptions
        logs_list = _get_log_file_list(exec_type)
        if paths.get("storage_type") == "remote_ssh":
            logs_path = (
                f"{paths['username']}@{paths['hostname']}:{paths['remote_path']}/logs"
            )
            print(f"├── Logs: {logs_path} (remote)")
            print("│   └── Key files:")
            for filename, desc in logs_list:
                print(f"│       ├── {filename} - {desc}")
        else:
            lp = paths.get("logs_dir")
            if lp:
                exists = self._check_path_exists(paths, "logs")
                print(f"├── Logs: {lp} {exists} (local)")
                print("│   └── Key files:")
                for filename, desc in logs_list:
                    print(f"│       ├── {filename} - {desc}")

        # executor-specific
        d = job_data.data or {}
        cfg_exec_type = ((job_data.config or {}).get("execution") or {}).get("type")
        exec_type = (job_data.executor or cfg_exec_type or "").lower()

        if exec_type == "slurm":
            sj = d.get("slurm_job_id")
            if sj:
                print(f"├── Slurm Job ID: {sj}")
        elif exec_type == "gitlab":
            pid = d.get("pipeline_id")
            if pid:
                print(f"├── Pipeline ID: {pid}")
        elif exec_type == "lepton":
            jn = d.get("lepton_job_name")
            if jn:
                print(f"├── Lepton Job: {jn}")
            en = d.get("endpoint_name")
            if en:
                print(f"├── Endpoint: {en}")
            eu = d.get("endpoint_url")
            if eu:
                print(f"├── Endpoint URL: {eu}")

    def _show_logs_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        print("Log locations:\n")
        for job_id, job_data in jobs:
            paths = get_job_paths(job_data)
            cfg_exec_type = ((job_data.config or {}).get("execution") or {}).get("type")
            exec_type = (job_data.executor or cfg_exec_type or "").lower()
            logs_list = _get_log_file_list(exec_type)

            if paths.get("storage_type") == "remote_ssh":
                logs_path = f"ssh://{paths['username']}@{paths['hostname']}{paths['remote_path']}/logs"
                print(f"{job_id}: {logs_path} (remote)")
                print("  └── Key files:")
                for filename, desc in logs_list:
                    print(f"      ├── {filename} - {desc}")
            else:
                lp = paths.get("logs_dir")
                if lp:
                    exists = self._check_path_exists(paths, "logs")
                    print(f"{job_id}: {lp} {exists} (local)")
                    print("  └── Key files:")
                    for filename, desc in logs_list:
                        print(f"      ├── {filename} - {desc}")

    def _show_artifacts_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        print("Artifact locations:\n")
        for job_id, job_data in jobs:
            paths = get_job_paths(job_data)
            artifacts_list = _get_artifacts_file_list()

            if paths.get("storage_type") == "remote_ssh":
                artifacts_path = f"ssh://{paths['username']}@{paths['hostname']}{paths['remote_path']}/artifacts"
                print(f"{job_id}: {artifacts_path} (remote)")
                print("  └── Key files:")
                for filename, desc in artifacts_list:
                    print(f"      ├── {filename} - {desc}")
            else:
                ap = paths.get("artifacts_dir")
                if ap:
                    exists = self._check_path_exists(paths, "artifacts")
                    print(f"{job_id}: {ap} {exists} (local)")
                    print("  └── Key files:")
                    for filename, desc in artifacts_list:
                        print(f"      ├── {filename} - {desc}")

    def _show_config_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        for job_id, job_data in jobs:
            print(f"Configuration for {job_id}:")
            if job_data.config:
                import yaml

                config_yaml = yaml.dump(
                    job_data.config, default_flow_style=False, indent=2
                )
                print(config_yaml)
            else:
                print("  No configuration stored for this job.")
            print()

    def _copy_content(
        self,
        jobs: List[Tuple[str, JobData]],
        dest_dir: str,
        copy_logs: bool,
        copy_artifacts: bool,
    ) -> None:
        _, failed_jobs = copy_artifacts_fn(
            [job for id, job in jobs],
            Path(dest_dir),
            copy_local=True,
            only_required=False,
            copy_logs=copy_logs,
            copy_artifacts=copy_artifacts,
        )
        if len(failed_jobs) == 0:
            logger.info(
                "Content copy completed successfully",
                dest_dir=dest_dir,
                job_count=len(jobs),
            )
        else:
            logger.warning(
                "Content copy for some of the jobs failed",
                failed_jobs=failed_jobs,
                dest_dir=dest_dir,
            )

    def _check_path_exists(self, paths: Dict[str, Any], path_type: str) -> str:
        """Check if a path exists and return indicator."""
        try:
            if paths.get("storage_type") == "remote_ssh":
                # For remote paths, we can't easily check existence
                return "(remote)"
            elif path_type == "logs" and "logs_dir" in paths:
                logs_dir = Path(paths["logs_dir"])
                return "(exists)" if logs_dir.exists() else "(not found)"
            elif path_type == "artifacts" and "artifacts_dir" in paths:
                artifacts_dir = Path(paths["artifacts_dir"])
                return "(exists)" if artifacts_dir.exists() else "(not found)"
        except Exception:
            pass
        return ""


# Helper functions for file descriptions (based on actual code and content analysis)
def _get_artifacts_file_list() -> list[tuple[str, str]]:
    """Files generated in artifacts/."""
    return [
        (
            "results.yml",
            "Benchmark scores, task results and resolved run configuration.",
        ),
        (
            "eval_factory_metrics.json",
            "Response + runtime stats (latency, tokens count, memory)",
        ),
        ("metadata.yaml", "Metadata about the evaluation run"),
        (
            "run_config.yml",
            "Nemo-Evaluator run configuration used inside the evaluation container",
        ),
        ("report.html", "Request-Response Pairs samples in HTML format (if enabled)"),
        ("report.json", "Report data in json format, if enabled"),
    ]


def _get_log_file_list(executor_type: str) -> list[tuple[str, str]]:
    """Files actually generated in logs/ - executor-specific."""
    et = (executor_type or "local").lower()
    if et == "slurm":
        return [
            ("client-{SLURM_JOB_ID}.log", "Evaluation container/process output"),
            (
                "slurm-{SLURM_JOB_ID}.log",
                "SLURM scheduler stdout/stderr (batch submission, export steps).",
            ),
            (
                "server-{SLURM_JOB_ID}.out",
                "Model server logs when a deployment is used.",
            ),
        ]
    # local executor
    return [
        (
            "stdout.log",
            "Global log file for the evaluation run.",
        ),
        ("client_stdout.log", "Evaluation container/process output"),
        ("server_stdout.log", "Model server logs when a deployment is used."),
    ]
