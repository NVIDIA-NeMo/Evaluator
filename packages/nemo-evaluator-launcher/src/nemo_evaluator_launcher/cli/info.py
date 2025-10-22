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

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from simple_parsing import field

from nemo_evaluator_launcher.cli.version import Cmd as VersionCmd
from nemo_evaluator_launcher.common.execdb import EXEC_DB_FILE, ExecutionDB, JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.local import LocalExporter
from nemo_evaluator_launcher.exporters.utils import get_task_name

# Local exporter helper to copy logs and artifacts
_EXPORT_HELPER = LocalExporter({})


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
        logger.info("Info command started", invocation_ids=self.invocation_ids)

        if not self.invocation_ids:
            logger.error("No job or invocation IDs provided.")
            raise ValueError("No job or invocation IDs provided.")

        jobs = self._resolve_jobs()
        logger.info(
            "Resolved jobs",
            total_ids=len(self.invocation_ids),
            valid_jobs=len(jobs),
            job_ids=[jid for jid, _ in jobs],
        )

        if not jobs:
            logger.info(
                "No valid jobs found (jobs may have been deleted or IDs may be incorrect)."
            )
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

        # copy ops
        args = sys.argv[1:]
        copy_logs_flag = "--copy-logs" in args
        copy_artifacts_flag = "--copy-artifacts" in args

        if copy_logs_flag:
            if self.copy_logs is None:
                raise ValueError("--copy-logs requires a directory path")
            if not self.copy_logs.strip():
                raise ValueError("--copy-logs requires a directory path")
            logger.info(
                "Copying logs to local directory",
                dest_dir=self.copy_logs,
                job_count=len(jobs),
            )
            self._copy_logs(jobs, self.copy_logs)

        if copy_artifacts_flag:
            if self.copy_artifacts is None:
                raise ValueError("--copy-artifacts requires a directory path")
            if not self.copy_artifacts.strip():
                raise ValueError("--copy-artifacts requires a directory path")
            logger.info(
                "Copying artifacts to local directory",
                dest_dir=self.copy_artifacts,
                job_count=len(jobs),
            )
            self._copy_artifacts(jobs, self.copy_artifacts)

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
            logger.info(
                "Job metadata details",
                invocation_id=jobs[0][1].invocation_id if jobs else None,
                jobs=len(jobs),
            )
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
        logger.info("Job information", jobs=len(jobs), invocation=inv)
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
        logger.info("Job", job_id=job_id)
        print(f"Job {job_id}")

        # metadata
        try:
            when = datetime.fromtimestamp(job_data.timestamp).isoformat(
                timespec="seconds"
            )
        except Exception:
            when = str(job_data.timestamp)
        logger.info("Executor", job_id=job_id, executor=job_data.executor)
        logger.info("Created", job_id=job_id, created=when)
        print(f"├── Executor: {job_data.executor}")
        print(f"├── Created: {when}")

        task_name = get_task_name(job_data)
        if task_name:
            logger.info("Task", job_id=job_id, name=task_name)
            print(f"├── Task: {task_name}")

        # Determine executor type for file descriptions
        cfg_exec_type = ((job_data.config or {}).get("execution") or {}).get("type")
        exec_type = (job_data.executor or cfg_exec_type or "").lower()

        # locations via exporter helper
        paths = _EXPORT_HELPER.get_job_paths(job_data)

        # Artifacts with file descriptions
        artifacts_list = _get_artifacts_file_list()
        if paths.get("storage_type") == "remote_ssh":
            artifacts_path = f"{paths['username']}@{paths['hostname']}:{paths['remote_path']}/artifacts"
            logger.info("Artifacts", job_id=job_id, path=artifacts_path, remote=True)
            print(f"├── Artifacts: {artifacts_path} (remote)")
            print("│   └── Key files:")
            for filename, desc in artifacts_list:
                print(f"│       ├── {filename} - {desc}")
        else:
            ap = paths.get("artifacts_dir")
            if ap:
                exists = self._check_path_exists(paths, "artifacts")
                logger.info(
                    "Artifacts", job_id=job_id, path=str(ap), exists_indicator=exists
                )
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
            logger.info("Logs", job_id=job_id, path=logs_path, remote=True)
            print(f"├── Logs: {logs_path} (remote)")
            print("│   └── Key files:")
            for filename, desc in logs_list:
                print(f"│       ├── {filename} - {desc}")
        else:
            lp = paths.get("logs_dir")
            if lp:
                exists = self._check_path_exists(paths, "logs")
                logger.info(
                    "Logs", job_id=job_id, path=str(lp), exists_indicator=exists
                )
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
        logger.info("Log locations")
        print("Log locations:\n")
        for job_id, job_data in jobs:
            paths = _EXPORT_HELPER.get_job_paths(job_data)
            cfg_exec_type = ((job_data.config or {}).get("execution") or {}).get("type")
            exec_type = (job_data.executor or cfg_exec_type or "").lower()
            logs_list = _get_log_file_list(exec_type)

            if paths.get("storage_type") == "remote_ssh":
                logs_path = f"ssh://{paths['username']}@{paths['hostname']}{paths['remote_path']}/logs"
                logger.info("Logs", job_id=job_id, path=logs_path, remote=True)
                print(f"{job_id}: {logs_path} (remote)")
                print("  └── Key files:")
                for filename, desc in logs_list:
                    print(f"      ├── {filename} - {desc}")
            else:
                lp = paths.get("logs_dir")
                if lp:
                    exists = self._check_path_exists(paths, "logs")
                    logger.info(
                        "Logs", job_id=job_id, path=str(lp), exists_indicator=exists
                    )
                    print(f"{job_id}: {lp} {exists} (local)")
                    print("  └── Key files:")
                    for filename, desc in logs_list:
                        print(f"      ├── {filename} - {desc}")

    def _show_artifacts_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        logger.info("Artifact locations")
        print("Artifact locations:\n")
        for job_id, job_data in jobs:
            paths = _EXPORT_HELPER.get_job_paths(job_data)
            artifacts_list = _get_artifacts_file_list()

            if paths.get("storage_type") == "remote_ssh":
                artifacts_path = f"ssh://{paths['username']}@{paths['hostname']}{paths['remote_path']}/artifacts"
                logger.info(
                    "Artifacts", job_id=job_id, path=artifacts_path, remote=True
                )
                print(f"{job_id}: {artifacts_path} (remote)")
                print("  └── Key files:")
                for filename, desc in artifacts_list:
                    print(f"      ├── {filename} - {desc}")
            else:
                ap = paths.get("artifacts_dir")
                if ap:
                    exists = self._check_path_exists(paths, "artifacts")
                    logger.info(
                        "Artifacts",
                        job_id=job_id,
                        path=str(ap),
                        exists_indicator=exists,
                    )
                    print(f"{job_id}: {ap} {exists} (local)")
                    print("  └── Key files:")
                    for filename, desc in artifacts_list:
                        print(f"      ├── {filename} - {desc}")

    def _show_config_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        for job_id, job_data in jobs:
            logger.info("Configuration for job", job_id=job_id)
            print(f"Configuration for {job_id}:")
            if job_data.config:
                import yaml

                config_yaml = yaml.dump(
                    job_data.config, default_flow_style=False, indent=2
                )
                logger.info("Configuration YAML", job_id=job_id, config=config_yaml)
                print(config_yaml)
            else:
                logger.info("No configuration stored for this job", job_id=job_id)
                print("  No configuration stored for this job.")
            print()

    def _copy_logs(self, jobs: List[Tuple[str, JobData]], dest_dir: str) -> None:
        """Copy logs using export functionality."""
        self._copy_content(jobs, dest_dir, copy_logs=True, copy_artifacts=False)

    def _copy_artifacts(self, jobs: List[Tuple[str, JobData]], dest_dir: str) -> None:
        """Copy artifacts using export functionality."""
        self._copy_content(jobs, dest_dir, copy_logs=False, copy_artifacts=True)

    def _copy_content(
        self,
        jobs: List[Tuple[str, JobData]],
        dest_dir: str,
        copy_logs: bool,
        copy_artifacts: bool,
    ) -> None:
        logger.debug(
            "Preparing export call",
            dest_dir=dest_dir,
            copy_logs=copy_logs,
            copy_artifacts=copy_artifacts,
            job_ids=[jid for jid, _ in jobs],
        )

        from nemo_evaluator_launcher.api.functional import export_results

        config = {
            "output_dir": dest_dir,
            "only_required": True,
            "copy_logs": bool(copy_logs) and not bool(copy_artifacts),
            "copy_artifacts": bool(copy_artifacts) and not bool(copy_logs),
        }
        # skip artifact validation
        if copy_logs and not copy_artifacts:
            config["skip_validation"] = True

        job_ids = [job_id for job_id, _ in jobs]
        kind = "logs" if copy_logs else "artifacts"
        logger.info(
            "Copying content", kind=kind, job_count=len(job_ids), dest_dir=dest_dir
        )
        print(f"Copying {kind} for {len(job_ids)} job(s) to {dest_dir}...")

        result = export_results(job_ids, "local", config)
        logger.debug("Export API call completed", success=result.get("success"))

        if result.get("success"):
            logger.info(
                "Content copy completed successfully",
                dest_dir=dest_dir,
                job_count=len(jobs),
            )
            if "jobs" in result:
                for jid, job_result in result["jobs"].items():
                    if job_result.get("success"):
                        print(f"{jid}: Success")
                    else:
                        print(
                            f"{jid}: Failed - {job_result.get('message', 'Unknown error')}"
                        )
            # Show full destination path
            full_dest_path = Path(dest_dir).resolve()
            print(f"Copied to: {full_dest_path}")
        else:
            err = result.get("error", "Unknown error")
            logger.warning("Content copy failed", error=err, dest_dir=dest_dir)
            print(f"Failed to copy {kind}: {err}")

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
        ("metrics.json", "Harness/benchmark metric and configuration"),
        ("report.html", "Request-Response Pairs samples in HTML format (if enabled)"),
        ("report.json", "Report data in json format, if enabled"),
    ]


def _get_log_file_list(executor_type: str) -> list[tuple[str, str]]:
    """Files actually generated in logs/ - executor-specific."""
    et = (executor_type or "local").lower()
    if et == "slurm":
        return [
            ("client-{SLURM_JOB_ID}.out", "Evaluation container/process output"),
            (
                "slurm-{SLURM_JOB_ID}.out",
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
            "Complete evaluation output (timestamps, resolved config, run/export messages).",
        ),
    ]
