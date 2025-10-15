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

"""Debugging helper functionalities for nemo-evaluator-launcher."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simple_parsing import field

from nemo_evaluator_launcher.cli.export import ExportCmd
from nemo_evaluator_launcher.cli.version import Cmd as VersionCmd
from nemo_evaluator_launcher.common.execdb import EXEC_DB_FILE, ExecutionDB, JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.local import LocalExporter
from nemo_evaluator_launcher.exporters.utils import get_task_name

# Local exporter helper to copy logs and artifacts
_EXPORT_HELPER = LocalExporter({})


@dataclass
class DebugCmd(ExportCmd):
    """Debugging functionalities for nemo-evaluator-launcher.

    Examples:
      nemo-evaluator-launcher debug <inv>                 # Full debug info
      nemo-evaluator-launcher debug <inv> --config        # Show stored job config (YAML)
      nemo-evaluator-launcher debug <inv> --artifacts     # Show artifact locations
      nemo-evaluator-launcher debug <inv> --logs          # Show log locations
      nemo-evaluator-launcher debug <inv> --copy-logs <path>       # Copy logs (default: current dir)
      nemo-evaluator-launcher debug <inv> --copy-artifacts <path>   # Copy artifacts (default: current dir)

    Notes:
      - Supports invocation IDs and job IDs
      - Shows local or remote paths depending on executor (local/slurm/lepton)
    """

    # local exporter destination defaults to local
    dest: str = field(default="local", init=False)

    # debug modes
    config: bool = field(default=False, help="Show job configuration")
    artifacts: bool = field(default=False, help="Show artifact locations")
    logs: bool = field(default=False, help="Show log locations")

    # copy operations
    copy_logs: Optional[str] = field(
        default=None,
        alias=["--copy-logs"],
        nargs="?",
        help="Copy logs to local directory (default: current dir)",
    )
    copy_artifacts: Optional[str] = field(
        default=None,
        alias=["--copy-artifacts"],
        nargs="?",
        help="Copy artifacts to local directory (default: current dir)",
    )

    def execute(self) -> None:
        # show version
        VersionCmd().execute()

        logger.info("Debug command started", invocation_ids=self.invocation_ids)

        if not self.invocation_ids:
            logger.error("No invocation IDs provided")
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
                "No valid jobs found (jobs may have been deletedd or IDs may be incorrect)."
            )
            return

        if self.config:
            logger.info("Showing job configuration", job_count=len(jobs))
            self._show_config_info(jobs)
        elif self.logs:
            logger.info("Showing job logs locations", job_count=len(jobs))
            self._show_logs_info(jobs)
        elif self.artifacts:
            logger.info("Showing artifacts locations", job_count=len(jobs))
            self._show_artifacts_info(jobs)
        elif self.copy_logs is not None:
            dest = self.copy_logs or "."
            if not self.copy_logs:
                print(
                    "No destination provided for --copy-logs; defaulting to current dir"
                )
            logger.info(
                "Copying logs to local directory", dest_dir=dest, job_count=len(jobs)
            )
            self._copy_logs(jobs, dest)
        elif self.copy_artifacts is not None:
            dest = self.copy_artifacts or "."
            if not self.copy_artifacts:
                print(
                    "No destination provided for --copy-artifacts; defaulting to current dir)"
                )
            logger.info(
                "Copying artifacts to local directory",
                dest_dir=dest,
                job_count=len(jobs),
            )
            self._copy_artifacts(jobs, dest)
        else:
            logger.info(
                "Job metadata details",
                invocation_id=jobs[0][1].invocation_id if jobs else None,
                jobs=len(jobs),
            )
            self._show_invocation_debug_info(jobs)

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

    def _show_invocation_debug_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        inv = jobs[0][1].invocation_id if jobs else None
        logger.info("Debug information", jobs=len(jobs), invocation=inv)
        print(
            f"Debug information for {len(jobs)} job(s){f' under invocation {inv}' if inv else ''}:\n"
        )

        for job_id, job_data in jobs:
            self._show_job_debug_info(job_id, job_data)
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
        print("  - Use --copy-logs [DIR] to copy logs to a local directory.")
        print("  - Use --copy-artifacts [DIR] to copy artifacts to a local directory.")

    def _show_job_debug_info(self, job_id: str, job_data: JobData) -> None:
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

        # locations via exporter helper
        paths = _EXPORT_HELPER.get_job_paths(job_data)

        # Artifacts
        if paths.get("storage_type") == "remote_ssh":
            artifacts_path = f"{paths['username']}@{paths['hostname']}:{paths['remote_path']}/artifacts"
            logger.info("Artifacts", job_id=job_id, path=artifacts_path, remote=True)
            print(f"├── Artifacts: {artifacts_path} (remote)")
        else:
            ap = paths.get("artifacts_dir")
            if ap:
                exists = self._check_path_exists(paths, "artifacts")
                logger.info(
                    "Artifacts", job_id=job_id, path=str(ap), exists_indicator=exists
                )
                print(f"├── Artifacts: {ap} {exists} (local)")

        # Logs
        if paths.get("storage_type") == "remote_ssh":
            logs_path = (
                f"{paths['username']}@{paths['hostname']}:{paths['remote_path']}/logs"
            )
            logger.info("Logs", job_id=job_id, path=logs_path, remote=True)
            print(f"├── Logs: {logs_path} (remote)")
        else:
            lp = paths.get("logs_dir")
            if lp:
                exists = self._check_path_exists(paths, "logs")
                logger.info(
                    "Logs", job_id=job_id, path=str(lp), exists_indicator=exists
                )
                print(f"├── Logs: {lp} {exists} (local)")

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
        # local and others: paths already displayed above; no extra fields needed

    def _show_logs_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        logger.info("Log locations")
        print("Log locations:\n")
        for job_id, job_data in jobs:
            paths = _EXPORT_HELPER.get_job_paths(job_data)
            if paths.get("storage_type") == "remote_ssh":
                logs_path = f"ssh://{paths['username']}@{paths['hostname']}{paths['remote_path']}/logs"
                logger.info("Logs", job_id=job_id, path=logs_path, remote=True)
                print(f"{job_id}: {logs_path} (remote)")
            else:
                lp = paths.get("logs_dir")
                if lp:
                    exists = self._check_path_exists(paths, "logs")
                    logger.info(
                        "Logs", job_id=job_id, path=str(lp), exists_indicator=exists
                    )
                    print(f"{job_id}: {lp} {exists} (local)")

    def _show_artifacts_info(self, jobs: List[Tuple[str, JobData]]) -> None:
        logger.info("Artifact locations")
        print("Artifact locations:\n")
        for job_id, job_data in jobs:
            paths = _EXPORT_HELPER.get_job_paths(job_data)
            if paths.get("storage_type") == "remote_ssh":
                artifacts_path = f"ssh://{paths['username']}@{paths['hostname']}{paths['remote_path']}/artifacts"
                logger.info(
                    "Artifacts", job_id=job_id, path=artifacts_path, remote=True
                )
                print(f"{job_id}: {artifacts_path} (remote)")
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
