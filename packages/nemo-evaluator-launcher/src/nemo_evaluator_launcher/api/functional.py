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
"""Public API functions for nemo-evaluator-launcher.

This module provides the main functional entry points for running evaluations, querying job status, and listing available tasks. These functions are intended to be used by CLI commands and external integrations.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.common.mapping import load_tasks_mapping
from nemo_evaluator_launcher.executors.registry import get_executor
from nemo_evaluator_launcher.exporters import create_exporter


def get_tasks_list() -> list[list[Any]]:
    """Get a list of available tasks from the mapping.

    Returns:
        list[list[Any]]: Each sublist contains task name, endpoint type, harness, and container.
    """
    mapping = load_tasks_mapping()
    data = [
        [
            task_data.get("task"),
            task_data.get("endpoint_type"),
            task_data.get("harness"),
            task_data.get("container"),
        ]
        for task_data in mapping.values()
    ]
    return data


def _validate_no_missing_values(cfg: Any, path: str = "") -> None:
    """Recursively validate that no MISSING values exist in the configuration.

    Args:
        cfg: The configuration object to validate.
        path: Current path in the configuration for error reporting.

    Raises:
        ValueError: If any MISSING values are found in the configuration.
    """
    if OmegaConf.is_dict(cfg):
        for key, value in cfg.items():
            current_path = f"{path}.{key!s}" if path else str(key)
            # Check if this specific key has a MISSING value
            if OmegaConf.is_missing(cfg, key):
                raise ValueError(
                    f"Configuration has MISSING value at path: {current_path!s}"
                )
            _validate_no_missing_values(value, current_path)
    elif OmegaConf.is_list(cfg):
        for i, value in enumerate(cfg):
            current_path = f"{path}[{i}]"
            _validate_no_missing_values(value, current_path)


def run_eval(cfg: RunConfig, dry_run: bool = False) -> Optional[str]:
    """Run evaluation with specified config and overrides.

    Args:
        cfg: The configuration object for the evaluation run.
        dry_run: If True, do not run the evaluation, just prepare scripts and save them.

    Returns:
        Optional[str]: The invocation ID for the evaluation run.

    Raises:
        ValueError: If configuration validation fails or MISSING values are found.
        RuntimeError: If the executor fails to start the evaluation.
    """
    # Validate that no MISSING values exist in the configuration
    _validate_no_missing_values(cfg)

    if dry_run:
        print(OmegaConf.to_yaml(cfg))

    _check_api_endpoint_when_deployment_is_configured(cfg)
    return get_executor(cfg.execution.type).execute_eval(cfg, dry_run)


def get_status(ids_or_prefixes: list[str]) -> list[dict[str, Any]]:
    """Get status of jobs by their IDs or invocation IDs.

    Args:
        job_ids: List of job IDs or invocation IDs to check status for. Short ones are allowed,
                 we would try to match the full ones from prefixes if no collisions are
                 present.

    Returns:
        list[dict[str, Any]]: List of status dictionaries for each job or invocation.
            Each dictionary contains keys: 'invocation', 'job_id', 'status', and 'data'.
            If a job or invocation is not found, status is 'not_found'.
            If an error occurs, status is 'error' and 'data' contains error details.
    """
    db = ExecutionDB()
    results: List[dict[str, Any]] = []

    for id_or_prefix in ids_or_prefixes:
        # If id looks like an invocation_id (no dot), get all jobs for it
        if "." not in id_or_prefix:
            jobs = db.get_jobs(id_or_prefix)
            if not jobs:
                results.append(
                    {
                        "invocation": id_or_prefix,
                        "job_id": None,
                        "status": "not_found",
                        "data": {},
                    }
                )
                continue

            # Get the executor class from the first job
            first_job_data = next(iter(jobs.values()))
            try:
                executor_cls = get_executor(first_job_data.executor)
            except ValueError as e:
                results.append(
                    {
                        "invocation": id_or_prefix,
                        "job_id": None,
                        "status": "error",
                        "data": {"error": str(e)},
                    }
                )
                continue

            # Get status from the executor for all jobs in the invocation
            try:
                status_list = executor_cls.get_status(id_or_prefix)

                # Create a result for each job in the invocation
                for job_id_in_invocation, job_data in jobs.items():
                    # Find the status for this specific job
                    job_status: str | None = None
                    job_progress: Optional[dict[str, Any]] = None
                    for status in status_list:
                        if status.id == job_id_in_invocation:
                            job_status = status.state.value
                            job_progress = status.progress
                            break

                    results.append(
                        {
                            "invocation": job_data.invocation_id,
                            "job_id": job_id_in_invocation,
                            "status": (
                                job_status if job_status is not None else "unknown"
                            ),
                            "progress": (
                                job_progress if job_progress is not None else "unknown"
                            ),
                            "data": job_data.data,
                        }
                    )

            except Exception as e:
                results.append(
                    {
                        "invocation": id_or_prefix,
                        "job_id": None,
                        "status": "error",
                        "data": {"error": str(e)},
                    }
                )
        else:
            # Otherwise, treat as job_id
            single_job_data: Optional[JobData] = db.get_job(id_or_prefix)

            if single_job_data is None:
                results.append(
                    {
                        "invocation": None,
                        "job_id": id_or_prefix,
                        "status": "not_found",
                        "data": {},
                    }
                )
                continue

            # Get the executor class
            try:
                executor_cls = get_executor(single_job_data.executor)
            except ValueError as e:
                results.append(
                    {
                        "invocation": None,
                        "job_id": id_or_prefix,
                        "status": "error",
                        "data": {"error": str(e)},
                    }
                )
                continue

            # Get status from the executor
            try:
                status_list = executor_cls.get_status(id_or_prefix)

                if not status_list:
                    results.append(
                        {
                            "invocation": single_job_data.invocation_id,
                            "job_id": single_job_data.job_id,
                            "status": "unknown",
                            "data": single_job_data.data,
                        }
                    )
                else:
                    # For individual job queries, return the first status
                    results.append(
                        {
                            "invocation": single_job_data.invocation_id,
                            "job_id": single_job_data.job_id,
                            "status": (
                                status_list[0].state.value if status_list else "unknown"
                            ),
                            "progress": (
                                status_list[0].progress if status_list else "unknown"
                            ),
                            "data": single_job_data.data,
                        }
                    )

            except Exception as e:
                results.append(
                    {
                        "invocation": (
                            single_job_data.invocation_id if single_job_data else None
                        ),
                        "job_id": (
                            single_job_data.job_id if single_job_data else id_or_prefix
                        ),
                        "status": "error",
                        "data": {"error": str(e)},
                    }
                )

    return results


def list_all_invocations_summary() -> list[dict[str, Any]]:
    """Return a concise per-invocation summary from the exec DB.

    Columns: invocation_id, earliest_job_ts, num_jobs, executor (or 'mixed').
    Sorted by earliest_job_ts (newest first).
    """
    db = ExecutionDB()
    jobs = db.get_all_jobs()

    inv_to_earliest: dict[str, float] = {}
    inv_to_count: dict[str, int] = {}
    inv_to_execs: dict[str, set[str]] = {}

    for jd in jobs.values():
        inv = jd.invocation_id
        ts = jd.timestamp or 0.0
        if inv not in inv_to_earliest or ts < inv_to_earliest[inv]:
            inv_to_earliest[inv] = ts
        inv_to_count[inv] = inv_to_count.get(inv, 0) + 1
        if inv not in inv_to_execs:
            inv_to_execs[inv] = set()
        inv_to_execs[inv].add(jd.executor)

    rows: list[dict[str, Any]] = []
    for inv, earliest_ts in inv_to_earliest.items():
        execs = inv_to_execs.get(inv, set())
        executor = (
            next(iter(execs)) if len(execs) == 1 else ("mixed" if execs else None)
        )
        rows.append(
            {
                "invocation_id": inv,
                "earliest_job_ts": earliest_ts,
                "num_jobs": inv_to_count.get(inv, 0),
                "executor": executor,
            }
        )

    rows.sort(key=lambda r: r.get("earliest_job_ts") or 0, reverse=True)
    return rows


def get_invocation_benchmarks(invocation_id: str) -> list[str]:
    """Return a sorted list of benchmark/task names for a given invocation.

    Extracted from stored configs in the execution DB. If anything goes wrong,
    returns an empty list; callers can display 'unknown' if desired.
    """
    db = ExecutionDB()
    jobs = db.get_jobs(invocation_id)
    names: set[str] = set()
    for jd in jobs.values():
        try:
            cfg = jd.config or {}
            tasks = (cfg.get("evaluation", {}) or {}).get("tasks", []) or []
            for t in tasks:
                n = t.get("name") if isinstance(t, dict) else None
                if n:
                    names.add(str(n))
        except Exception:
            # Ignore malformed entries; continue collecting from others
            continue
    return sorted(names)


def kill_job_or_invocation(id: str) -> list[dict[str, Any]]:
    """Kill a job or an entire invocation by its ID.

    Args:
        id: The job ID (e.g., aefc4819.0) or invocation ID (e.g., aefc4819) to kill.

    Returns:
        list[dict[str, Any]]: List of kill operation results.
            Each dictionary contains keys: 'invocation', 'job_id', 'status', and 'data'.
            If a job is not found, status is 'not_found'.
            If an error occurs, status is 'error' and 'data' contains error details.
    """
    db = ExecutionDB()
    results = []

    def kill_single_job(job_id: str, job_data: JobData) -> dict[str, Any]:
        """Helper function to kill a single job."""
        try:
            executor_cls = get_executor(job_data.executor)
            if hasattr(executor_cls, "kill_job"):
                executor_cls.kill_job(job_id)
                # Success - job was killed
                return {
                    "invocation": job_data.invocation_id,
                    "job_id": job_id,
                    "status": "killed",
                    "data": {"result": "Successfully killed job"},
                }
            else:
                return {
                    "invocation": job_data.invocation_id,
                    "job_id": job_id,
                    "status": "error",
                    "data": {
                        "error": f"Executor {job_data.executor} does not support killing jobs"
                    },
                }
        except (ValueError, RuntimeError) as e:
            # Expected errors from kill_job
            return {
                "invocation": job_data.invocation_id,
                "job_id": job_id,
                "status": "error",
                "data": {"error": str(e)},
            }
        except Exception as e:
            # Unexpected errors
            return {
                "invocation": job_data.invocation_id,
                "job_id": job_id,
                "status": "error",
                "data": {"error": f"Unexpected error: {str(e)}"},
            }

    # Determine if this is a job ID or invocation ID
    if "." in id:
        # This is a job ID - kill single job
        job_data = db.get_job(id)
        if job_data is None:
            return [
                {
                    "invocation": None,
                    "job_id": id,
                    "status": "not_found",
                    "data": {},
                }
            ]
        results.append(kill_single_job(id, job_data))
    else:
        # This is an invocation ID - kill all jobs in the invocation
        jobs = db.get_jobs(id)
        if not jobs:
            return [
                {
                    "invocation": id,
                    "job_id": None,
                    "status": "not_found",
                    "data": {},
                }
            ]

        # Kill each job in the invocation
        for job_id, job_data in jobs.items():
            results.append(kill_single_job(job_id, job_data))

    return results


def export_results(
    invocation_ids: Union[str, List[str]],
    dest: str = "local",
    config: dict[Any, Any] | None = None,
) -> dict:
    """Export results for one or more IDs (jobs/invocations/pipeline IDs) to a destination.

    Args:
        invocation_ids: Single invocation ID or list of invocation/job IDs
        dest: Export destination (local, wandb, jet, mlflow, gsheets)
        config: exporter configuration

    Returns:
        Export evaluation results dictionary
    """

    try:
        # Normalize to list
        if isinstance(invocation_ids, str):
            invocation_ids = [invocation_ids]

        exporter = create_exporter(dest, config or {})

        if len(invocation_ids) == 1:
            # Single id (job or invocation)
            single_id = invocation_ids[0]

            if "." in single_id:  # job_id
                # Try reading config from artifacts working dir (auto-export on remote node)
                cfg_file = None
                for name in ("run_config.yml", "config.yml"):
                    p = Path(name)
                    if p.exists():
                        cfg_file = p
                        break

                md_job_data = None
                if cfg_file:
                    try:
                        cfg_yaml = (
                            yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                        )

                        # Merge exporter override file if present
                        ypath_export = Path("export_config.yml")
                        if ypath_export.exists():
                            exp_yaml = (
                                yaml.safe_load(ypath_export.read_text(encoding="utf-8"))
                                or {}
                            )
                            exec_cfg = cfg_yaml.get("execution") or {}
                            auto_exp = (exp_yaml.get("execution") or {}).get(
                                "auto_export"
                            )
                            if auto_exp is not None:
                                exec_cfg["auto_export"] = auto_exp
                                cfg_yaml["execution"] = exec_cfg
                            if "export" in exp_yaml:
                                cfg_yaml["export"] = exp_yaml["export"]
                            if "evaluation" in exp_yaml and exp_yaml["evaluation"]:
                                eval_cfg = cfg_yaml.get("evaluation") or {}
                                eval_cfg.update(exp_yaml["evaluation"])
                                cfg_yaml["evaluation"] = eval_cfg

                        executor_name = (cfg_yaml.get("execution") or {}).get(
                            "type", "local"
                        )
                        md_job_data = JobData(
                            invocation_id=single_id.split(".")[0],
                            job_id=single_id,
                            timestamp=0.0,
                            executor=executor_name,  # ensures slurm tag is preserved
                            data={
                                "output_dir": str(Path.cwd().parent),
                                "storage_type": "remote_local",  # no SSH in auto-export path
                            },
                            config=cfg_yaml,
                        )
                    except Exception:
                        md_job_data = None

                job_data = md_job_data or ExecutionDB().get_job(single_id)
                if job_data is None:
                    return {
                        "success": False,
                        "error": f"Job {single_id} not found in ExecutionDB",
                    }

                job_result = exporter.export_job(job_data)
                return {
                    "success": job_result.success,
                    "invocation_id": job_data.invocation_id,
                    "jobs": {
                        job_data.job_id: {
                            "success": job_result.success,
                            "message": job_result.message,
                            "metadata": job_result.metadata or {},
                            "dest": getattr(job_result, "dest", None),
                        }
                    },
                    "metadata": job_result.metadata or {},
                }

            elif single_id.isdigit():  # pipeline_id
                db = ExecutionDB()
                for job_id, job_data in db._jobs.items():
                    if job_data.data.get("pipeline_id") == int(single_id):
                        job_result = exporter.export_job(job_data)
                        return {
                            "success": job_result.success,
                            "invocation_id": job_data.invocation_id,
                            "jobs": {
                                job_data.job_id: {
                                    "success": job_result.success,
                                    "message": job_result.message,
                                    "metadata": job_result.metadata or {},
                                }
                            },
                            "metadata": job_result.metadata or {},
                        }
                return {"success": False, "error": f"Pipeline {single_id} not found"}

            else:  # invocation_id
                result = exporter.export_invocation(single_id)
                if "jobs" in result:
                    for job_id, job_result in result["jobs"].items():
                        job_result.setdefault("metadata", {})
                return result
        else:
            # Multiple IDs - parse and group
            db = ExecutionDB()
            grouped_jobs: dict[
                str, dict[str, Any]
            ] = {}  # invocation_id -> {job_id: job_data}
            invocation_only = set()  # invocation_ids with no specific jobs
            all_jobs_for_consolidated = {}  # job_id -> job_data (for consolidated export)

            # Parse and group IDs
            for id_str in invocation_ids:
                if "." in id_str:  # job_id
                    job_data = db.get_job(id_str)
                    if job_data:
                        inv_id = job_data.invocation_id
                        if inv_id not in grouped_jobs:
                            grouped_jobs[inv_id] = {}
                        grouped_jobs[inv_id][id_str] = job_data
                        all_jobs_for_consolidated[id_str] = job_data
                elif id_str.isdigit():  # pipeline_id
                    # Find job by pipeline_id and add to group
                    for job_id, job_data in db._jobs.items():
                        if job_data.data.get("pipeline_id") == int(id_str):
                            inv_id = job_data.invocation_id
                            if inv_id not in grouped_jobs:
                                grouped_jobs[inv_id] = {}
                            grouped_jobs[inv_id][job_id] = job_data
                            all_jobs_for_consolidated[job_id] = job_data
                            break
                else:  # invocation_id
                    invocation_only.add(id_str)
                    # Add all jobs from this invocation for consolidated export
                    invocation_jobs = db.get_jobs(id_str)
                    all_jobs_for_consolidated.update(invocation_jobs)

            # Check if we should use consolidated export (local + json/csv format)
            should_consolidate = (
                dest == "local"
                and config
                and config.get("format") in ["json", "csv"]
                and (
                    len(invocation_only) > 1
                    or (len(invocation_only) == 1 and len(grouped_jobs) > 0)
                )
            )

            if should_consolidate and hasattr(exporter, "export_multiple_invocations"):
                # Use consolidated export for local exporter with JSON/CSV format
                all_invocation_ids = list(invocation_only)
                # Add invocations from grouped jobs
                all_invocation_ids.extend(
                    set(
                        job_data.invocation_id
                        for jobs in grouped_jobs.values()
                        for job_data in jobs.values()
                    )
                )
                all_invocation_ids = list(set(all_invocation_ids))  # remove duplicates

                try:
                    consolidated_result = exporter.export_multiple_invocations(
                        all_invocation_ids
                    )
                    return consolidated_result  # type: ignore[no-any-return]
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Consolidated export failed: {str(e)}",
                    }

            # Regular multi-invocation export
            all_results = {}
            overall_success = True

            # Export grouped jobs (partial invocations)
            for inv_id, jobs in grouped_jobs.items():
                try:
                    # Create a custom partial invocation export
                    results = {}
                    for job_id, job_data in jobs.items():
                        job_result = exporter.export_job(job_data)
                        results[job_id] = {
                            "success": job_result.success,
                            "message": job_result.message,
                            "metadata": job_result.metadata or {},
                        }
                        if not job_result.success:
                            overall_success = False

                    all_results[inv_id] = {
                        "success": all(r["success"] for r in results.values()),
                        "invocation_id": inv_id,
                        "jobs": results,
                        "partial": True,  # indicate this was partial invocation
                    }
                except Exception as e:
                    all_results[inv_id] = {
                        "success": False,
                        "error": f"Partial invocation export failed: {str(e)}",
                    }
                    overall_success = False

            # Export full invocations
            for inv_id in invocation_only:
                result = exporter.export_invocation(inv_id)
                # Ensure metadata is present in job results to prevent KeyError
                if "jobs" in result:
                    for job_id, job_result in result["jobs"].items():
                        if "metadata" not in job_result:
                            job_result["metadata"] = {}
                all_results[inv_id] = result
                if not result.get("success", False):
                    overall_success = False

            return {
                "success": overall_success,
                "invocations": all_results,
                "metadata": {
                    "total_invocations": len(all_results),
                    "successful_invocations": sum(
                        1 for r in all_results.values() if r.get("success")
                    ),
                    "mixed_export": len(grouped_jobs)
                    > 0,  # indicates mixed job/invocation export
                },
            }

    except Exception as e:
        return {"success": False, "error": f"Export failed: {str(e)}"}


def _check_api_endpoint_when_deployment_is_configured(cfg: RunConfig) -> None:
    """Check API endpoint configuration when deployment is configured.

    Args:
        cfg: Configuration object.

    Raises:
        ValueError: If invalid configuration is detected.
    """
    if cfg.deployment.type == "none":
        return
    if "target" not in cfg or not isinstance(cfg.target, DictConfig):
        return
    if "api_endpoint" not in cfg.target or not isinstance(
        cfg.target.api_endpoint, DictConfig
    ):
        return
    if "url" in cfg.target.api_endpoint:
        raise ValueError(
            "when deployment is configured, url field should not exist in target.api_endpoint"
        )
    if "model_id" in cfg.target.api_endpoint:
        raise ValueError(
            "when deployment is configured, model_id field should not exist in target.api_endpoint"
        )
