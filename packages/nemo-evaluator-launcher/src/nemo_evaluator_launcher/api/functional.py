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

import copy
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from nemo_evaluator.core.utils import validate_params_in_command
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from nemo_evaluator_launcher import __version__
from nemo_evaluator_launcher.api.types import RunConfig
from nemo_evaluator_launcher.common.config_models import (
    EvaluationModel,
    MountsModel,
    TaskModel,
)
from nemo_evaluator_launcher.common.execdb import ExecutionDB, JobData
from nemo_evaluator_launcher.common.helpers import (
    get_eval_factory_config,
    get_unique_task_name,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import (
    get_task_definition_for_job,
    load_tasks_mapping,
)
from nemo_evaluator_launcher.executors.registry import get_executor
from nemo_evaluator_launcher.exporters import create_exporter


def get_tasks_list() -> list[list[Any]]:
    """Get a list of available tasks from the mapping.

    Returns:
        list[list[Any]]: Each sublist contains task name, endpoint type, harness, container, arch, description, and type.
    """
    mapping = load_tasks_mapping()
    data = [
        [
            task_data.get("task"),
            task_data.get("endpoint_type"),
            task_data.get("harness"),
            task_data.get("container"),
            task_data.get("arch", ""),
            task_data.get("description", ""),
            task_data.get("type", ""),
        ]
        for task_data in mapping.values()
    ]
    return data


def _allowed_keys(model_cls: type) -> str:
    return ", ".join(sorted(model_cls.model_fields))


def _validate_config_sections(cfg: RunConfig) -> None:
    """Validate known config sections using Pydantic models (extra fields forbidden).

    Raises:
        ValueError: If an unknown or removed key is present in a validated section.
    """
    # Check for removed execution.env_vars — must use env_vars / deployment.env_vars / evaluation.env_vars
    if (
        cfg.execution is not None
        and getattr(cfg.execution, "env_vars", None) is not None
    ):
        raise ValueError(
            "'execution.env_vars' is not supported. "
            "Use top-level 'env_vars' for variables that should apply to all jobs. "
            "For variables that should be applied only to deployment or evaluation jobs, "
            "use 'deployment.env_vars' or 'evaluation.env_vars' respectively.\n"
            "Run 'nel migrate-config <config.yaml>' to migrate your config automatically."
        )

    if cfg.evaluation is not None:
        eval_dict = OmegaConf.to_container(cfg.evaluation, resolve=False)
        try:
            eval_model = EvaluationModel.model_validate(eval_dict)
        except ValidationError as e:
            raise ValueError(
                f"Invalid 'evaluation' config:\n{e}\n"
                f"Allowed top-level evaluation keys: {_allowed_keys(EvaluationModel)}\n"
                f"Allowed task keys: {_allowed_keys(TaskModel)}"
            ) from e
        _validate_nemo_evaluator_config_params(cfg, eval_model)

    if cfg.execution is not None and getattr(cfg.execution, "mounts", None) is not None:
        mounts_dict = OmegaConf.to_container(cfg.execution.mounts, resolve=False)
        try:
            MountsModel.model_validate(mounts_dict)
        except ValidationError as e:
            raise ValueError(
                f"Invalid 'execution.mounts' config:\n{e}\n"
                f"Allowed mounts keys: {_allowed_keys(MountsModel)}"
            ) from e


def _validate_nemo_evaluator_config_params(
    cfg: RunConfig, evaluation: EvaluationModel
) -> None:
    """Warn if nemo_evaluator_config params are not referenced in the task's command template.

    Uses packaged IRs or, when a task specifies a custom container, the mapping from that
    container. Needs cfg to merge task config via get_eval_factory_config.
    """
    if not evaluation.tasks:
        return

    default_mapping: dict | None = None
    try:
        default_mapping = load_tasks_mapping()
    except Exception as e:
        logger.warning(
            "Skipping nemo_evaluator_config param validation: mapping unavailable",
            error=str(e),
        )
        return

    for i, task in enumerate(evaluation.tasks):
        if not task.name:
            continue

        try:
            task_def = get_task_definition_for_job(
                task_query=task.name,
                base_mapping=default_mapping,
                container=task.container,
                endpoint_type=task.endpoint_type,
            )
        except (ValueError, KeyError) as e:
            logger.warning(
                "Task not found or could not resolve task definition, skipping param validation",
                task=task.name,
                error=str(e),
            )
            continue

        command = task_def.get("command", "")
        if not command:
            logger.warning(
                "No command template in task definition, skipping param validation",
                task=task.name,
            )
            continue

        raw_task = cfg.evaluation.tasks[i]
        try:
            merged_config = get_eval_factory_config(cfg, raw_task)
        except Exception as e:
            logger.warning(
                "Could not build merged nemo_evaluator_config for task",
                task=task.name,
                error=str(e),
            )
            continue

        try:
            validate_params_in_command(command, merged_config)
        except Exception as e:
            logger.warning(
                f"nemo_evaluator_config param validation failed for task '{task.name}'",
                error=str(e),
            )


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


def _find_matching_task_indices(
    tasks: list, task_name_filters: list[str]
) -> tuple[list[int], list[str]]:
    """Match each filter against the task list by plain name or unique name.

    A filter like ``"mmlu"`` matches every task with that plain name.
    A filter like ``"mmlu.0"`` matches a single task by its unique name
    (unique names are generated as ``"{name}.{index}"`` and task names
    cannot contain a dot-digit suffix, so there is no ambiguity).

    Returns ``(found_indices_sorted, unmatched_filters)``.
    """
    lookup: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tasks):
        lookup[t.name].append(i)
        lookup[get_unique_task_name(t.name, i)].append(i)

    found: set[int] = set()
    unmatched_filters: list[str] = []
    for f in task_name_filters:
        if f in lookup:
            found.update(lookup[f])
        else:
            unmatched_filters.append(f)

    return sorted(found), unmatched_filters


def filter_tasks(cfg: RunConfig, task_name_filters: list[str]) -> RunConfig:
    """Filter evaluation tasks to only include specified task names.

    Accepts both plain names (``"mmlu"`` — all instances) and positional
    unique names (``"mmlu.0"`` — one specific instance).  Task names
    cannot contain a dot-digit suffix, so there is no ambiguity between
    plain names and unique names.

    Args:
        cfg: The configuration object for the evaluation run.
        task_name_filters: List of task names or unique task names to include.

    Returns:
        RunConfig: A new configuration with filtered tasks (input is not mutated).

    Raises:
        ValueError: If any requested task is not found in config or no tasks defined.
    """
    if not task_name_filters:
        return cfg

    if not cfg.evaluation.tasks:
        raise ValueError("No tasks defined in config. Cannot filter tasks.")

    found, unmatched_filters = _find_matching_task_indices(
        cfg.evaluation.tasks, task_name_filters
    )

    if unmatched_filters:
        available = [
            get_unique_task_name(t.name, i) for i, t in enumerate(cfg.evaluation.tasks)
        ]
        raise ValueError(
            f"Requested task(s) not found in config: {sorted(unmatched_filters)}. "
            f"Available tasks: {available}"
        )

    result = copy.deepcopy(cfg)
    result.evaluation.tasks = [result.evaluation.tasks[i] for i in found]
    return result


def run_eval(
    cfg: RunConfig, dry_run: bool = False, tasks: Optional[list[str]] = None
) -> Optional[str]:
    """Run evaluation with specified config and overrides.

    Args:
        cfg: The configuration object for the evaluation run.
        dry_run: If True, do not run the evaluation, just prepare scripts and save them.
        tasks: Optional list of task names to run. If provided, only these tasks will be executed.

    Returns:
        Optional[str]: The invocation ID for the evaluation run.

    Raises:
        ValueError: If configuration validation fails or MISSING values are found.
        RuntimeError: If the executor fails to start the evaluation.
    """
    # Filter tasks if specified
    if tasks:
        cfg = filter_tasks(cfg, tasks)

    # Validate that no MISSING values exist in the configuration
    _validate_no_missing_values(cfg)
    _validate_config_sections(cfg)

    if dry_run:
        print(OmegaConf.to_yaml(cfg))

    _check_api_endpoint_when_deployment_is_configured(cfg)

    # Set up telemetry
    from nemo_evaluator.config import TelemetryLevel
    from nemo_evaluator.telemetry import (
        TELEMETRY_LEVEL_ENV_VAR,
        TELEMETRY_SESSION_ID_ENV_VAR,
        StatusEnum,
        TelemetryHandler,
        get_session_id,
        get_telemetry_level,
    )

    from nemo_evaluator_launcher.telemetry import LauncherJobEvent

    session_id = get_session_id()
    telemetry_handler = None
    telemetry_level = get_telemetry_level()

    # Extract telemetry metadata from config (evaluation.tasks must exist)
    task_names = []
    if hasattr(cfg, "evaluation") and cfg.evaluation.tasks:
        task_names = [t.name for t in cfg.evaluation.tasks if hasattr(t, "name")]

    exporter_names = []
    auto_export = (
        cfg.execution.get("auto_export") if hasattr(cfg, "execution") else None
    )
    if auto_export:
        exporter_names = list(auto_export.get("destinations", []) or [])

    model_name = "unknown"
    if hasattr(cfg, "target") and hasattr(cfg.target, "api_endpoint"):
        if hasattr(cfg.target.api_endpoint, "model_id"):
            model_name = cfg.target.api_endpoint.model_id or "unknown"
    # Also check deployment for model name
    if model_name == "unknown" and hasattr(cfg, "deployment"):
        if hasattr(cfg.deployment, "served_model_name"):
            model_name = cfg.deployment.served_model_name or "unknown"

    model_name_for_telemetry = (
        model_name if telemetry_level == TelemetryLevel.DEFAULT else "redacted"
    )

    executor_type = cfg.execution.type if hasattr(cfg.execution, "type") else "unknown"
    deployment_type = cfg.deployment.type if hasattr(cfg.deployment, "type") else "none"

    if not dry_run:
        # Propagate session ID and telemetry level to containers/child processes
        os.environ[TELEMETRY_SESSION_ID_ENV_VAR] = session_id
        os.environ[TELEMETRY_LEVEL_ENV_VAR] = str(telemetry_level.value)

        if telemetry_level != TelemetryLevel.OFF:
            telemetry_handler = TelemetryHandler(
                source_client_version=__version__,
                session_id=session_id,
                telemetry_level=telemetry_level,
            )
            telemetry_handler.start()
            telemetry_handler.enqueue(
                LauncherJobEvent(
                    executor_type=executor_type,
                    deployment_type=deployment_type,
                    model=model_name_for_telemetry,
                    tasks=task_names,
                    exporters=exporter_names,
                    status=StatusEnum.STARTED,
                )
            )

    status = StatusEnum.FAILURE
    try:
        result = get_executor(cfg.execution.type).execute_eval(cfg, dry_run)
        status = StatusEnum.SUCCESS
        return result
    finally:
        if telemetry_handler:
            telemetry_handler.enqueue(
                LauncherJobEvent(
                    executor_type=executor_type,
                    deployment_type=deployment_type,
                    model=model_name_for_telemetry,
                    tasks=task_names,
                    exporters=exporter_names,
                    status=status,
                )
            )
            telemetry_handler.stop()


def resume_eval(invocation_id: str) -> str:
    """Resume an evaluation by re-executing existing scripts.

    Automates the manual process of navigating to the execution directory
    and re-executing ``bash run.sh`` (local) or ``sbatch run.sub`` (SLURM).

    Args:
        invocation_id: The invocation ID to resume. Supports partial IDs.

    Returns:
        str: The resumed invocation ID.

    Raises:
        ValueError: If invocation not found.
        FileNotFoundError: If run scripts no longer exist on disk.
        RuntimeError: If script execution fails immediately.
    """

    db = ExecutionDB()
    jobs = db.get_jobs(invocation_id)

    if not jobs:
        raise ValueError(
            f"Invocation '{invocation_id}' not found in execution database. "
            f"Use 'nel ls runs' to see available invocations."
        )

    first_job = next(iter(jobs.values()))
    executor_cls = get_executor(first_job.executor)
    resolved_id = first_job.invocation_id

    logger.info(
        f"Resuming invocation {resolved_id}",
        num_jobs=len(jobs),
        executor=first_job.executor,
    )

    executor_cls.resume_invocation(resolved_id, jobs)

    logger.info(f"Resumed invocation {resolved_id} successfully")
    return resolved_id


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

    # TODO(agronskiy): refactor the `.`-checking job in all the functions.
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


def stream_logs(
    ids_or_prefixes: Union[str, list[str]],
) -> Iterator[Tuple[str, str, str]]:
    """Stream logs from jobs or invocations by their IDs or invocation IDs.

    Args:
        ids_or_prefixes: Single ID/prefix or list of job IDs or invocation IDs to stream logs from.
                         Short prefixes are allowed, we would try to match the full ones from
                         prefixes if no collisions are present.

    Yields:
        Tuple[str, str, str]: Tuples of (job_id, task_name, log_line) for each log line.
            Empty lines are yielded as empty strings.

    Raises:
        ValueError: If the executor doesn't support log streaming.
    """
    db = ExecutionDB()

    # Normalize to list for consistent processing
    if isinstance(ids_or_prefixes, str):
        ids_or_prefixes = [ids_or_prefixes]

    # Collect all jobs from all IDs, grouped by executor
    executor_to_jobs: Dict[str, Dict[str, JobData]] = {}
    executor_to_invocations: Dict[str, list[str]] = {}

    # TODO(agronskiy): refactor the `.`-checking job in all the functions.
    for id_or_prefix in ids_or_prefixes:
        # Determine if this is a job ID or invocation ID
        if "." in id_or_prefix:
            # This is a job ID
            job_data = db.get_job(id_or_prefix)
            if job_data is None:
                continue

            executor = job_data.executor
            if executor not in executor_to_jobs:
                executor_to_jobs[executor] = {}
            executor_to_jobs[executor][id_or_prefix] = job_data
        else:
            # This is an invocation ID
            jobs = db.get_jobs(id_or_prefix)
            if not jobs:
                continue

            # Get the executor class from the first job
            first_job_data = next(iter(jobs.values()))
            executor = first_job_data.executor
            if executor not in executor_to_invocations:
                executor_to_invocations[executor] = []
            executor_to_invocations[executor].append(id_or_prefix)

    # Stream logs from each executor simultaneously
    # For each executor, collect all job IDs and stream them together
    for executor, jobs_dict in executor_to_jobs.items():
        try:
            executor_cls = get_executor(executor)
        except ValueError:
            continue

        # For local executor with multiple jobs, pass list to stream simultaneously
        # For other executors or single jobs, pass individual job IDs
        if executor == "local" and len(jobs_dict) > 1:
            # Pass all job IDs as a list to stream simultaneously
            try:
                yield from executor_cls.stream_logs(
                    list(jobs_dict.keys()), executor_name=executor
                )
            except NotImplementedError:
                raise ValueError(
                    f"Log streaming is not yet implemented for executor '{executor}'"
                )
        else:
            # Single job or non-local executor
            for job_id in jobs_dict.keys():
                try:
                    yield from executor_cls.stream_logs(job_id, executor_name=executor)
                except NotImplementedError:
                    raise ValueError(
                        f"Log streaming is not yet implemented for executor '{executor}'"
                    )

    # Stream logs from invocation IDs
    for executor, invocation_ids in executor_to_invocations.items():
        try:
            executor_cls = get_executor(executor)
        except ValueError:
            continue

        # Stream each invocation (each invocation already handles multiple jobs internally)
        for invocation_id in invocation_ids:
            try:
                yield from executor_cls.stream_logs(
                    invocation_id, executor_name=executor
                )
            except NotImplementedError:
                raise ValueError(
                    f"Log streaming is not yet implemented for executor '{executor}'"
                )


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

    # TODO(agronskiy): refactor the `.`-checking job in all the functions.
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
    invocation_ids: Union[str, list[str]],
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

        exporter = create_exporter(dest, config)
        export_result = exporter.export(invocation_ids)

        return {
            "success": len(export_result.failed_jobs) == 0,
            "metadata": {
                "successful_jobs": len(export_result.successful_jobs),
                "failed_jobs": len(export_result.failed_jobs),
                "skipped_jobs": len(export_result.skipped_jobs),
            },
        }

    except Exception as e:
        return {"success": False, "metadata": {"error": f"Export failed: {str(e)}"}}


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
