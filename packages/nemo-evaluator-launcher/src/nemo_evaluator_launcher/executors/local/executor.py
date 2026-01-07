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
"""Local executor implementation for nemo-evaluator-launcher.

Handles running evaluation jobs locally using shell scripts and Docker containers.
"""

import copy
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import time
import warnings
from typing import Iterator, List, Optional, Tuple, Union

import jinja2
import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.execdb import (
    ExecutionDB,
    JobData,
    generate_invocation_id,
    generate_job_id,
)
from nemo_evaluator_launcher.common.helpers import (
    get_api_key_name,
    get_endpoint_url,
    get_eval_factory_command,
    get_eval_factory_dataset_size_from_run_config,
    get_health_url,
    get_timestamp_string,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import (
    get_task_definition_for_job,
    load_tasks_mapping,
)
from nemo_evaluator_launcher.common.printing_utils import bold, cyan, grey, red
from nemo_evaluator_launcher.executors.base import (
    BaseExecutor,
    ExecutionState,
    ExecutionStatus,
)
from nemo_evaluator_launcher.executors.registry import register_executor


@register_executor("local")
class LocalExecutor(BaseExecutor):
    @classmethod
    def execute_eval(cls, cfg: DictConfig, dry_run: bool = False) -> str:
        """Run evaluation jobs locally using the provided configuration.

        Args:
            cfg: The configuration object for the evaluation run.
            dry_run: If True, prepare scripts and save them without execution.

        Returns:
            str: The invocation ID for the evaluation run.

        Raises:
            RuntimeError: If the run script fails.
        """
        # Check if docker is available (skip in dry_run mode)
        if not dry_run and shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is not installed or not in PATH. "
                "Please install Docker to run local evaluations."
            )

        # Generate invocation ID for this evaluation run
        invocation_id = generate_invocation_id()

        output_dir = pathlib.Path(cfg.execution.output_dir).absolute() / (
            get_timestamp_string(include_microseconds=False) + "-" + invocation_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks_mapping = load_tasks_mapping()
        evaluation_tasks = []
        job_ids = []

        run_template = jinja2.Template(
            open(pathlib.Path(__file__).parent / "run.template.sh", "r").read()
        )

        execution_mode = cfg.execution.get("mode", "parallel")
        if execution_mode == "parallel":
            if cfg.deployment.type != "none":
                raise ValueError(
                    f"Execution mode 'parallel' is not supported with deployment type: {cfg.deployment.type}. Use 'sequential' instead."
                )
            is_execution_mode_sequential = False
        elif execution_mode == "sequential":
            is_execution_mode_sequential = True
        else:
            raise ValueError(
                "unknown execution mode: {}. Choose one of {}".format(
                    repr(execution_mode), ["parallel", "sequential"]
                )
            )

        # Will accumulate if any task contains unsafe commands.
        is_potentially_unsafe = False

        deployment = None

        for idx, task in enumerate(cfg.evaluation.tasks):
            timestamp = get_timestamp_string()
            task_definition = get_task_definition_for_job(
                task_query=task.name,
                base_mapping=tasks_mapping,
                container=task.get("container"),
            )

            if cfg.deployment.type != "none":
                # container name
                server_container_name = f"server-{task.name}-{timestamp}"

                # health_url
                health_url = get_health_url(
                    cfg, get_endpoint_url(cfg, task, task_definition["endpoint_type"])
                )

                # mounts
                deployment_mounts_list = []
                if checkpoint_path := cfg.deployment.get("checkpoint_path"):
                    deployment_mounts_list.append(f"{checkpoint_path}:/checkpoint:ro")
                if cache_path := cfg.deployment.get("cache_path"):
                    deployment_mounts_list.append(f"{cache_path}:/cache")
                for source_mnt, target_mnt in (
                    cfg.execution.get("mounts", {}).get("deployment", {}).items()
                ):
                    deployment_mounts_list.append(f"{source_mnt}:{target_mnt}")

                # env vars
                deployment_env_vars = cfg.execution.get("env_vars", {}).get(
                    "deployment", {}
                )

                if cfg.deployment.get("env_vars"):
                    warnings.warn(
                        "cfg.deployment.env_vars will be deprecated in future versions. "
                        "Use cfg.execution.env_vars.deployment instead.",
                        category=DeprecationWarning,
                        stacklevel=2,
                    )
                    deployment_env_vars.update(cfg.deployment["env_vars"])

                command = cfg.deployment.command
                deployment_extra_docker_args = cfg.execution.get(
                    "extra_docker_args", ""
                )

                deployment = {
                    "container_name": server_container_name,
                    "image": cfg.deployment.image,
                    "command": command,
                    "mounts": deployment_mounts_list,
                    "env_vars": [f"{k}={v}" for k, v in deployment_env_vars.items()],
                    "health_url": health_url,
                    "port": cfg.deployment.port,
                    "extra_docker_args": deployment_extra_docker_args,
                }

            # Create job ID as <invocation_id>.<n>
            job_id = generate_job_id(invocation_id, idx)
            job_ids.append(job_id)
            client_container_name = f"client-{task.name}-{timestamp}"

            # collect all env vars
            env_vars = copy.deepcopy(dict(cfg.evaluation.get("env_vars", {})))
            env_vars.update(task.get("env_vars", {}))
            if api_key_name := get_api_key_name(cfg):
                assert "API_KEY" not in env_vars
                env_vars["API_KEY"] = api_key_name

            # check if the environment variables are set
            for env_var in env_vars.values():
                if os.getenv(env_var) is None:
                    raise ValueError(
                        f"Trying to pass an unset environment variable {env_var}."
                    )

            # check if required env vars are defined (excluding NEMO_EVALUATOR_DATASET_DIR which is handled separately):
            for required_env_var in task_definition.get("required_env_vars", []):
                # Skip NEMO_EVALUATOR_DATASET_DIR as it's handled by dataset mounting logic below
                if required_env_var == "NEMO_EVALUATOR_DATASET_DIR":
                    continue
                if required_env_var not in env_vars.keys():
                    raise ValueError(
                        f"{task.name} task requires environment variable {required_env_var}."
                        " Specify it in the task subconfig in the 'env_vars' dict as the following"
                        f" pair {required_env_var}: YOUR_ENV_VAR_NAME"
                    )

            # Handle dataset directory mounting if NEMO_EVALUATOR_DATASET_DIR is required
            dataset_mount_host = None
            dataset_mount_container = None
            dataset_env_var_value = None
            if "NEMO_EVALUATOR_DATASET_DIR" in task_definition.get(
                "required_env_vars", []
            ):
                # Get dataset directory from task config
                if "dataset_dir" in task:
                    dataset_mount_host = task["dataset_dir"]
                else:
                    raise ValueError(
                        f"{task.name} task requires a dataset_dir to be specified. "
                        f"Add 'dataset_dir: /path/to/your/dataset' under the task configuration."
                    )
                # Get container mount path (default to /datasets if not specified)
                dataset_mount_container = task.get("dataset_mount_path", "/datasets")
                # Set NEMO_EVALUATOR_DATASET_DIR to the container mount path
                dataset_env_var_value = dataset_mount_container

            # format env_vars for a template
            env_vars_list = [
                f"{env_var_dst}=${env_var_src}"
                for env_var_dst, env_var_src in env_vars.items()
            ]

            # Add dataset env var if needed (directly with value, not from host env)
            if dataset_env_var_value:
                env_vars_list.append(
                    f"NEMO_EVALUATOR_DATASET_DIR={dataset_env_var_value}"
                )

            eval_image = task_definition["container"]
            if "container" in task:
                eval_image = task["container"]

            task_output_dir = output_dir / task.name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            eval_factory_command_struct = get_eval_factory_command(
                cfg, task, task_definition
            )
            eval_factory_command = eval_factory_command_struct.cmd
            # The debug comment for placing into the script and easy debug. Reason
            # (see `CmdAndReadableComment`) is the current way of passing the command
            # is base64-encoded config `echo`-ed into file.
            # TODO(agronskiy): cleaner way is to encode everything with base64, not
            # some parts (like ef_config.yaml) and just output as logs somewhere.
            eval_factory_command_debug_comment = eval_factory_command_struct.debug
            is_potentially_unsafe = (
                is_potentially_unsafe
                or eval_factory_command_struct.is_potentially_unsafe
            )
            evaluation_task = {
                "deployment": deployment,
                "name": task.name,
                "job_id": job_id,
                "eval_image": eval_image,
                "client_container_name": client_container_name,
                "env_vars": env_vars_list,
                "output_dir": task_output_dir,
                "eval_factory_command": eval_factory_command,
                "eval_factory_command_debug_comment": eval_factory_command_debug_comment,
                "dataset_mount_host": dataset_mount_host,
                "dataset_mount_container": dataset_mount_container,
            }
            evaluation_tasks.append(evaluation_task)

            # Check if auto-export is enabled by presence of destination(s)
            auto_export_config = cfg.execution.get("auto_export", {})
            auto_export_destinations = auto_export_config.get("destinations", [])

            extra_docker_args = cfg.execution.get("extra_docker_args", "")

            run_sh_content = (
                run_template.render(
                    evaluation_tasks=[evaluation_task],
                    auto_export_destinations=auto_export_destinations,
                    extra_docker_args=extra_docker_args,
                ).rstrip("\n")
                + "\n"
            )

            (task_output_dir / "run.sh").write_text(run_sh_content)

        run_all_sequentially_sh_content = (
            run_template.render(
                evaluation_tasks=evaluation_tasks,
                auto_export_destinations=auto_export_destinations,
                extra_docker_args=extra_docker_args,
            ).rstrip("\n")
            + "\n"
        )
        (output_dir / "run_all.sequential.sh").write_text(
            run_all_sequentially_sh_content
        )

        if dry_run:
            print(bold("\n\n=============================================\n\n"))
            print(bold(cyan(f"DRY RUN: Scripts prepared and saved to {output_dir}")))
            if is_execution_mode_sequential:
                print(
                    cyan(
                        "\n\n=========== Main script | run_all.sequential.sh =====================\n\n"
                    )
                )

                with open(output_dir / "run_all.sequential.sh", "r") as f:
                    print(grey(f.read()))
            else:
                for idx, task in enumerate(cfg.evaluation.tasks):
                    task_output_dir = output_dir / task.name
                    print(
                        cyan(
                            f"\n\n=========== Task script | {task.name}/run.sh =====================\n\n"
                        )
                    )
                    with open(task_output_dir / "run.sh", "r") as f:
                        print(grey(f.read()))
            print(bold("\nTo execute, run without --dry-run"))

            if is_potentially_unsafe:
                print(
                    red(
                        "\nFound `pre_cmd` which carries security risk. When running without --dry-run "
                        "make sure you trust the command and set NEMO_EVALUATOR_TRUST_PRE_CMD=1"
                    )
                )
            return invocation_id

        if is_potentially_unsafe:
            if os.environ.get("NEMO_EVALUATOR_TRUST_PRE_CMD", "") == "1":
                logger.warning(
                    "Found non-empty task commands (e.g. `pre_cmd`) and NEMO_EVALUATOR_TRUST_PRE_CMD "
                    "is set, proceeding with caution."
                )

            else:
                logger.error(
                    "Found non-empty task commands (e.g. `pre_cmd`) and NEMO_EVALUATOR_TRUST_PRE_CMD "
                    "is not set. This might carry security risk and unstable environments. "
                    "To continue, make sure you trust the command and set NEMO_EVALUATOR_TRUST_PRE_CMD=1.",
                )
                raise AttributeError(
                    "Untrusted command found in config, make sure you trust and "
                    "set NEMO_EVALUATOR_TRUST_PRE_CMD=1."
                )

        # Save launched jobs metadata
        db = ExecutionDB()
        for job_id, task, evaluation_task in zip(
            job_ids, cfg.evaluation.tasks, evaluation_tasks
        ):
            db.write_job(
                job=JobData(
                    invocation_id=invocation_id,
                    job_id=job_id,
                    timestamp=time.time(),
                    executor="local",
                    data={
                        "output_dir": str(evaluation_task["output_dir"]),
                        "container": evaluation_task["client_container_name"],
                        "eval_image": evaluation_task["eval_image"],
                    },
                    config=OmegaConf.to_object(cfg),
                )
            )

        # Launch bash scripts with Popen for non-blocking execution.
        # To ensure subprocess continues after python exits:
        # - on Unix-like systems, to fully detach the subprocess
        #   so it does not die when Python exits, pass start_new_session=True;
        # - on Windows use creationflags=subprocess.CREATE_NEW_PROCESS_GROUP flag.
        os_name = platform.system()
        processes = []

        if is_execution_mode_sequential:
            if os_name == "Windows":
                proc = subprocess.Popen(
                    shlex.split("bash run_all.sequential.sh"),
                    cwd=output_dir,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    shlex.split("bash run_all.sequential.sh"),
                    cwd=output_dir,
                    start_new_session=True,
                )
            processes.append(("run_all.sequential.sh", proc, output_dir))
        else:
            for task in cfg.evaluation.tasks:
                if os_name == "Windows":
                    proc = subprocess.Popen(
                        shlex.split("bash run.sh"),
                        cwd=output_dir / task.name,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    proc = subprocess.Popen(
                        shlex.split("bash run.sh"),
                        cwd=output_dir / task.name,
                        start_new_session=True,
                    )
                processes.append((task.name, proc, output_dir / task.name))

        # Wait briefly and check if bash scripts exited immediately (which means error)
        time.sleep(0.3)

        for name, proc, work_dir in processes:
            exit_code = proc.poll()
            if exit_code is not None and exit_code != 0:
                error_msg = f"Script for {name} exited with code {exit_code}"
                raise RuntimeError(f"Job startup failed | {error_msg}")

        print(bold(cyan("\nCommands for real-time monitoring:")))
        for job_id, evaluation_task in zip(job_ids, evaluation_tasks):
            print(f"\n  Job {job_id} ({evaluation_task['name']}):")
            print(f"    nemo-evaluator-launcher logs {job_id}")

        print(bold(cyan("\nFollow all logs for this invocation:")))
        print(f"  nemo-evaluator-launcher logs {invocation_id}")

        return invocation_id

    @staticmethod
    def get_status(id: str) -> List[ExecutionStatus]:
        """Get the status of a specific job or all jobs in an invocation group.

        Args:
            id: Unique job identifier or invocation identifier.

        Returns:
            List containing the execution status for the job(s).
        """
        db = ExecutionDB()

        # If id looks like an invocation_id (no dot), get all jobs for it
        if "." not in id:
            jobs = db.get_jobs(id)
            statuses: List[ExecutionStatus] = []
            for job_id, _ in jobs.items():
                statuses.extend(LocalExecutor.get_status(job_id))
            return statuses

        # Otherwise, treat as job_id
        job_data = db.get_job(id)
        if job_data is None:
            return []
        if job_data.executor != "local":
            return []

        output_dir = pathlib.Path(job_data.data.get("output_dir", ""))
        if not output_dir.exists():
            return [ExecutionStatus(id=id, state=ExecutionState.PENDING)]

        artifacts_dir = output_dir / "artifacts"
        progress = _get_progress(artifacts_dir)

        logs_dir = output_dir / "logs"
        if not logs_dir.exists():
            return [
                ExecutionStatus(
                    id=id,
                    state=ExecutionState.PENDING,
                    progress=dict(progress=progress),
                )
            ]

        # Check if job was killed
        if job_data.data.get("killed", False):
            return [
                ExecutionStatus(
                    id=id, state=ExecutionState.KILLED, progress=dict(progress=progress)
                )
            ]

        stage_files = {
            "pre_start": logs_dir / "stage.pre-start",
            "running": logs_dir / "stage.running",
            "exit": logs_dir / "stage.exit",
        }

        if stage_files["exit"].exists():
            try:
                content = stage_files["exit"].read_text().strip()
                if " " in content:
                    timestamp, exit_code_str = content.rsplit(" ", 1)
                    exit_code = int(exit_code_str)
                    if exit_code == 0:
                        return [
                            ExecutionStatus(
                                id=id,
                                state=ExecutionState.SUCCESS,
                                progress=dict(progress=progress),
                            )
                        ]
                    else:
                        return [
                            ExecutionStatus(
                                id=id,
                                state=ExecutionState.FAILED,
                                progress=dict(progress=progress),
                            )
                        ]
                else:
                    return [
                        ExecutionStatus(
                            id=id,
                            state=ExecutionState.FAILED,
                            progress=dict(progress=progress),
                        )
                    ]
            except (ValueError, OSError):
                return [
                    ExecutionStatus(
                        id=id,
                        state=ExecutionState.FAILED,
                        progress=dict(progress=progress),
                    )
                ]
        elif stage_files["running"].exists():
            return [
                ExecutionStatus(
                    id=id,
                    state=ExecutionState.RUNNING,
                    progress=dict(progress=progress),
                )
            ]
        elif stage_files["pre_start"].exists():
            return [
                ExecutionStatus(
                    id=id,
                    state=ExecutionState.PENDING,
                    progress=dict(progress=progress),
                )
            ]

        return [
            ExecutionStatus(
                id=id, state=ExecutionState.PENDING, progress=dict(progress=progress)
            )
        ]

    @staticmethod
    def kill_job(job_id: str) -> None:
        """Kill a local job.

        Args:
            job_id: The job ID (e.g., abc123.0) to kill.

        Raises:
            ValueError: If job is not found or invalid.
            RuntimeError: If Docker container cannot be stopped.
        """
        db = ExecutionDB()
        job_data = db.get_job(job_id)

        if job_data is None:
            raise ValueError(f"Job {job_id} not found")

        if job_data.executor != "local":
            raise ValueError(
                f"Job {job_id} is not a local job (executor: {job_data.executor})"
            )

        # Get container name from database
        container_name = job_data.data.get("container")
        if not container_name:
            raise ValueError(f"No container name found for job {job_id}")

        killed_something = False

        # First, try to stop the Docker container if it's running
        result = subprocess.run(
            shlex.split(f"docker stop {container_name}"),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            killed_something = True
        # Don't raise error if container doesn't exist (might be still pulling)

        # Find and kill Docker processes for this container
        result = subprocess.run(
            shlex.split(f"pkill -f 'docker run.*{container_name}'"),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            killed_something = True

        # If we successfully killed something, mark as killed
        if killed_something:
            job_data.data["killed"] = True
            db.write_job(job_data)
            LocalExecutor._add_to_killed_jobs(job_data.invocation_id, job_id)
            return

        # If nothing was killed, check if this is a pending job
        status_list = LocalExecutor.get_status(job_id)
        if status_list and status_list[0].state == ExecutionState.PENDING:
            # For pending jobs, mark as killed even though there's nothing to kill yet
            job_data.data["killed"] = True
            db.write_job(job_data)
            LocalExecutor._add_to_killed_jobs(job_data.invocation_id, job_id)
            return

        # Use common helper to get informative error message based on job status
        current_status = status_list[0].state if status_list else None
        error_msg = LocalExecutor.get_kill_failure_message(
            job_id, f"container: {container_name}", current_status
        )
        raise RuntimeError(error_msg)

    @staticmethod
    def stream_logs(
        id: Union[str, List[str]], executor_name: Optional[str] = None
    ) -> Iterator[Tuple[str, str, str]]:
        """Stream logs from a job or invocation group.

        Args:
            id: Unique job identifier, invocation identifier, or list of job IDs to stream simultaneously.

        Yields:
            Tuple[str, str, str]: Tuples of (job_id, task_name, log_line) for each log line.
                Empty lines are yielded as empty strings.
        """
        db = ExecutionDB()

        # Handle list of job IDs for simultaneous streaming
        if isinstance(id, list):
            # Collect all jobs from the list of job IDs
            jobs = {}
            for job_id in id:
                job_data = db.get_job(job_id)
                if job_data is None or job_data.executor != "local":
                    continue
                jobs[job_id] = job_data
            if not jobs:
                return
        # If id looks like an invocation_id (no dot), get all jobs for it
        elif "." not in id:
            jobs = db.get_jobs(id)
            if not jobs:
                return
        else:
            # Otherwise, treat as job_id
            job_data = db.get_job(id)
            if job_data is None or job_data.executor != "local":
                return
            jobs = {id: job_data}

        # Collect log file paths and metadata
        log_files = []

        for job_id, job_data in jobs.items():
            output_dir = pathlib.Path(job_data.data.get("output_dir", ""))
            if not output_dir:
                continue

            # Get task name from config
            task_name = LocalExecutor._extract_task_name(job_data, job_id)

            log_file_path = output_dir / "logs" / "client_stdout.log"

            log_files.append(
                {
                    "job_id": job_id,
                    "task_name": task_name,
                    "path": log_file_path,
                    "file_handle": None,
                    "position": 0,
                }
            )

        if not log_files:
            return

        # Track which files we've seen before (for tail behavior)
        file_seen_before = {}

        # Open files that exist, keep track of which ones we're waiting for
        # First, yield the last 15 lines from existing files
        for log_info in log_files:
            if log_info["path"].exists():
                file_seen_before[log_info["path"]] = True
                # Read and yield last 15 lines
                last_lines = LocalExecutor._read_last_n_lines(log_info["path"], 15)
                for line in last_lines:
                    yield (
                        log_info["job_id"],
                        log_info["task_name"],
                        line,
                    )
                try:
                    log_info["file_handle"] = open(
                        log_info["path"], "r", encoding="utf-8", errors="replace"
                    )
                    # Seek to end if file already exists (tail behavior)
                    log_info["file_handle"].seek(0, 2)
                    log_info["position"] = log_info["file_handle"].tell()
                except Exception as e:
                    logger.error(f"Could not open {log_info['path']}: {e}")
            else:
                file_seen_before[log_info["path"]] = False

        try:
            while True:
                any_activity = False

                for log_info in log_files:
                    # Try to open file if it doesn't exist yet
                    if log_info["file_handle"] is None:
                        if log_info["path"].exists():
                            try:
                                # If file was just created, read last 15 lines first
                                if not file_seen_before.get(log_info["path"], False):
                                    last_lines = LocalExecutor._read_last_n_lines(
                                        log_info["path"], 15
                                    )
                                    for line in last_lines:
                                        yield (
                                            log_info["job_id"],
                                            log_info["task_name"],
                                            line,
                                        )
                                    file_seen_before[log_info["path"]] = True

                                log_info["file_handle"] = open(
                                    log_info["path"],
                                    "r",
                                    encoding="utf-8",
                                    errors="replace",
                                )
                                # Seek to end for tail behavior
                                log_info["file_handle"].seek(0, 2)
                                log_info["position"] = log_info["file_handle"].tell()
                            except Exception as e:
                                logger.error(f"Could not open {log_info['path']}: {e}")
                                continue

                    # Read new lines from file
                    if log_info["file_handle"] is not None:
                        try:
                            # Check if file has grown
                            current_size = log_info["path"].stat().st_size
                            if current_size > log_info["position"]:
                                log_info["file_handle"].seek(log_info["position"])
                                new_lines = log_info["file_handle"].readlines()
                                log_info["position"] = log_info["file_handle"].tell()

                                # Yield new lines
                                for line in new_lines:
                                    line_stripped = line.rstrip("\n\r")
                                    yield (
                                        log_info["job_id"],
                                        log_info["task_name"],
                                        line_stripped,
                                    )
                                    any_activity = True
                        except (OSError, IOError) as e:
                            # File might have been deleted or moved
                            # Don't log error for every check, only on first error
                            if log_info.get("error_printed", False) is False:
                                logger.error(f"Error reading {log_info['path']}: {e}")
                                log_info["error_printed"] = True
                            log_info["file_handle"] = None
                        except Exception:
                            # Reset error flag if we successfully read again
                            log_info["error_printed"] = False

                # If no activity, sleep briefly to avoid busy waiting
                if not any_activity:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            # Clean exit on Ctrl+C
            pass
        finally:
            # Close all file handles
            for log_info in log_files:
                if log_info["file_handle"] is not None:
                    try:
                        log_info["file_handle"].close()
                    except Exception:
                        pass

    @staticmethod
    def _read_last_n_lines(file_path: pathlib.Path, n: int) -> List[str]:
        """Read the last N lines from a file efficiently.

        Args:
            file_path: Path to the file to read from.
            n: Number of lines to read from the end.

        Returns:
            List of the last N lines (or fewer if file has fewer lines).
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                # Read all lines
                all_lines = f.readlines()
                # Return last n lines, stripping newlines
                return [line.rstrip("\n\r") for line in all_lines[-n:]]
        except Exception as e:
            logger.warning(f"Could not read last {n} lines from {file_path}: {e}")
            return []

    @staticmethod
    def _extract_task_name(job_data: JobData, job_id: str) -> str:
        """Extract task name from job data config.

        Args:
            job_data: JobData object containing config.
            job_id: Job ID for error reporting.

        Returns:
            Task name string.
        """
        config = job_data.config or {}
        evaluation = config.get("evaluation", {})
        tasks = evaluation.get("tasks", [])

        # Find the task that matches this job
        # For job_id like "15b9f667.0", index is 0
        try:
            if "." in job_id:
                index = int(job_id.split(".")[1])
                if len(tasks) > 0 and index >= len(tasks):
                    raise AttributeError(
                        f"Job task index {job_id} is larger than number of tasks {len(tasks)} in invocation"
                    )
                # If index is valid and tasks exist, return the task name
                if len(tasks) > 0 and index < len(tasks):
                    return tasks[index].get("name", "unknown")
        except (ValueError, IndexError):
            pass

        # Fallback: try to get task name from output_dir
        # output_dir typically ends with task name
        output_dir = job_data.data.get("output_dir", "")
        if output_dir:
            parts = pathlib.Path(output_dir).parts
            if parts:
                return parts[-1]

        return "unknown"

    @staticmethod
    def _add_to_killed_jobs(invocation_id: str, job_id: str) -> None:
        """Add a job ID to the killed jobs file for this invocation.

        Args:
            invocation_id: The invocation ID.
            job_id: The job ID to mark as killed.
        """
        db = ExecutionDB()
        jobs = db.get_jobs(invocation_id)
        if not jobs:
            return

        # Get invocation output directory from any job's output_dir
        first_job_data = next(iter(jobs.values()))
        job_output_dir = pathlib.Path(first_job_data.data.get("output_dir", ""))
        if not job_output_dir.exists():
            return

        # Invocation dir is parent of job output dir
        invocation_dir = job_output_dir.parent
        killed_jobs_file = invocation_dir / "killed_jobs.txt"

        # Append job_id to file
        with open(killed_jobs_file, "a") as f:
            f.write(f"{job_id}\n")


def _get_progress(artifacts_dir: pathlib.Path) -> Optional[float]:
    """Get the progress of a local job.

    Args:
        artifacts_dir: The directory containing the evaluation artifacts.

    Returns:
        The progress of the job as a float between 0 and 1.
    """
    progress_filepath = artifacts_dir / "progress"
    if not progress_filepath.exists():
        return None
    progress_str = progress_filepath.read_text().strip()
    try:
        processed_samples = int(progress_str)
    except ValueError:
        return None

    dataset_size = _get_dataset_size(artifacts_dir)
    if dataset_size is not None:
        progress = processed_samples / dataset_size
    else:
        # NOTE(dfridman): if we don't know the dataset size, report the number of processed samples
        progress = processed_samples
    return progress


def _get_dataset_size(artifacts_dir: pathlib.Path) -> Optional[int]:
    """Get the dataset size for a benchmark.

    Args:
        artifacts_dir: The directory containing the evaluation artifacts.

    Returns:
        The dataset size for the benchmark.
    """
    run_config = artifacts_dir / "run_config.yml"
    if not run_config.exists():
        return None
    run_config = yaml.safe_load(run_config.read_text())
    return get_eval_factory_dataset_size_from_run_config(run_config)
