"""Local executor implementation for nv-eval-platform.

Handles running evaluation jobs locally using shell scripts and Docker containers.
"""

import copy
import os
import pathlib
import shlex
import subprocess
import time
from typing import List, Optional

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
    get_eval_factory_command,
    get_eval_factory_dataset_size_from_run_config,
    get_timestamp_string,
)
from nemo_evaluator_launcher.common.mapping import (
    get_task_from_mapping,
    load_tasks_mapping,
)
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
            NotImplementedError: If deployment is not 'none'.
            RuntimeError: If the run script fails.
        """
        if cfg.deployment.type != "none":
            raise NotImplementedError(
                f"type {cfg.deployment.type} is not implemented -- add deployment support"
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

        eval_template = jinja2.Template(
            open(pathlib.Path(__file__).parent / "run.template.sh", "r").read()
        )

        for idx, task in enumerate(cfg.evaluation.tasks):
            task_definition = get_task_from_mapping(task.name, tasks_mapping)

            # Create job ID as <invocation_id>.<n>
            job_id = generate_job_id(invocation_id, idx)
            job_ids.append(job_id)
            container_name = f"{task.name}-{get_timestamp_string()}"

            # collect all env vars
            env_vars = copy.deepcopy(dict(cfg.evaluation.get("env_vars", {})))
            env_vars.update(task.get("env_vars", {}))
            if cfg.target.api_endpoint.api_key_name:
                assert "API_KEY" not in env_vars
                env_vars["API_KEY"] = cfg.target.api_endpoint.api_key_name

            # check if the environment variables are set
            for env_var in env_vars.values():
                if os.getenv(env_var) is None:
                    raise ValueError(
                        f"Trying to pass an unset environment variable {env_var}."
                    )

            # check if required env vars are defined:
            for required_env_var in task_definition.get("required_env_vars", []):
                if required_env_var not in env_vars.keys():
                    raise ValueError(
                        f"{task.name} task requires environment variable {required_env_var}."
                        " Specify it in the task subconfig in the 'env_vars' dict as the following"
                        f" key: value pair {required_env_var}: YOUR_ENV_VAR_NAME"
                    )

            # format env_vars for a template
            env_vars = [
                f"{env_var_dst}=${env_var_src}"
                for env_var_dst, env_var_src in env_vars.items()
            ]

            eval_image = task_definition["container"].replace(":5005", "")
            if "container" in task:
                eval_image = task["container"]

            task_output_dir = output_dir / task.name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            evaluation_task = {
                "job_id": job_id,
                "eval_image": eval_image,
                "container_name": container_name,
                "env_vars": env_vars,
                "output_dir": task_output_dir,
                "eval_factory_command": get_eval_factory_command(
                    cfg, task, task_definition
                ),
            }
            evaluation_tasks.append(evaluation_task)
            # NOTE(dfridman): create a run.sh file for each task for quick relaunching of the failed tasks

            # Check if auto-export is enabled by presence of destination(s)
            auto_export_config = cfg.execution.get("auto_export", {})
            auto_export_destinations = auto_export_config.get("destinations", [])

            # Template context with auto-export variables
            template_context = {
                "evaluation_tasks": evaluation_tasks,
                "auto_export_destinations": auto_export_destinations,
                "invocation_id": invocation_id,
            }

            (task_output_dir / "run.sh").write_text(
                eval_template.render(
                    evaluation_tasks=[evaluation_task],
                    auto_export_destinations=auto_export_destinations,
                    invocation_id=invocation_id,
                    job_id=job_id,
                ).rstrip("\n")
                + "\n"
            )

        # Template context with auto-export variables
        template_context = {
            "evaluation_tasks": evaluation_tasks,
            "auto_export_destinations": auto_export_destinations,
            "invocation_id": invocation_id,
        }

        (output_dir / "run_all.sh").write_text(
            eval_template.render(**template_context).rstrip("\n") + "\n"
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
                        "container": evaluation_task["container_name"],
                        "eval_image": evaluation_task["eval_image"],
                    },
                    config=OmegaConf.to_object(cfg),
                )
            )

        if dry_run:
            print("\n\n=============================================\n\n")
            print(f"DRY RUN: Scripts prepared and saved to {output_dir}")
            print(f"   - Main script: {output_dir}/run_all.sh")
            print(
                "\n\n =========== Main script (run_all.sh) ===================== \n\n"
            )
            with open(output_dir / "run_all.sh", "r") as f:
                print(f.read())
            for idx, task in enumerate(cfg.evaluation.tasks):
                task_output_dir = output_dir / task.name
                print(
                    f"\n\n =========== Task script ({task.name} run.sh) ===================== \n\n"
                )
                with open(task_output_dir / "run.sh", "r") as f:
                    print(f.read())
            print("\nTo execute, run without --dry-run")
            return invocation_id

        completed_process = subprocess.run(
            args=shlex.split("bash run_all.sh"), cwd=output_dir
        )
        if completed_process.returncode != 0:
            raise RuntimeError("failed to execute run_all.sh successfully")

        print("\nCommands for real-time monitoring:")
        for job_id, evaluation_task in zip(job_ids, evaluation_tasks):
            log_file = evaluation_task["output_dir"] / "logs" / "stdout.log"
            print(f"  tail -f {log_file}")

        print("\nFollow all logs for this invocation:")
        print(f"  tail -f {output_dir}/*/logs/stdout.log")

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

        # If id looks like an invocation_id (8 hex digits, no dot), get all jobs for it
        if len(id) == 8 and "." not in id:
            jobs = db.get_jobs(id)
            statuses: List[ExecutionStatus] = []
            for job_id, job_data in jobs.items():
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
            return [ExecutionStatus(id=id, state=ExecutionState.FAILED)]

        artifacts_dir = output_dir / "artifacts"
        progress = _get_progress(artifacts_dir)

        logs_dir = output_dir / "logs"
        if not logs_dir.exists():
            return [
                ExecutionStatus(id=id, state=ExecutionState.FAILED, progress=progress)
            ]

        # Check if job was killed
        if job_data.data.get("killed", False):
            return [
                ExecutionStatus(id=id, state=ExecutionState.KILLED, progress=progress)
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
                                id=id, state=ExecutionState.SUCCESS, progress=progress
                            )
                        ]
                    else:
                        return [
                            ExecutionStatus(
                                id=id, state=ExecutionState.FAILED, progress=progress
                            )
                        ]
            except (ValueError, OSError):
                return [
                    ExecutionStatus(
                        id=id, state=ExecutionState.FAILED, progress=progress
                    )
                ]
        elif stage_files["running"].exists():
            return [
                ExecutionStatus(id=id, state=ExecutionState.RUNNING, progress=progress)
            ]
        elif stage_files["pre_start"].exists():
            return [
                ExecutionStatus(id=id, state=ExecutionState.FAILED, progress=progress)
            ]

        return [ExecutionStatus(id=id, state=ExecutionState.FAILED, progress=progress)]

    @staticmethod
    def kill_job(job_id: str) -> None:
        """Kill a local job by stopping its Docker container and related processes.

        Args:
            job_id: The job ID to kill.

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

        # Mark job as killed in database if we killed something
        if killed_something:
            job_data.data["killed"] = True
            db.write_job(job_data)
        else:
            raise RuntimeError(
                f"Could not find or kill job {job_id} (container: {container_name})"
            )


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
