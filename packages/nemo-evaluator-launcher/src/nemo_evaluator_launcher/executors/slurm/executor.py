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
"""SLURM executor implementation for nemo-evaluator-launcher.

Handles submitting evaluation jobs to a SLURM cluster via SSH and sbatch scripts.
"""

import copy
import os
import re
import shlex
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig, OmegaConf

from nemo_evaluator_launcher.common.execdb import (
    ExecutionDB,
    JobData,
    generate_invocation_id,
    generate_job_id,
)
from nemo_evaluator_launcher.common.helpers import (
    CmdAndReadableComment,
    _str_to_echo_command,
    get_api_key_name,
    get_eval_factory_command,
    get_eval_factory_dataset_size_from_run_config,
    get_timestamp_string,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import (
    get_task_from_mapping,
    load_tasks_mapping,
)
from nemo_evaluator_launcher.common.printing_utils import bold, cyan, grey, red
from nemo_evaluator_launcher.executors.base import (
    BaseExecutor,
    ExecutionState,
    ExecutionStatus,
)
from nemo_evaluator_launcher.executors.registry import register_executor


@register_executor("slurm")
class SlurmExecutor(BaseExecutor):
    @staticmethod
    def execute_eval(cfg: DictConfig, dry_run: bool = False) -> str:
        """Submit evaluation jobs to a SLURM cluster using the provided configuration.

        Args:
            cfg: The configuration object for the evaluation run.
            dry_run: If True, prepare scripts and save them without submission.

        Returns:
            str: The invocation ID for the evaluation run.

        Raises:
            AssertionError: If deployment type is 'none'.
            RuntimeError: If remote directory creation or sbatch submission fails.
        """

        # Generate invocation ID
        invocation_id = generate_invocation_id()

        local_runsub_paths = []
        remote_runsub_paths = []

        with tempfile.TemporaryDirectory() as tmpdirname:
            timestamp = get_timestamp_string(include_microseconds=False)
            rundir_name = timestamp + "-" + invocation_id
            remote_rundir = Path(cfg.execution.output_dir) / rundir_name
            local_rundir = Path(tmpdirname) / rundir_name
            local_rundir.mkdir()

            # Preload mapping for image resolution
            tasks_mapping = load_tasks_mapping()
            eval_images: list[str] = []

            is_potentially_unsafe = False
            for idx, task in enumerate(cfg.evaluation.tasks):
                # calculate job_id
                job_id = f"{invocation_id}.{idx}"

                # preapre locally
                remote_task_subdir = remote_rundir / task.name
                local_task_subdir = local_rundir / task.name
                local_task_subdir.mkdir()  # this ensures the task.name hasn't been used
                (local_task_subdir / "logs").mkdir()
                (local_task_subdir / "artifacts").mkdir()

                # resolve eval image and pass directly via task override
                task_definition = get_task_from_mapping(task.name, tasks_mapping)
                eval_image = task_definition["container"]
                if "container" in task:
                    eval_image = task["container"]

                eval_images.append(eval_image)

                # generate and write down sbatch script
                sbatch_script_content_struct = _create_slurm_sbatch_script(
                    cfg=cfg,
                    task=task,
                    eval_image=eval_image,
                    remote_task_subdir=remote_task_subdir,
                    invocation_id=invocation_id,
                    job_id=job_id,
                )

                # Create proxy config file with placeholder IPs for multi-instance deployments
                if cfg.deployment.get("multiple_instances", False):
                    proxy_type = cfg.execution.get("proxy", {}).get("type", "haproxy")
                    if proxy_type == "haproxy":
                        proxy_config = _generate_haproxy_config_with_placeholders(cfg)
                    else:
                        raise ValueError(
                            f"Unsupported proxy type: {proxy_type}. Currently only 'haproxy' is supported."
                        )

                    # Save both template and working config
                    proxy_template_path = local_task_subdir / "proxy.cfg.template"
                    proxy_config_path = local_task_subdir / "proxy.cfg"
                    with open(proxy_template_path, "w") as f:
                        f.write(proxy_config)
                    with open(proxy_config_path, "w") as f:
                        f.write(proxy_config)

                sbatch_script_content_str = sbatch_script_content_struct.cmd

                # We accumulate if any task contains unsafe commands
                is_potentially_unsafe = (
                    is_potentially_unsafe
                    or sbatch_script_content_struct.is_potentially_unsafe
                )
                local_runsub_path = local_task_subdir / "run.sub"
                remote_runsub_path = remote_task_subdir / "run.sub"
                with open(local_runsub_path, "w") as f:
                    f.write(sbatch_script_content_str.rstrip("\n") + "\n")

                local_runsub_paths.append(local_runsub_path)
                remote_runsub_paths.append(remote_runsub_path)

            if dry_run:
                print(bold("\n\n=============================================\n\n"))
                print(bold(cyan("DRY RUN: SLURM scripts prepared")))
                for idx, local_runsub_path in enumerate(local_runsub_paths):
                    print(cyan(f"\n\n=========== Task {idx} =====================\n\n"))
                    with open(local_runsub_path, "r") as f:
                        print(grey(f.read()))
                print(bold("To submit jobs") + ", run the executor without --dry-run")
                if is_potentially_unsafe:
                    print(
                        red(
                            "\nFound `pre_cmd` (evaluation or deployment) which carries security risk. When running without --dry-run "
                            "make sure you trust the command and set NEMO_EVALUATOR_TRUST_PRE_CMD=1"
                        )
                    )

                return invocation_id

            if is_potentially_unsafe:
                if os.environ.get("NEMO_EVALUATOR_TRUST_PRE_CMD", "") == "1":
                    logger.warning(
                        "Found non-empty commands (e.g. `pre_cmd` in evaluation or deployment) and NEMO_EVALUATOR_TRUST_PRE_CMD "
                        "is set, proceeding with caution."
                    )

                else:
                    logger.error(
                        "Found non-empty commands (e.g. `pre_cmd` in evaluation or deployment) and NEMO_EVALUATOR_TRUST_PRE_CMD "
                        "is not set. This might carry security risk and unstable environments. "
                        "To continue, make sure you trust the command and set NEMO_EVALUATOR_TRUST_PRE_CMD=1.",
                    )
                    raise AttributeError(
                        "Untrusted command found in config, make sure you trust and "
                        "set NEMO_EVALUATOR_TRUST_PRE_CMD=1."
                    )

            socket = str(Path(tmpdirname) / "socket")
            socket_or_none = _open_master_connection(
                username=cfg.execution.username,
                hostname=cfg.execution.hostname,
                socket=socket,
            )
            _make_remote_execution_output_dir(
                dirpath=cfg.execution.output_dir,
                username=cfg.execution.username,
                hostname=cfg.execution.hostname,
                socket=socket_or_none,
            )
            _rsync_upload_rundirs(
                local_sources=[local_rundir],
                remote_target=cfg.execution.output_dir,
                username=cfg.execution.username,
                hostname=cfg.execution.hostname,
            )
            slurm_job_ids = _sbatch_remote_runsubs(
                remote_runsub_paths=remote_runsub_paths,
                username=cfg.execution.username,
                hostname=cfg.execution.hostname,
                socket=socket_or_none,
            )
            _close_master_connection(
                username=cfg.execution.username,
                hostname=cfg.execution.hostname,
                socket=socket_or_none,
            )

            # save launched jobs metadata
            db = ExecutionDB()
            for idx, (slurm_job_id, remote_runsub_path) in enumerate(
                zip(slurm_job_ids, remote_runsub_paths)
            ):
                job_id = generate_job_id(invocation_id, idx)
                db.write_job(
                    job=JobData(
                        invocation_id=invocation_id,
                        job_id=job_id,
                        timestamp=time.time(),
                        executor="slurm",
                        data={
                            "slurm_job_id": slurm_job_id,
                            "remote_rundir_path": str(remote_runsub_path.parent),
                            "hostname": cfg.execution.hostname,
                            "username": cfg.execution.username,
                            "eval_image": eval_images[idx],
                        },
                        config=OmegaConf.to_object(cfg),
                    )
                )
            return invocation_id

    @staticmethod
    def get_status(id: str) -> List[ExecutionStatus]:
        """Get the status of a specific SLURM job or all jobs in an invocation group.

        Args:
            id: Unique job identifier or invocation identifier.

        Returns:
            List containing the execution status for the job(s).
        """
        db = ExecutionDB()

        # If id looks like an invocation_id
        if "." not in id:
            jobs = db.get_jobs(id)
            if not jobs:
                return []
            return SlurmExecutor._get_status_for_invocation(jobs)

        # Otherwise, treat as job_id
        else:
            job_data = db.get_job(id)
            if job_data is None or job_data.executor != "slurm":
                return []
            return [SlurmExecutor._get_status_for_job(id, job_data)]

    @staticmethod
    def _get_status_for_job(id: str, job_data: JobData) -> ExecutionStatus:
        slurm_job_id = job_data.data.get("slurm_job_id")
        if not slurm_job_id:
            return ExecutionStatus(id=id, state=ExecutionState.FAILED)

        try:
            return SlurmExecutor._query_slurm_for_status_and_progress(
                slurm_job_ids=[slurm_job_id],
                remote_rundir_paths=[Path(job_data.data.get("remote_rundir_path"))],
                username=job_data.data["username"],
                hostname=job_data.data["hostname"],
                job_id_to_execdb_id={slurm_job_id: id},
            )[0]
        except Exception:
            return ExecutionStatus(id=id, state=ExecutionState.FAILED)

    @staticmethod
    def _get_status_for_invocation(jobs: dict) -> List[ExecutionStatus]:
        slurm_job_ids = []
        remote_rundir_paths = []
        job_id_to_execdb_id = {}
        username = None
        hostname = None

        for job_id, job_data in jobs.items():
            if job_data.executor != "slurm":
                continue
            slurm_job_id = job_data.data.get("slurm_job_id")
            if slurm_job_id:
                slurm_job_ids.append(slurm_job_id)
                remote_rundir_paths.append(
                    Path(job_data.data.get("remote_rundir_path"))
                )
                job_id_to_execdb_id[slurm_job_id] = job_id
                username = job_data.data.get("username")
                hostname = job_data.data.get("hostname")

        if not slurm_job_ids or not remote_rundir_paths or not username or not hostname:
            return [
                ExecutionStatus(id=job_id, state=ExecutionState.FAILED)
                for job_id in jobs.keys()
            ]

        try:
            return SlurmExecutor._query_slurm_for_status_and_progress(
                slurm_job_ids=slurm_job_ids,
                remote_rundir_paths=remote_rundir_paths,
                username=username,
                hostname=hostname,
                job_id_to_execdb_id=job_id_to_execdb_id,
            )
        except Exception:
            return [
                ExecutionStatus(id=job_id, state=ExecutionState.FAILED)
                for job_id in jobs.keys()
            ]

    @staticmethod
    def _query_slurm_for_status_and_progress(
        slurm_job_ids: List[str],
        remote_rundir_paths: List[Path],
        username: str,
        hostname: str,
        job_id_to_execdb_id: dict,
    ) -> List[ExecutionStatus]:
        with tempfile.TemporaryDirectory() as tmpdirname:
            socket = str(Path(tmpdirname) / "socket")
            socket_or_none = _open_master_connection(
                username=username,
                hostname=hostname,
                socket=socket,
            )
            # get slurm job status for initial jobs:
            slurm_jobs_status = _query_slurm_jobs_status(
                slurm_job_ids=slurm_job_ids,
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
            # handle slurm status for autoresumed jobs:
            autoresumed_slurm_job_ids = _read_autoresumed_slurm_job_ids(
                slurm_job_ids=slurm_job_ids,
                remote_rundir_paths=remote_rundir_paths,
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
            latest_slurm_job_ids = {
                slurm_job_id: slurm_job_id_list[-1]
                for slurm_job_id, slurm_job_id_list in autoresumed_slurm_job_ids.items()
                if len(slurm_job_id_list) > 0 and slurm_job_id_list[-1] != slurm_job_id
            }
            latest_slurm_jobs_status = _query_slurm_jobs_status(
                slurm_job_ids=list(latest_slurm_job_ids.values()),
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
            # get progress:
            progress_list = _get_progress(
                remote_rundir_paths=remote_rundir_paths,
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
            _close_master_connection(
                username=username,
                hostname=hostname,
                socket=socket_or_none,
            )
        statuses = []
        for i, slurm_job_id in enumerate(slurm_job_ids):
            slurm_status = slurm_jobs_status[slurm_job_id]
            if slurm_job_id in latest_slurm_job_ids:
                latest_slurm_job_id = latest_slurm_job_ids[slurm_job_id]
                slurm_status = latest_slurm_jobs_status[latest_slurm_job_id]
            progress = progress_list[i]
            progress = progress if progress is not None else "unknown"
            execution_state = SlurmExecutor._map_slurm_state_to_execution_state(
                slurm_status
            )
            execdb_job_id = job_id_to_execdb_id.get(slurm_job_id)
            if execdb_job_id:
                statuses.append(
                    ExecutionStatus(
                        id=execdb_job_id,
                        state=execution_state,
                        progress=progress,
                    )
                )
        return statuses

    @staticmethod
    def _map_slurm_state_to_execution_state(slurm_status: str) -> ExecutionState:
        """Map SLURM state to ExecutionState.

        Args:
            slurm_status: SLURM status string.

        Returns:
            Corresponding ExecutionState.
        """
        if slurm_status in ["COMPLETED"]:
            return ExecutionState.SUCCESS
        elif slurm_status in [
            "PENDING",
            "RESV_DEL_HOLD",
            "REQUEUE_FED",
            "REQUEUE_HOLD",
            "REQUEUED",
            "REVOKED",
        ]:
            return ExecutionState.PENDING
        elif slurm_status in ["RUNNING", "CONFIGURING", "SUSPENDED", "COMPLETING"]:
            return ExecutionState.RUNNING
        elif slurm_status in ["PREEMPTED", "TIMEOUT", "NODE_FAIL"]:
            return ExecutionState.PENDING  # autoresume
        elif slurm_status in ["CANCELLED"]:
            return ExecutionState.KILLED
        elif slurm_status in ["FAILED"]:
            return ExecutionState.FAILED
        else:
            return ExecutionState.FAILED

    @staticmethod
    def kill_job(job_id: str) -> None:
        """Kill a SLURM job.

        Args:
            job_id: The job ID (e.g., abc123.0) to kill.
        """
        db = ExecutionDB()
        job_data = db.get_job(job_id)

        if job_data is None:
            raise ValueError(f"Job {job_id} not found")

        if job_data.executor != "slurm":
            raise ValueError(
                f"Job {job_id} is not a slurm job (executor: {job_data.executor})"
            )

        # OPTIMIZATION: Query status AND kill in ONE SSH call
        slurm_status, result = _kill_slurm_job(
            slurm_job_ids=[job_data.data.get("slurm_job_id")],
            username=job_data.data.get("username"),
            hostname=job_data.data.get("hostname"),
            socket=job_data.data.get("socket"),
        )

        # Mark job as killed in database if kill succeeded
        if result.returncode == 0:
            job_data.data["killed"] = True
            db.write_job(job_data)
        else:
            # Use the pre-fetched status for better error message
            current_status = None
            if slurm_status:
                current_status = SlurmExecutor._map_slurm_state_to_execution_state(
                    slurm_status
                )
            error_msg = SlurmExecutor.get_kill_failure_message(
                job_id,
                f"slurm_job_id: {job_data.data.get('slurm_job_id')}",
                current_status,
            )
            raise RuntimeError(error_msg)


def _create_slurm_sbatch_script(
    cfg: DictConfig,
    task: DictConfig,
    eval_image: str,
    remote_task_subdir: Path,
    invocation_id: str,
    job_id: str,
) -> CmdAndReadableComment:
    """Generate the contents of a SLURM sbatch script for a given evaluation task.

    Args:
        cfg: The configuration object for the evaluation run.
        task: The evaluation task configuration.
        remote_task_subdir: The remote directory path for the `run.sub` file.
        invocation_id: The invocation ID for this evaluation run.
        job_id: The complete job ID string.

    Returns:
        str: The contents of the sbatch script.
    """
    # get task from mapping, overrides, urls
    tasks_mapping = load_tasks_mapping()
    task_definition = get_task_from_mapping(task.name, tasks_mapping)

    # TODO(public release): convert to template
    s = "#!/bin/bash\n"

    # SBATCH headers
    s += "#SBATCH --time {}\n".format(cfg.execution.walltime)
    s += "#SBATCH --account {}\n".format(cfg.execution.account)
    s += "#SBATCH --partition {}\n".format(cfg.execution.partition)
    s += "#SBATCH --nodes {}\n".format(cfg.execution.num_nodes)
    s += "#SBATCH --ntasks-per-node {}\n".format(cfg.execution.ntasks_per_node)
    if cfg.execution.get("gpus_per_node", None) is not None:
        s += "#SBATCH --gpus-per-node {}\n".format(cfg.execution.gpus_per_node)
    if hasattr(cfg.execution, "gres"):
        s += "#SBATCH --gres {}\n".format(cfg.execution.gres)
    if cfg.execution.get("sbatch_comment"):
        s += "#SBATCH --comment='{}'\n".format(cfg.execution.sbatch_comment)
    job_name = "{account}-{subproject}.{details}".format(
        account=cfg.execution.account,
        subproject=cfg.execution.subproject,
        details=remote_task_subdir.name,
    )
    s += "#SBATCH --job-name {}\n".format(job_name)
    s += "#SBATCH --exclusive\n"
    s += "#SBATCH --output {}\n".format(remote_task_subdir / "logs" / "slurm-%A.out")
    s += "\n"
    s += f'TASK_DIR="{str(remote_task_subdir)}"\n'
    s += "\n"

    # collect all env vars
    env_vars = copy.deepcopy(dict(cfg.evaluation.get("env_vars", {})))
    env_vars.update(task.get("env_vars", {}))
    api_key_name = get_api_key_name(cfg)
    if api_key_name:
        assert "API_KEY" not in env_vars
        env_vars["API_KEY"] = api_key_name

    # check if the environment variables are set
    for env_var in env_vars.values():
        if os.getenv(env_var) is None:
            raise ValueError(f"Trying to pass an unset environment variable {env_var}.")

    # check if required env vars are defined:
    for required_env_var in task_definition.get("required_env_vars", []):
        if required_env_var not in env_vars.keys():
            raise ValueError(
                f"{task.name} task requires environment variable {required_env_var}."
                " Specify it in the task subconfig in the 'env_vars' dict as the following"
                f" pair {required_env_var}: YOUR_ENV_VAR_NAME"
            )

    # save env vars:
    for env_var_dst, env_var_src in env_vars.items():
        s += f"export {env_var_dst}={os.getenv(env_var_src)}\n"
    all_env_vars = {
        **cfg.execution.get("env_vars", {}).get("deployment", {}),
        **cfg.execution.get("env_vars", {}).get("evaluation", {}),
    }
    if cfg.deployment.get("env_vars"):
        warnings.warn(
            "cfg.deployment.env_vars will be deprecated in future versions. "
            "Use cfg.execution.env_vars.deployment instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        all_env_vars.update(cfg.deployment["env_vars"])
    for env_var_dst, env_var_value in all_env_vars.items():
        s += f"export {env_var_dst}={env_var_value}\n"
    if env_vars:
        s += "\n"

    # auto resume after timeout
    s += _AUTORESUME_HANDLER
    s += "\n\n"

    # echo the current SLURM_JOB_ID
    s += "# save the current job id\n"
    s += "echo $SLURM_JOB_ID >> {}\n\n".format(
        remote_task_subdir / ".slurm_job_id.list"
    )

    # shell options
    s += "set -e  # exit immediately if any command exits with a non-zero status\n"
    s += "set -u  # treat unset variables as an error when substituting\n"
    s += "set -x  # print commands and their arguments as they are executed\n"
    s += "\n"

    # prepare deployment mounts
    deployment_mounts_list = []
    deployment_is_unsafe = False
    if cfg.deployment.type != "none":
        if checkpoint_path := cfg.deployment.get("checkpoint_path"):
            deployment_mounts_list.append(f"{checkpoint_path}:/checkpoint:ro")
        if cache_path := cfg.deployment.get("cache_path"):
            deployment_mounts_list.append(f"{cache_path}:/cache")
        for source_mnt, target_mnt in (
            cfg.execution.get("mounts", {}).get("deployment", {}).items()
        ):
            deployment_mounts_list.append(f"{source_mnt}:{target_mnt}")

        # add deployment srun command
        deployment_srun_cmd, deployment_is_unsafe, deployment_debug = (
            _generate_deployment_srun_command(
                cfg, deployment_mounts_list, remote_task_subdir
            )
        )
        s += deployment_srun_cmd

        # wait for the server to initialize
        health_path = cfg.deployment.get("health_check_path", "/health")
        # For multi-instance check all node IPs, for single instance check localhost
        if cfg.deployment.get("multiple_instances", False):
            ip_list = '"${NODES_IPS_ARRAY[@]}"'
        else:
            ip_list = '"127.0.0.1"'
        s += _get_wait_for_server_handler(
            ip_list,
            cfg.deployment.port,
            health_path,
            "server",
            check_pid=True,
        )
        s += "\n\n"

        # add proxy load balancer for multi-instance deployments
        if cfg.deployment.get("multiple_instances", False):
            s += _get_proxy_server_srun_command(cfg, remote_task_subdir)

    # prepare evaluation mounts
    evaluation_mounts_list = [
        "{}:/results".format(remote_task_subdir / "artifacts"),
    ]
    for source_mnt, target_mnt in (
        cfg.execution.get("mounts", {}).get("evaluation", {}).items()
    ):
        evaluation_mounts_list.append(f"{source_mnt}:{target_mnt}")

    eval_factory_command_struct = get_eval_factory_command(
        cfg,
        task,
        task_definition,
    )
    eval_factory_command = eval_factory_command_struct.cmd
    # The debug comment for placing into the script and easy debug. Reason
    # (see `CmdAndReadableComment`) is the current way of passing the command
    # is base64-encoded config `echo`-ed into file.
    # TODO(agronskiy): cleaner way is to encode everything with base64, not
    # some parts (like ef_config.yaml) and just output as logs somewhere.
    eval_factory_command_debug_comment = eval_factory_command_struct.debug

    # add evaluation srun command
    s += "# Debug contents of the eval factory command's config\n"
    s += eval_factory_command_debug_comment
    s += "\n\n"

    s += "# evaluation client\n"
    s += "srun --mpi pmix --overlap "
    s += "--nodes 1 --ntasks 1 "  # Client always runs on single node
    s += "--container-image {} ".format(eval_image)
    evaluation_env_var_names = list(
        cfg.execution.get("env_vars", {}).get("evaluation", {})
    )
    if evaluation_env_var_names:
        s += "--container-env {} ".format(",".join(evaluation_env_var_names))
    if not cfg.execution.get("mounts", {}).get("mount_home", True):
        s += "--no-container-mount-home "

    s += "--container-mounts {} ".format(",".join(evaluation_mounts_list))
    s += "--output {} ".format(remote_task_subdir / "logs" / "client-%A.out")
    s += "bash -c '\n"
    s += eval_factory_command
    s += "'\n\n"

    # terminate the server after all evaluation clients finish
    if cfg.deployment.type != "none":
        s += "kill $SERVER_PID  # terminate the server to finish gracefully\n"
        if cfg.deployment.get("multiple_instances", False):
            s += "kill $PROXY_PID  # terminate proxy to finish gracefully\n"
        s += "\n"

    # auto-export
    ae_cfg = cfg.execution.get("auto_export")
    destinations: list = []
    if isinstance(ae_cfg, list):
        destinations = list(ae_cfg)
    elif isinstance(ae_cfg, dict) or isinstance(ae_cfg, DictConfig):
        destinations = list(ae_cfg.get("destinations", []) or [])

    if destinations:
        export_env = dict(cfg.execution.get("env_vars", {}).get("export", {}) or {})
        s += _generate_auto_export_section(cfg, job_id, destinations, export_env)

    debug_str = "\n".join(["# " + line for line in s.splitlines()])

    # Combine unsafe flags from both deployment and evaluation
    is_potentially_unsafe = (
        eval_factory_command_struct.is_potentially_unsafe or deployment_is_unsafe
    )

    return CmdAndReadableComment(
        cmd=s,
        debug=debug_str,
        is_potentially_unsafe=is_potentially_unsafe,
    )


def _generate_auto_export_section(
    cfg: DictConfig,
    job_id: str,
    destinations: list,
    export_env: dict,
) -> str:
    """Generate simple auto-export section for sbatch script."""
    if not destinations:
        return ""

    s = "\n# Auto-export on success\n"
    s += "EVAL_EXIT_CODE=$?\n"
    s += "if [ $EVAL_EXIT_CODE -eq 0 ]; then\n"
    s += "    echo 'Evaluation completed successfully. Starting auto-export...'\n"
    s += "    set +e\n"
    s += "    set +x\n"
    s += "    set +u\n"
    s += '    cd "$TASK_DIR/artifacts"\n'

    # Work with DictConfig; convert only for YAML at the end
    exec_type = (
        cfg.execution.type
        if hasattr(cfg.execution, "type")
        else cfg.execution.get("type", "slurm")
    )
    eval_tasks = (
        list(cfg.evaluation.tasks)
        if hasattr(cfg, "evaluation") and hasattr(cfg.evaluation, "tasks")
        else list((cfg.get("evaluation", {}) or {}).get("tasks", []) or [])
    )
    export_block = cfg.get("export", {}) or {}

    payload = {
        "execution": {
            "auto_export": {
                "destinations": list(destinations),
                **({"env_vars": dict(export_env)} if export_env else {}),
            },
            "type": exec_type,
        },
        "evaluation": {"tasks": eval_tasks},
    }
    if export_block:
        # Convert just this block to plain for YAML
        payload["export"] = (
            OmegaConf.to_object(export_block)
            if OmegaConf.is_config(export_block)
            else dict(export_block)
        )

    # Final YAML (single conversion at the end)
    payload_clean = OmegaConf.to_container(OmegaConf.create(payload), resolve=True)
    yaml_str = yaml.safe_dump(payload_clean, sort_keys=False)
    s += "    cat > export_config.yml << 'EOF'\n"
    s += yaml_str
    s += "EOF\n"

    # write launcher config as config.yml for exporters (no core command)
    submitted_yaml = yaml.safe_dump(
        OmegaConf.to_container(cfg, resolve=True), sort_keys=False
    )
    s += "    cat > config.yml << 'EOF'\n"
    s += submitted_yaml
    s += "EOF\n"

    # Export host only env before running auto export
    for k, v in (export_env or {}).items():
        if isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
            s += f'    export {k}="${{{v}}}"\n'
        else:
            esc = str(v).replace('"', '\\"')
            s += f'    export {k}="{esc}"\n'

    for dest in destinations:
        s += f"    echo 'Exporting to {dest}...'\n"
        s += f"    nemo-evaluator-launcher export {job_id} --dest {dest} || echo 'Export to {dest} failed'\n"

    s += "    echo 'Auto-export completed.'\n"
    s += "else\n"
    s += "    echo 'Evaluation failed with exit code $EVAL_EXIT_CODE. Skipping auto-export.'\n"
    s += "fi\n"

    return s


def _open_master_connection(
    username: str,
    hostname: str,
    socket: str,
) -> str | None:
    ssh_command = f"ssh -MNf -S {socket} {username}@{hostname}"
    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )
    if completed_process.returncode == 0:
        return socket
    return None


def _close_master_connection(
    username: str,
    hostname: str,
    socket: str | None,
) -> None:
    if socket is None:
        return
    ssh_command = f"ssh -O exit -S {socket} {username}@{hostname}"
    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )
    if completed_process.returncode != 0:
        raise RuntimeError(
            "failed to close the master connection\n{}".format(
                completed_process.stderr.decode("utf-8")
            )
        )


def _make_remote_execution_output_dir(
    dirpath: str,
    username: str,
    hostname: str,
    socket: str | None,
) -> None:
    mkdir_command = f"mkdir -p {dirpath}"
    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(mkdir_command)
    ssh_command = " ".join(ssh_command)
    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )
    if completed_process.returncode != 0:
        error_msg = (
            completed_process.stderr.decode("utf-8")
            if completed_process.stderr
            else "Unknown error"
        )
        raise RuntimeError(
            "failed to make a remote execution output dir\n{}".format(error_msg)
        )


def _rsync_upload_rundirs(
    local_sources: List[Path],
    remote_target: str,
    username: str,
    hostname: str,
) -> None:
    """Upload local run directories to a remote host using rsync over SSH.

    Args:
        local_sources: List of local Path objects to upload.
        remote_target: Remote directory path as a string.
        hostname: SSH hostname.
        username: SSH username.

    Raises:
        RuntimeError: If rsync fails.
    """
    for local_source in local_sources:
        assert local_source.is_dir()
    remote_destination_str = f"{username}@{hostname}:{remote_target}"
    local_sources_str = " ".join(map(str, local_sources))
    rsync_upload_command = f"rsync -qcaz {local_sources_str} {remote_destination_str}"
    completed_process = subprocess.run(
        args=shlex.split(rsync_upload_command), capture_output=True
    )
    if completed_process.returncode != 0:
        error_msg = (
            completed_process.stderr.decode("utf-8")
            if completed_process.stderr
            else "Unknown error"
        )
        raise RuntimeError("failed to upload local sources\n{}".format(error_msg))


def _sbatch_remote_runsubs(
    remote_runsub_paths: List[Path],
    username: str,
    hostname: str,
    socket: str | None,
) -> List[str]:
    sbatch_commands = [
        "sbatch {}".format(remote_runsub_path)
        for remote_runsub_path in remote_runsub_paths
    ]
    sbatch_commands = " ; ".join(sbatch_commands)

    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(sbatch_commands)
    ssh_command = " ".join(ssh_command)

    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )
    if completed_process.returncode != 0:
        error_msg = completed_process.stderr.decode("utf-8")
        raise RuntimeError(
            "failed to submit sbatch scripts for execution\n{}".format(error_msg)
        )

    sbatch_output = completed_process.stdout.decode("utf-8")
    slurm_job_ids = re.findall(r"(?<=Submitted batch job )\d+", sbatch_output)
    return slurm_job_ids


def _query_slurm_jobs_status(
    slurm_job_ids: List[str],
    username: str,
    hostname: str,
    socket: str | None,
) -> Dict[str, str]:
    """Query SLURM for job statuses using sacct command.

    Args:
        slurm_job_ids: List of SLURM job IDs to query.
        username: SSH username.
        hostname: SSH hostname.
        socket: control socket location or None

    Returns:
        Dict mapping from slurm_job_id to returned slurm status.
    """
    if len(slurm_job_ids) == 0:
        return {}
    sacct_command = "sacct -j {} --format='JobID,State%32' --noheader -P".format(
        ",".join(slurm_job_ids)
    )
    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(sacct_command)
    ssh_command = " ".join(ssh_command)
    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )
    if completed_process.returncode != 0:
        raise RuntimeError(
            "failed to query slurm job status\n{}".format(
                completed_process.stderr.decode("utf-8")
            )
        )
    sacct_output = completed_process.stdout.decode("utf-8")
    sacct_output_lines = sacct_output.strip().split("\n")
    slurm_jobs_status = {}
    for slurm_job_id in slurm_job_ids:
        slurm_job_status = _parse_slurm_job_status(slurm_job_id, sacct_output_lines)
        slurm_jobs_status[slurm_job_id] = slurm_job_status
    return slurm_jobs_status


def _kill_slurm_job(
    slurm_job_ids: List[str], username: str, hostname: str, socket: str | None
) -> tuple[str | None, subprocess.CompletedProcess]:
    """Kill a SLURM job, querying status first in one SSH call for efficiency.

    Args:
        slurm_job_ids: List of SLURM job IDs to kill.
        username: SSH username.
        hostname: SSH hostname.
        socket: control socket location or None

    Returns:
        Tuple of (status_string, completed_process) where status_string is the SLURM status or None
    """
    if len(slurm_job_ids) == 0:
        return None, subprocess.CompletedProcess(args=[], returncode=0)

    jobs_str = ",".join(slurm_job_ids)
    # Combine both commands in one SSH call: query THEN kill
    combined_command = (
        f"sacct -j {jobs_str} --format='JobID,State%32' --noheader -P 2>/dev/null; "
        f"scancel {jobs_str}"
    )

    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(combined_command)
    ssh_command = " ".join(ssh_command)

    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )

    # Parse the sacct output (before scancel runs)
    sacct_output = completed_process.stdout.decode("utf-8")
    sacct_output_lines = sacct_output.strip().split("\n")
    slurm_status = None
    if sacct_output_lines and len(slurm_job_ids) == 1:
        slurm_status = _parse_slurm_job_status(slurm_job_ids[0], sacct_output_lines)

    return slurm_status, completed_process


def _parse_slurm_job_status(slurm_job_id: str, sacct_output_lines: List[str]) -> str:
    """Parse SLURM job status from sacct output for a specific job.

    Args:
        slurm_job_id: The SLURM job ID to parse.
        sacct_output_lines: Lines from sacct output.

    Returns:
        SLURM status string.
    """
    for line in sacct_output_lines:
        if line.startswith(f"{slurm_job_id}|"):
            state = line.split("|")[1]
            state = state.strip()
            if state:
                state_split = state.split()
                if len(state_split) > 0:
                    return state_split[0]
    return "UNKNOWN"


def _read_autoresumed_slurm_job_ids(
    slurm_job_ids: List[str],
    remote_rundir_paths: List[Path],
    username: str,
    hostname: str,
    socket: str | None,
) -> Dict[str, List[str]]:
    assert len(slurm_job_ids) == len(remote_rundir_paths)
    slurm_job_id_list_paths = [
        str(remote_rundir_path / ".slurm_job_id.list")
        for remote_rundir_path in remote_rundir_paths
    ]
    slurm_job_id_list_strs = _read_files_from_remote(
        slurm_job_id_list_paths, username, hostname, socket
    )
    assert len(slurm_job_id_list_strs) == len(slurm_job_ids)
    autoresumed_slurm_job_ids = {}
    for i, slurm_job_id_list_str in enumerate(slurm_job_id_list_strs):
        slurm_job_id = slurm_job_ids[i]
        slurm_job_id_list = slurm_job_id_list_str.split()
        autoresumed_slurm_job_ids[slurm_job_id] = slurm_job_id_list
    return autoresumed_slurm_job_ids


def _read_files_from_remote(
    filepaths: List[Path],
    username: str,
    hostname: str,
    socket: str | None,
) -> List[str]:
    cat_commands = [
        "echo _START_OF_FILE_ ; cat {} 2>/dev/null ; echo _END_OF_FILE_ ".format(
            filepath
        )
        for filepath in filepaths
    ]
    cat_commands = " ; ".join(cat_commands)
    ssh_command = ["ssh"]
    if socket is not None:
        ssh_command.append(f"-S {socket}")
    ssh_command.append(f"{username}@{hostname}")
    ssh_command.append(cat_commands)
    ssh_command = " ".join(ssh_command)
    completed_process = subprocess.run(
        args=shlex.split(ssh_command), capture_output=True
    )
    if completed_process.returncode != 0:
        raise RuntimeError(
            "failed to read files from remote\n{}".format(
                completed_process.stderr.decode("utf-8")
            )
        )
    cat_outputs = completed_process.stdout.decode("utf-8")
    cat_outputs = cat_outputs.replace("\n", " ")
    matches = re.findall(r"(?<=_START_OF_FILE_)(.*?)(?=_END_OF_FILE_)", cat_outputs)
    outputs = [match.strip() for match in matches]
    return outputs


def _get_progress(
    remote_rundir_paths: List[Path],
    username: str,
    hostname: str,
    socket: str | None,
) -> List[Optional[float]]:
    remote_progress_paths = [
        remote_rundir_path / "artifacts" / "progress"
        for remote_rundir_path in remote_rundir_paths
    ]
    remote_run_config_paths = [
        remote_rundir_path / "artifacts" / "run_config.yml"
        for remote_rundir_path in remote_rundir_paths
    ]
    progress_strs = _read_files_from_remote(
        remote_progress_paths, username, hostname, socket
    )
    if any(map(bool, progress_strs)):
        run_config_strs = _read_files_from_remote(
            remote_run_config_paths, username, hostname, socket
        )
    else:
        run_config_strs = [""] * len(progress_strs)
    progress_list = []
    for progress_str, run_config_str in zip(progress_strs, run_config_strs):
        if not progress_str or not run_config_str:
            progress_list.append(None)
            continue
        run_config = yaml.safe_load(run_config_str)
        dataset_size = get_eval_factory_dataset_size_from_run_config(run_config)
        if dataset_size is not None:
            progress = int(progress_str) / dataset_size
        else:
            progress = int(progress_str)
        progress_list.append(progress)
    return progress_list


_AUTORESUME_HANDLER = """
_this_script=$0
_prev_slurm_job_id=$1
# Handle automatic resumption after some failed state.
if [[ "$_prev_slurm_job_id" != "" ]]; then
    _prev_state=`sacct -j $_prev_slurm_job_id -P -n -o State | head -n 1`
    _prev_info="previous SLURM_JOB_ID $_prev_slurm_job_id finished with '$_prev_state' state."
    if [[ $_prev_state == 'TIMEOUT' || $_prev_state == 'PREEMPTED' || $_prev_state == 'NODE_FAIL' ]]; then
        echo "$_prev_info RESUMING..."
    else
        echo "$_prev_info EXIT!"
        if [[ $_prev_state == 'COMPLETED' ]]; then
            exit 0
        else
            exit 1
        fi
    fi
fi
# Schedule next execution of this script  with the current $SLURM_JOB_ID as an argument.
# "afternotok" means next execution will be invoked only if the current execution terminates in some failed state.
sbatch --dependency=afternotok:$SLURM_JOB_ID $_this_script $SLURM_JOB_ID
""".strip()


def _generate_haproxy_config_with_placeholders(cfg):
    """Generate HAProxy configuration with placeholder IPs using Jinja template."""
    # Set up Jinja environment
    template_dir = Path(__file__).parent
    template_path = template_dir / "proxy.cfg.template"

    if not template_path.exists():
        raise FileNotFoundError(f"Proxy template not found: {template_path}")

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("proxy.cfg.template")

    # Prepare template data with placeholder IPs - use actual number of nodes
    num_nodes = cfg.execution.num_nodes
    nodes = []
    for i in range(num_nodes):
        nodes.append({"ip": f"{{IP_{i}}}", "port": cfg.deployment.port})

    # Get health check parameters from execution config
    proxy_config = cfg.execution.get("proxy", {}).get("config", {})
    health_check_path = proxy_config.get("health_check_path", "/health")
    health_check_status = proxy_config.get("health_check_status", 200)
    haproxy_port = proxy_config.get("haproxy_port", 5009)

    # Render template
    config = template.render(
        haproxy_port=haproxy_port,
        health_check_path=health_check_path,
        health_check_status=health_check_status,
        nodes=nodes,
    )

    return config


def _generate_haproxy_config(cfg, nodes_ips):
    """Generate HAProxy configuration using Jinja template."""
    # Set up Jinja environment
    template_dir = Path(__file__).parent
    template_path = template_dir / "proxy.cfg.template"

    if not template_path.exists():
        raise FileNotFoundError(f"Proxy template not found: {template_path}")

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("proxy.cfg.template")

    # Prepare template data
    nodes = []
    for i, ip in enumerate(nodes_ips, 1):
        nodes.append(
            {"ip": ip, "port": cfg.deployment.port}  # All nodes use the same port
        )

    # Get health check parameters from deployment config
    health_check_path = cfg.deployment.get("health_check_path", "/health")
    health_check_status = cfg.deployment.get("health_check_status", 200)
    haproxy_port = cfg.deployment.get("haproxy_port", 5009)

    # Render template
    config = template.render(
        haproxy_port=haproxy_port,
        health_check_path=health_check_path,
        health_check_status=health_check_status,
        nodes=nodes,
    )

    return config


def _generate_deployment_srun_command(
    cfg, deployment_mounts_list, remote_task_subdir, instance_id: int = 0
):
    """Generate the deployment srun command with proper node/ntask configuration.

    Returns:
        tuple: (script_string, is_potentially_unsafe, debug_comment)
    """
    s = ""
    debug_comment = ""
    is_potentially_unsafe = False

    s += "# deployment server\n"

    # Extract pre_cmd for later use inside container
    pre_cmd: str = cfg.deployment.get("pre_cmd") or ""
    if pre_cmd:
        is_potentially_unsafe = True
        create_pre_script_cmd = _str_to_echo_command(
            pre_cmd, filename="deployment_pre_cmd.sh"
        )
        debug_comment += create_pre_script_cmd.debug + "\n\n"

    s += "# Get node IPs\n"
    s += "nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )\n"
    s += 'nodes_array=("${nodes[@]}")  # Ensure nodes are stored properly\n'
    s += 'export NODES_IPS_ARRAY=($(for node in "${nodes_array[@]}"; do srun --nodelist=$node --ntasks=1 --nodes=1 hostname --ip-address; done))\n'
    s += 'echo "Node IPs: ${NODES_IPS_ARRAY[@]}"\n'
    s += "# Export MASTER_IP as the first node IP\n"
    s += "export MASTER_IP=${NODES_IPS_ARRAY[0]}\n"
    s += 'echo "MASTER_IP: $MASTER_IP"\n'

    # Add debug comment for deployment pre_cmd before srun command
    if debug_comment:
        s += "# Debug contents of deployment pre_cmd\n"
        s += debug_comment
        s += "\n"

    s += "srun --mpi pmix --overlap "
    s += f"--nodes {cfg.execution.num_nodes} --ntasks {cfg.execution.get('deployment', {}).get('n_tasks', 1)} "
    s += "--container-image {} ".format(cfg.deployment.image)
    if deployment_mounts_list:
        s += "--container-mounts {} ".format(",".join(deployment_mounts_list))
    if not cfg.execution.get("mounts", {}).get("mount_home", True):
        s += "--no-container-mount-home "
    s += "--output {} ".format(remote_task_subdir / "logs" / "server-%A-%t.out")

    deployment_env_var_names = list(
        cfg.execution.get("env_vars", {}).get("deployment", {})
    )
    if cfg.deployment.get("env_vars"):
        warnings.warn(
            "cfg.deployment.env_vars will be deprecated in future versions. "
            "Use cfg.execution.env_vars.deployment instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        deployment_env_var_names.extend(list(cfg.deployment["env_vars"]))

    # Always add MASTER_IP to the environment variables
    if "MASTER_IP" not in deployment_env_var_names:
        deployment_env_var_names.append("MASTER_IP")

    if deployment_env_var_names:
        s += f"--container-env {','.join(deployment_env_var_names)} "

    # Wrap deployment command to execute pre_cmd inside container if needed
    if pre_cmd:
        # Create a wrapper command that runs inside the container:
        # 1. Create deployment_pre_cmd.sh file
        # 2. Source it
        # 3. Execute the original deployment command
        create_pre_script_cmd = _str_to_echo_command(
            pre_cmd, filename="deployment_pre_cmd.sh"
        )
        # Escape single quotes in the deployment command for bash -c
        escaped_deployment_cmd = cfg.deployment.command.replace("'", "'\"'\"'")
        wrapped_command = (
            f"bash -c '{create_pre_script_cmd.cmd} && "
            f"source deployment_pre_cmd.sh && "
            f"{escaped_deployment_cmd}'"
        )
        s += "{} &\n\n".format(wrapped_command)
    else:
        s += "{} &\n\n".format(cfg.deployment.command)  # run asynchronously

    s += "SERVER_PID=$!  # capture the PID of the server background srun process\n\n"

    return s, is_potentially_unsafe, debug_comment


def _get_wait_for_server_handler(
    ip_list: str,
    port: int,
    health_check_path: str,
    service_name: str = "server",
    check_pid: bool = False,
):
    """Generate wait for server handler that takes a list of IPs."""
    pid_check = ""
    if check_pid:
        pid_check = 'kill -0 "$SERVER_PID" 2>/dev/null || { echo "Server process $SERVER_PID died"; exit 1; }'

    handler = f"""date
# wait for the {service_name} to initialize
for ip in {ip_list}; do
  echo "Waiting for {service_name} on $ip..."
  while [[ "$(curl -s -o /dev/null -w "%{{http_code}}" http://$ip:{port}{health_check_path})" != "200" ]]; do
    {pid_check}
    sleep 5
  done
  echo "{service_name} ready on $ip!"
done
date
""".strip()

    return handler


def _get_proxy_server_srun_command(cfg, remote_task_subdir):
    """Generate proxy server srun command based on proxy type."""
    proxy_type = cfg.execution.get("proxy", {}).get("type", "haproxy")

    if proxy_type == "haproxy":
        return _generate_haproxy_srun_command(cfg, remote_task_subdir)
    else:
        raise ValueError(
            f"Unsupported proxy type: {proxy_type}. Currently only 'haproxy' is supported."
        )


def _generate_haproxy_srun_command(cfg, remote_task_subdir):
    """Generate HAProxy-specific srun command using template-based config."""
    s = ""
    s += "# Proxy load balancer\n"
    s += "# Copy template to config file (important for restarts)\n"
    s += f"cp {remote_task_subdir}/proxy.cfg.template {remote_task_subdir}/proxy.cfg\n"
    s += "# Replace placeholder IPs with actual node IPs\n"
    s += f"proxy_config_file={remote_task_subdir}/proxy.cfg\n"
    s += 'for i in "${!NODES_IPS_ARRAY[@]}"; do\n'
    s += '    ip="${NODES_IPS_ARRAY[$i]}"\n'
    s += '    sed -i "s/{IP_$i}/$ip/g" "$proxy_config_file"\n'
    s += "done\n"
    s += "\n"
    s += "srun --mpi pmix --overlap "
    s += "--nodes 1 --ntasks 1 "
    s += f"--container-image {cfg.execution.get('proxy', {}).get('image', 'haproxy:latest')} "
    s += f"--container-mounts {remote_task_subdir}/proxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro "
    s += f"--output {remote_task_subdir}/logs/proxy-%A.out "
    s += "haproxy -f /usr/local/etc/haproxy/haproxy.cfg &\n"
    s += "PROXY_PID=$!  # capture the PID of the proxy background srun process\n"
    s += 'echo "Proxy started with PID: $PROXY_PID"\n\n'

    # Wait for proxy to be ready on localhost
    proxy_config = cfg.execution.get("proxy", {}).get("config", {})
    haproxy_port = proxy_config.get("haproxy_port", 5009)
    health_path = proxy_config.get("health_check_path", "/health")
    s += _get_wait_for_server_handler(
        "127.0.0.1", haproxy_port, health_path, "Proxy", check_pid=False
    )
    s += "\n"

    return s
