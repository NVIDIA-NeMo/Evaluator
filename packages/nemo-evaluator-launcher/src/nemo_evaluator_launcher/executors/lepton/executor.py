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
"""Lepton executor implementation for nemo-evaluator-launcher.

Handles deployment and evaluation using Lepton endpoints with NIM containers.
"""

import os
import time
from pathlib import Path
from typing import List

from omegaconf import DictConfig

from nemo_evaluator_launcher.common.execdb import (
    ExecutionDB,
    JobData,
    generate_invocation_id,
    generate_job_id,
)
from nemo_evaluator_launcher.common.helpers import get_eval_factory_command
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import (
    get_task_from_mapping,
    load_tasks_mapping,
)
from nemo_evaluator_launcher.common.printing_utils import red
from nemo_evaluator_launcher.executors.base import (
    BaseExecutor,
    ExecutionState,
    ExecutionStatus,
)
from nemo_evaluator_launcher.executors.registry import register_executor

from .deployment_helpers import (
    create_lepton_endpoint,
    delete_lepton_endpoint,
    get_lepton_endpoint_status,
    get_lepton_endpoint_url,
    wait_for_lepton_endpoint_ready,
)
from .job_helpers import create_lepton_job, delete_lepton_job, get_lepton_job_status


@register_executor("lepton")
class LeptonExecutor(BaseExecutor):
    @staticmethod
    def execute_eval(cfg: DictConfig, dry_run: bool = False) -> str:
        """Deploy dedicated endpoints for each task on Lepton and run evaluation jobs.

        For better resource isolation and parallel execution, each evaluation task
        gets its own dedicated endpoint deployment of the same model.

        Args:
            cfg: The configuration object for the evaluation run.
            dry_run: If True, prepare job configurations without submission.

        Returns:
            str: The invocation ID for the evaluation run.

        Raises:
            ValueError: If deployment configuration is invalid.
            RuntimeError: If endpoint deployment or evaluation fails.
        """
        if cfg.deployment.type not in ["vllm", "sglang", "nim", "none"]:
            raise ValueError(
                "LeptonExecutor supports deployment types: 'vllm', 'sglang', 'nim', 'none'"
            )

        # Load tasks mapping
        tasks_mapping = load_tasks_mapping()
        job_ids = []
        lepton_job_names = []
        endpoint_names = []  # Track multiple endpoints
        db = ExecutionDB()

        # Generate invocation ID
        invocation_id = generate_invocation_id()

        # TODO(agronskiy): the structure of this executor differs from others,
        # so the best place to check for unsafe commands yelids a bit of duplication.
        # We can't use the get_eval_factory_command here because the port is not yet
        # populated.
        # Refactor the whole thing.
        is_potentially_unsafe = False
        for idx, task in enumerate(cfg.evaluation.tasks):
            pre_cmd: str = task.get("pre_cmd") or cfg.evaluation.get("pre_cmd") or ""
            if pre_cmd:
                is_potentially_unsafe = True
                break

        # Check for deployment pre_cmd
        deployment_pre_cmd: str = cfg.deployment.get("pre_cmd") or ""
        if deployment_pre_cmd:
            is_potentially_unsafe = True

        # DRY-RUN mode
        if dry_run:
            output_dir = Path(cfg.execution.output_dir).absolute() / invocation_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Validate configuration
            _dry_run_lepton(cfg, tasks_mapping, invocation_id=invocation_id)

            if cfg.deployment.type == "none":
                print("Using existing endpoint (deployment: none)")
                print("using shared endpoint")
            else:
                print(f"with endpoint type '{cfg.deployment.type}'")

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

        # For deployment: none, we use the existing endpoint for all tasks
        if cfg.deployment.type == "none":
            print("📌 Using existing endpoint (deployment: none)")
            shared_endpoint_url = cfg.target.api_endpoint.url
            print(f"✅ Using shared endpoint: {shared_endpoint_url}")

        try:
            # Create local directory for outputs
            output_dir = Path(cfg.execution.output_dir).absolute() / invocation_id
            output_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"🚀 Processing {len(cfg.evaluation.tasks)} evaluation tasks with dedicated endpoints..."
            )

            # For deployment: none, skip endpoint creation
            if cfg.deployment.type == "none":
                print("📌 Skipping endpoint creation (using existing endpoint)")
                task_endpoints = {}
                for idx, task in enumerate(cfg.evaluation.tasks):
                    task_endpoints[idx] = {
                        "name": None,
                        "url": shared_endpoint_url,
                        "full_url": shared_endpoint_url,
                    }
            else:
                # ================================================================
                # PARALLEL ENDPOINT DEPLOYMENT
                # ================================================================
                print(
                    f"🚀 Creating {len(cfg.evaluation.tasks)} endpoints in parallel..."
                )

                import queue
                import threading

                # Generate short endpoint names for all tasks
                task_endpoints = {}
                endpoint_creation_tasks = []

                for idx, task in enumerate(cfg.evaluation.tasks):
                    # Create shorter endpoint names: e.g., "nim-gpqa-0-abc123"
                    sanitized_task_name = task.name.replace("_", "-").lower()
                    if sanitized_task_name.count(".") > 0:
                        sanitized_task_name = sanitized_task_name.split(".")[-1]
                    # Take only first 6 chars of task name to keep it short (leaving room for index)
                    short_task_name = sanitized_task_name[:6]
                    short_invocation = invocation_id[:6]
                    task_index = str(idx)
                    endpoint_name = f"{cfg.deployment.type}-{short_task_name}-{task_index}-{short_invocation}"

                    if len(endpoint_name) > 36:
                        logger.info(
                            "Lepton endpoint name will be deployed under name {task_name}",
                            task_name=task.name,
                            original=endpoint_name,
                            limit=36,
                        )
                        # Truncate task name further if needed
                        max_task_len = (
                            36
                            - len(cfg.deployment.type)
                            - len(task_index)
                            - len(short_invocation)
                            - 3
                        )  # 3 hyphens
                        short_task_name = sanitized_task_name[:max_task_len]
                        endpoint_name = f"{cfg.deployment.type}-{short_task_name}-{task_index}-{short_invocation}"
                        logger.info(
                            "Lepton endpoint name is auto-generated",
                            task_name=task.name,
                            original=endpoint_name,
                            truncated=endpoint_name,
                            limit=36,
                        )

                    logger.info(
                        "Lepton endpoint name (auto-generated)",
                        task_name=task.name,
                        endpoint_name=endpoint_name,
                    )
                    endpoint_names.append(endpoint_name)
                    endpoint_creation_tasks.append((idx, task, endpoint_name))

                # Thread function to create a single endpoint
                def create_endpoint_worker(
                    task_info: tuple[int, "DictConfig", str], result_queue: queue.Queue
                ) -> None:
                    try:
                        idx, task, endpoint_name = task_info
                        print(f"🚀 Task {task.name}: Creating endpoint {endpoint_name}")

                        # Create Lepton endpoint
                        if not create_lepton_endpoint(cfg, endpoint_name):
                            result_queue.put(
                                (
                                    idx,
                                    False,
                                    f"Failed to create endpoint {endpoint_name}",
                                    None,
                                    None,
                                )
                            )
                            return

                        # Wait for endpoint to be ready
                        print(
                            f"⏳ Task {task.name}: Waiting for endpoint {endpoint_name} to be ready..."
                        )
                        # Get timeout from config, default to 600 seconds if not set
                        endpoint_timeout = (
                            cfg.execution.get("lepton_platform", {})
                            .get("deployment", {})
                            .get("endpoint_readiness_timeout", 600)
                        )
                        if not wait_for_lepton_endpoint_ready(
                            endpoint_name, timeout=endpoint_timeout
                        ):
                            result_queue.put(
                                (
                                    idx,
                                    False,
                                    f"Endpoint {endpoint_name} failed to become ready",
                                    None,
                                    None,
                                )
                            )
                            return

                        # Get endpoint URL
                        endpoint_url = get_lepton_endpoint_url(endpoint_name)
                        if not endpoint_url:
                            result_queue.put(
                                (
                                    idx,
                                    False,
                                    f"Could not get URL for endpoint {endpoint_name}",
                                    None,
                                    None,
                                )
                            )
                            return

                        # Construct the full endpoint URL
                        task_definition = get_task_from_mapping(
                            task.name, tasks_mapping
                        )
                        task_endpoint_type = task_definition["endpoint_type"]
                        endpoint_path = cfg.deployment.endpoints[task_endpoint_type]
                        full_endpoint_url = f"{endpoint_url.rstrip('/')}{endpoint_path}"

                        print(
                            f"✅ Task {task.name}: Endpoint {endpoint_name} ready at {endpoint_url}"
                        )
                        result_queue.put(
                            (
                                idx,
                                True,
                                None,
                                endpoint_name,
                                endpoint_url,
                                full_endpoint_url,
                            )
                        )

                    except Exception as e:
                        result_queue.put(
                            (
                                idx,
                                False,
                                f"Exception creating endpoint: {e}",
                                None,
                                None,
                            )
                        )

                # Create and start threads for parallel endpoint creation
                result_queue: queue.Queue = queue.Queue()
                threads = []

                for task_info in endpoint_creation_tasks:
                    thread = threading.Thread(
                        target=create_endpoint_worker, args=(task_info, result_queue)
                    )
                    thread.start()
                    threads.append(thread)

                # Wait for all threads to complete and collect results
                for thread in threads:
                    thread.join()

                # Process results
                failed_endpoints = []
                for _ in range(len(endpoint_creation_tasks)):
                    try:
                        result = result_queue.get_nowait()
                        idx = result[0]
                        success = result[1]

                        if success:
                            _, _, _, endpoint_name, endpoint_url, full_endpoint_url = (
                                result
                            )
                            task_endpoints[idx] = {
                                "name": endpoint_name,
                                "url": endpoint_url,
                                "full_url": full_endpoint_url,
                            }
                        else:
                            error_msg = result[2]
                            failed_endpoints.append((idx, error_msg))
                    except queue.Empty:
                        break

                # Check if any endpoints failed
                if failed_endpoints:
                    error_details = "; ".join(
                        [f"Task {idx}: {msg}" for idx, msg in failed_endpoints]
                    )
                    raise RuntimeError(
                        f"Failed to create {len(failed_endpoints)} endpoints: {error_details}"
                    )

                print(
                    f"✅ All {len(cfg.evaluation.tasks)} endpoints created successfully!"
                )

            # ================================================================
            # JOB SUBMISSION (Sequential, as before)
            # ================================================================
            print(f"📝 Submitting {len(cfg.evaluation.tasks)} evaluation jobs...")

            # Submit each evaluation task as a Lepton job
            for idx, task in enumerate(cfg.evaluation.tasks):
                task_definition = get_task_from_mapping(task.name, tasks_mapping)

                # Create job ID and Lepton job name (max 36 chars)
                job_id = generate_job_id(invocation_id, idx)
                # Sanitized task name for RFC 1123 compliance (no underscores, lowercase)
                sanitized_task_name = task.name.replace("_", "-").lower()
                if sanitized_task_name.count(".") > 0:
                    sanitized_task_name = sanitized_task_name.split(".")[-1]
                base_job_name = f"eval-{invocation_id[:6]}-{sanitized_task_name}"
                suffix = str(idx)

                # Ensure job name length is within 36 character limit
                max_base_length = 36 - 1 - len(suffix)  # -1 for the hyphen
                if len(base_job_name) > max_base_length:
                    base_job_name = base_job_name[:max_base_length]
                    logger.info(
                        "Lepton job auto-generated name",
                        task_name=task.name,
                        job_name=f"{base_job_name}-{suffix}",
                    )

                lepton_job_name = f"{base_job_name}-{suffix}"
                logger.info(
                    "Lepton job name (auto-generated)",
                    task_name=task.name,
                    job_name=lepton_job_name,
                )
                job_ids.append(job_id)
                lepton_job_names.append(lepton_job_name)

                # Create task output directory (for result collection)
                task_output_dir = output_dir / task.name
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # Determine evaluation image
                eval_image = task_definition["container"]
                if "container" in task:
                    eval_image = task["container"]

                # Get endpoint info for this task
                endpoint_info = task_endpoints[idx]
                endpoint_name = endpoint_info["name"]
                endpoint_url = endpoint_info["url"]
                full_endpoint_url = endpoint_info["full_url"]

                # Temporarily set the target URL for this specific task
                from omegaconf import OmegaConf

                # Temporarily disable struct mode to allow URL modification
                was_struct = OmegaConf.is_struct(cfg)
                if was_struct:
                    OmegaConf.set_struct(cfg, False)

                # Save original URL
                original_url = getattr(
                    cfg.get("target", {}).get("api_endpoint", {}), "url", None
                )

                try:
                    # Ensure target structure exists and set the task-specific URL
                    if "target" not in cfg:
                        cfg.target = OmegaConf.create({})
                    if "api_endpoint" not in cfg.target:
                        cfg.target.api_endpoint = OmegaConf.create({})

                    cfg.target.api_endpoint.url = full_endpoint_url

                    # Generate command with the correct endpoint URL
                    eval_command_struct = get_eval_factory_command(
                        cfg, task, task_definition
                    )
                    eval_command = eval_command_struct.cmd
                    # Debug string for explainability of some base64-parts of the command
                    eval_command_debug_comment = eval_command_struct.debug

                finally:
                    # Restore original URL and struct mode
                    if original_url is not None:
                        cfg.target.api_endpoint.url = original_url
                    elif (
                        "target" in cfg
                        and "api_endpoint" in cfg.target
                        and "url" in cfg.target.api_endpoint
                    ):
                        del cfg.target.api_endpoint.url

                    if was_struct:
                        OmegaConf.set_struct(cfg, True)

                # Create evaluation launch script
                launch_script = _create_evaluation_launch_script(
                    cfg=cfg,
                    task=task,
                    task_definition=task_definition,
                    endpoint_url=full_endpoint_url,
                    task_name=task.name,
                    invocation_id=invocation_id,
                    eval_command=eval_command,  # Pass the fixed command
                    eval_command_debug_comment=eval_command_debug_comment,
                )

                # Prepare job command to run the launch script
                container_command = [
                    "/bin/bash",
                    "-c",
                    f"echo '{launch_script}' > /tmp/launch_script.sh && chmod +x /tmp/launch_script.sh && bash /tmp/launch_script.sh",
                ]

                # Get evaluation job settings from configuration
                eval_settings = getattr(cfg.execution, "evaluation_tasks", {})
                eval_resource_shape = eval_settings.get("resource_shape", "cpu.small")
                eval_timeout = eval_settings.get("timeout", 3600)
                use_shared_storage = eval_settings.get("use_shared_storage", True)

                # Get environment variables for the job
                task_config = cfg.execution.lepton_platform.tasks
                node_group = task_config.get("node_group", "default")

                # Import DictConfig for both env vars and mounts processing
                from omegaconf import DictConfig

                # Priority: lepton_platform.tasks.env_vars over cfg.execution.env_var_names
                job_env_vars = {}

                # Get env vars from lepton_platform config
                lepton_env_vars = task_config.get("env_vars", {})
                for key, value in lepton_env_vars.items():
                    if isinstance(value, (dict, DictConfig)):
                        # Convert DictConfig to dict to prevent stringification
                        job_env_vars[key] = dict(value)
                    else:
                        job_env_vars[key] = value

                # Get mounts configuration and add invocation ID for isolation
                job_mounts = []
                original_mounts = task_config.get("mounts", [])

                for mount in original_mounts:
                    # Create a copy of the mount with invocation ID added to path
                    mount_dict = (
                        dict(mount) if isinstance(mount, DictConfig) else mount.copy()
                    )

                    # Add invocation ID to the path for evaluation isolation
                    if "path" in mount_dict:
                        original_path = mount_dict["path"]
                        # Add invocation ID subdirectory: /shared/nemo-evaluator-launcher-workspace/abc12345
                        mount_dict["path"] = (
                            f"{original_path.rstrip('/')}/{invocation_id}"
                        )

                    job_mounts.append(mount_dict)

                print(
                    f"   - Storage: {len(job_mounts)} mount(s) with evaluation ID isolation"
                )

                # Get image pull secrets
                image_pull_secrets = task_config.get("image_pull_secrets", [])

                # Submit the evaluation job to Lepton
                print(f"📝 Task {task.name}: Submitting job {lepton_job_name}")
                print(f"   - Endpoint: {endpoint_name if endpoint_name else 'shared'}")
                print(f"   - Resource: {eval_resource_shape}")

                job_success, error_msg = create_lepton_job(
                    job_name=lepton_job_name,
                    container_image=eval_image,
                    command=container_command,
                    resource_shape=eval_resource_shape,
                    env_vars=job_env_vars,
                    mounts=job_mounts,
                    timeout=eval_timeout,
                    node_group=node_group,
                    image_pull_secrets=image_pull_secrets,
                )

                if not job_success:
                    raise RuntimeError(
                        f"Failed to submit Lepton job | Task: {task.name} | Job ID: {job_id} | "
                        f"Lepton job name: {lepton_job_name} | Error: {error_msg}"
                    )

                # Store job metadata in database (with task-specific endpoint info)
                db.write_job(
                    job=JobData(
                        invocation_id=invocation_id,
                        job_id=job_id,
                        timestamp=time.time(),
                        executor="lepton",
                        data={
                            "endpoint_name": endpoint_name,  # Task-specific endpoint (or None for shared)
                            "endpoint_url": endpoint_url,  # Task-specific URL (or shared)
                            "lepton_job_name": lepton_job_name,
                            "output_dir": str(task_output_dir),
                            "task_name": task.name,
                            "status": "submitted",
                        },
                        config=OmegaConf.to_object(cfg),  # type: ignore[arg-type]
                    )
                )

            # Jobs submitted successfully - return immediately (non-blocking)
            print(
                f"\n✅ Successfully submitted {len(lepton_job_names)} evaluation jobs to Lepton"
            )
            print(
                "   Each task running against its own dedicated endpoint for isolation"
            )

            print(f"\n📋 Invocation ID: {invocation_id}")
            print(f"🔍 Check status: nemo-evaluator-launcher status {invocation_id}")
            print(f"📋 Monitor logs: nemo-evaluator-launcher logs {invocation_id}")

            if cfg.deployment.type != "none":
                print(f"🔗 Deployed {len(endpoint_names)} dedicated endpoints:")
                for i, endpoint_name in enumerate(endpoint_names):
                    task_name = cfg.evaluation.tasks[i].name
                    print(f"   - {task_name}: {endpoint_name}")
                print(
                    f"⚠️  Remember to clean up endpoints when done: nemo-evaluator-launcher kill {invocation_id}"
                )
            else:
                print(f"📌 All tasks using shared endpoint: {shared_endpoint_url}")

            print(f"📊 Evaluation results will be saved to: {output_dir}")

            # Note: Jobs will continue running on Lepton infrastructure
            # Status can be checked using nemo-evaluator-launcher status command

            return invocation_id

        except Exception:
            # Clean up any created endpoints on failure
            if cfg.deployment.type != "none" and "endpoint_names" in locals():
                for endpoint_name in endpoint_names:
                    if endpoint_name:
                        print(f"🧹 Cleaning up endpoint: {endpoint_name}")
                        delete_lepton_endpoint(endpoint_name)
            raise

    @staticmethod
    def get_status(id: str) -> List[ExecutionStatus]:
        """Get the status of Lepton evaluation jobs and endpoints.

        Args:
            id: Unique job identifier or invocation identifier.

        Returns:
            List containing the execution status for the job(s) and endpoint(s).
        """
        db = ExecutionDB()

        # If id looks like an invocation_id (8 hex digits, no dot), get all jobs for it
        if "." not in id:
            return _get_statuses_for_invocation_id(id=id, db=db)
        # Otherwise, treat as job_id
        job_data = db.get_job(id)
        if job_data is None:
            return []
        if job_data.executor != "lepton":
            return []

        # Check if this job has a Lepton job associated with it
        lepton_job_name = job_data.data.get("lepton_job_name")
        if lepton_job_name:
            # Get live status from Lepton
            lepton_status = get_lepton_job_status(lepton_job_name)
            if lepton_status:
                job_state = lepton_status.get("state", "Unknown")

                # Map Lepton job states to our execution states
                if job_state in ["Succeeded", "Completed"]:
                    state = ExecutionState.SUCCESS
                elif job_state in ["Running", "Pending", "Starting"]:
                    state = ExecutionState.RUNNING
                elif job_state in ["Failed", "Cancelled"]:
                    state = ExecutionState.FAILED
                else:
                    state = ExecutionState.PENDING

                progress_info = {
                    "type": "evaluation_job",
                    "task_name": job_data.data.get("task_name", "unknown"),
                    "lepton_job_name": lepton_job_name,
                    "lepton_state": job_state,
                    "start_time": lepton_status.get("start_time"),
                    "end_time": lepton_status.get("end_time"),
                    "endpoint_name": job_data.data.get("endpoint_name", "shared"),
                }

                return [ExecutionStatus(id=id, state=state, progress=progress_info)]

        # Fallback to stored status
        job_status = job_data.data.get("status", "unknown")

        if job_status in ["running", "submitted"]:
            state = ExecutionState.RUNNING
        elif job_status in ["succeeded", "completed"]:
            state = ExecutionState.SUCCESS
        elif job_status in ["failed", "cancelled"]:
            state = ExecutionState.FAILED
        else:
            state = ExecutionState.PENDING

        progress_info = {
            "type": "evaluation_job",
            "task_name": job_data.data.get("task_name", "unknown"),
            "status": job_status,
            "lepton_job_name": job_data.data.get("lepton_job_name"),
            "endpoint_name": job_data.data.get("endpoint_name", "shared"),
        }

        return [ExecutionStatus(id=id, state=state, progress=progress_info)]

    @staticmethod
    def kill_job(job_id: str) -> None:
        """Kill Lepton evaluation jobs and clean up endpoints.

        Args:
            job_id: The job ID to kill.

        Raises:
            ValueError: If job is not found or invalid.
            RuntimeError: If job cannot be killed.
        """
        db = ExecutionDB()
        job_data = db.get_job(job_id)
        if job_data is None:
            raise ValueError(f"Job {job_id} not found")

        if job_data.executor != "lepton":
            raise ValueError(
                f"Job {job_id} is not a Lepton job (executor: {job_data.executor})"
            )

        # Cancel the specific Lepton job
        lepton_job_name = job_data.data.get("lepton_job_name")

        if lepton_job_name:
            cancel_success = delete_lepton_job(lepton_job_name)
            if cancel_success:
                print(f"✅ Cancelled Lepton job: {lepton_job_name}")
                # Mark job as killed in database
                job_data.data["status"] = "killed"
                job_data.data["killed_time"] = time.time()
                db.write_job(job_data)
            else:
                # Use common helper to get informative error message based on job status
                status_list = LeptonExecutor.get_status(job_id)
                current_status = status_list[0].state if status_list else None
                error_msg = LeptonExecutor.get_kill_failure_message(
                    job_id, f"lepton_job: {lepton_job_name}", current_status
                )
                raise RuntimeError(error_msg)
        else:
            raise ValueError(f"No Lepton job name found for job {job_id}")

        print(f"🛑 Killed Lepton job {job_id}")

        # For individual jobs, also clean up the dedicated endpoint for this task
        # Check if this was the last job using this specific endpoint
        endpoint_name = job_data.data.get("endpoint_name")
        if endpoint_name:
            # Check if any other jobs are still using this endpoint
            jobs = db.get_jobs(job_data.invocation_id)
            other_jobs_using_endpoint = [
                j
                for j in jobs.values()
                if (
                    j.data.get("endpoint_name") == endpoint_name
                    and j.data.get("status")
                    not in ["killed", "failed", "succeeded", "cancelled"]
                    and j.job_id != job_id
                )
            ]

            if not other_jobs_using_endpoint:
                print(
                    f"🧹 No other jobs using endpoint {endpoint_name}, cleaning up..."
                )
                success = delete_lepton_endpoint(endpoint_name)
                if success:
                    print(f"✅ Cleaned up endpoint: {endpoint_name}")
                else:
                    print(f"⚠️  Failed to cleanup endpoint: {endpoint_name}")
            else:
                print(
                    f"📌 Keeping endpoint {endpoint_name} (still used by {len(other_jobs_using_endpoint)} other jobs)"
                )
        else:
            print("📌 No dedicated endpoint to clean up for this job")


def _create_evaluation_launch_script(
    cfg: DictConfig,
    task: DictConfig,
    task_definition: dict,
    endpoint_url: str,
    task_name: str,
    invocation_id: str,
    eval_command: str,
    eval_command_debug_comment: str,
) -> str:
    """Create bash script for running evaluation in Lepton job container.

    Based on the proven approach from the old implementation.

    Args:
        cfg: The configuration object.
        task: The evaluation task configuration.
        task_definition: Task definition from mapping.
        endpoint_url: URL of the deployed Lepton endpoint.
        task_name: Name of the evaluation task.
        invocation_id: Unique invocation identifier.
        eval_command: The evaluation command with correct endpoint URL.
        eval_command_debug_comment: The debug comment for placing into the script and easy debug

    Returns:
        String containing the bash launch script.
    """
    # Use the provided eval_command (already has correct endpoint URL)

    # Construct output directory path
    output_dir = f"{cfg.execution.output_dir}/{task_name}"

    # Replace the output directory in the evaluation command
    eval_command_modified = eval_command.replace(
        "--output_dir /results", f"--output_dir {output_dir}"
    )

    # Create the launch script (based on old implementation)
    script = f"""#!/bin/bash
set -e

# Create output directory structure
mkdir -p {output_dir}/artifacts
mkdir -p {output_dir}/logs

# Create stage files for status tracking
echo "started" > {output_dir}/logs/stage.pre-start
echo "running" > {output_dir}/logs/stage.running

# Log evaluation details
echo "Starting evaluation for task: {task_name}"
echo "Invocation ID: {invocation_id}"
echo "Endpoint URL: {endpoint_url}"
echo "Command: {eval_command_modified}"

{eval_command_debug_comment}

# Execute the evaluation with proper error handling
set +e
{eval_command_modified}
exit_code=$?

# Set proper permissions
chmod 777 -R {output_dir} 2>/dev/null || true

# Record completion status
echo "exit_code: $exit_code" > {output_dir}/logs/stage.exit

if [ "$exit_code" -ne 0 ]; then
    echo "Evaluation failed with exit code $exit_code" >&2
    exit "$exit_code"
fi

echo "Evaluation completed successfully"
exit 0
"""

    return script


def _dry_run_lepton(
    cfg: DictConfig, tasks_mapping: dict, invocation_id: str | None = None
) -> None:
    print("DRY RUN: Lepton job configurations prepared")
    try:
        # validate tasks
        for task in cfg.evaluation.tasks:
            get_task_from_mapping(task.name, tasks_mapping)

        # nice-to-have checks (existing endpoint URL or endpoints mapping)
        if getattr(cfg.deployment, "type", None) == "none":
            tgt = getattr(cfg, "target", {})
            api = (
                tgt.get("api_endpoint")
                if isinstance(tgt, dict)
                else getattr(tgt, "api_endpoint", None)
            ) or {}
            url = api.get("url") if isinstance(api, dict) else getattr(api, "url", None)
            if not url or not str(url).strip():
                raise ValueError(
                    "target.api_endpoint.url must be set when deployment.type == 'none'"
                )
        else:
            endpoints_cfg = getattr(cfg.deployment, "endpoints", {}) or {}
            for task in cfg.evaluation.tasks:
                td = get_task_from_mapping(task.name, tasks_mapping)
                etype = td.get("endpoint_type")
                if etype not in endpoints_cfg:
                    raise ValueError(
                        f"deployment.endpoints missing path for endpoint_type '{etype}' (task '{task.name}')"
                    )
                path = endpoints_cfg.get(etype)
                if not isinstance(path, str) or not path.startswith("/"):
                    raise ValueError(
                        f"deployment.endpoints['{etype}'] must be a non-empty path starting with '/'"
                    )

        # lepton env var presence (reference-level)
        tasks_cfg = getattr(cfg.execution, "lepton_platform", {}).get("tasks", {}) or {}
        lepton_env_vars = tasks_cfg.get("env_vars", {}) or {}
        api_key_name = getattr(
            getattr(cfg, "target", {}).get("api_endpoint", {}), "api_key_name", None
        )
        for task in cfg.evaluation.tasks:
            td = get_task_from_mapping(task.name, tasks_mapping)
            required = td.get("required_env_vars", []) or []
            for var in required:
                if var == "API_KEY":
                    if not (("API_KEY" in lepton_env_vars) or bool(api_key_name)):
                        raise ValueError(
                            f"Task '{task.name}' requires API_KEY: set execution.lepton_platform.tasks.env_vars.API_KEY "
                            "or target.api_endpoint.api_key_name"
                        )
                else:
                    if var not in lepton_env_vars:
                        raise ValueError(
                            f"Task '{task.name}' requires {var}: set it under execution.lepton_platform.tasks.env_vars"
                        )

        # success (use realized output directory if invocation_id is available)
        preview_output_dir = (
            Path(cfg.execution.output_dir).absolute() / invocation_id
            if invocation_id
            else Path(cfg.execution.output_dir).absolute() / "<invocation_id>"
        )
        print(f"   - Tasks: {len(cfg.evaluation.tasks)}")
        for idx, task in enumerate(cfg.evaluation.tasks):
            print(f"   - Task {idx}: {task.name}")
        print(f"   - Output directory: {preview_output_dir}")
        print("\nTo run evaluation, execute run command without --dry-run")
    except Exception as e:
        print(f"❌ Configuration invalid: {e}")
        logger.error("Lepton dry-run validation failed", error=str(e))
        return


def _get_statuses_for_invocation_id(id: str, db: ExecutionDB) -> List[ExecutionStatus]:
    """Helper method that returns statuses if id is the invocation id"""
    jobs = db.get_jobs(id)
    statuses: List[ExecutionStatus] = []

    # Get status for all endpoints (each task may have its own)
    endpoint_names = set()
    for job_data in jobs.values():
        endpoint_name = job_data.data.get("endpoint_name")
        if endpoint_name:
            endpoint_names.add(endpoint_name)

    # Show status for each unique endpoint
    for endpoint_name in endpoint_names:
        endpoint_status = get_lepton_endpoint_status(endpoint_name)
        if not endpoint_status:
            logger.warning(
                "Could not get Lepton endpoint statuses",
                endpoint_name=endpoint_name,
            )
            return statuses

        endpoint_state = endpoint_status.get("state", "Unknown")
        if endpoint_state == "Ready":
            state = ExecutionState.SUCCESS
        elif endpoint_state in ["Starting", "Pending"]:
            state = ExecutionState.RUNNING
        else:
            state = ExecutionState.FAILED

        # Find which task(s) use this endpoint
        using_tasks = [
            job_data.data.get("task_name", "unknown")
            for job_data in jobs.values()
            if job_data.data.get("endpoint_name") == endpoint_name
        ]

        statuses.append(
            ExecutionStatus(
                id=f"{id}-endpoint-{endpoint_name}",
                state=state,
                progress={
                    "type": "endpoint",
                    "name": endpoint_name,
                    "state": endpoint_state,
                    "url": endpoint_status.get("endpoint", {}).get("external_endpoint"),
                    "tasks": using_tasks,
                },
            )
        )

    # If no dedicated endpoints, note that shared endpoint is being used
    if not endpoint_names:
        statuses.append(
            ExecutionStatus(
                id=f"{id}-endpoint-shared",
                state=ExecutionState.SUCCESS,
                progress={
                    "type": "endpoint",
                    "name": "shared",
                    "state": "Using existing endpoint",
                    "url": "external",
                    "tasks": [
                        job_data.data.get("task_name", "unknown")
                        for job_data in jobs.values()
                    ],
                },
            )
        )

    # Get individual job statuses
    for job_id, job_data in jobs.items():
        statuses.extend(LeptonExecutor.get_status(job_id))
    return statuses
