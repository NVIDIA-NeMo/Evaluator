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
"""Lepton job helper functions for nemo-evaluator-launcher.

Handles Lepton job creation, management, and monitoring.
"""

import json
import subprocess
import time
from typing import Any, List, Union

from omegaconf import DictConfig

from nemo_evaluator_launcher.common.logging_utils import logger

# =============================================================================
# LEPTON JOB MANAGEMENT FUNCTIONS
# =============================================================================


def create_lepton_job(
    job_name: str,
    container_image: str,
    command: List[str],
    resource_shape: str = "cpu.small",
    env_vars: dict[Any, Any] | None = None,
    mounts: List[Union[dict[Any, Any], DictConfig]] | None = None,
    timeout: int = 3600,
    node_group: str | None = None,
    image_pull_secrets: List[str] | None = None,
) -> tuple[bool, str]:
    """Create a Lepton batch job for evaluation using the API client.

    Args:
        job_name: Name for the job.
        container_image: Docker image to use for the job.
        command: Command to run in the container.
        resource_shape: Resource requirements (cpu.small, gpu.a10, etc).
        env_vars: Environment variables for the job.
        mounts: Storage mounts for the job.
        timeout: Job timeout in seconds.
        node_group: Node group for job placement.
        image_pull_secrets: Secrets for pulling container images.

    Returns:
        Tuple of (success: bool, error_message: str).
    """
    return _create_lepton_job_api(
        job_name,
        container_image,
        command,
        resource_shape,
        env_vars,
        mounts,
        timeout,
        node_group,
        image_pull_secrets,
    )


def _create_lepton_job_api(
    job_name: str,
    container_image: str,
    command: List[str],
    resource_shape: str,
    env_vars: dict[Any, Any] | None = None,
    mounts: List[Union[dict[Any, Any], DictConfig]] | None = None,
    timeout: int = 3600,
    node_group: str | None = None,
    image_pull_secrets: List[str] | None = None,
) -> tuple[bool, str]:
    """Create Lepton job using API client (preferred method)."""
    try:
        # Import leptonai dependencies locally
        from leptonai.api.v1.types.affinity import LeptonResourceAffinity
        from leptonai.api.v1.types.common import LeptonVisibility, Metadata
        from leptonai.api.v1.types.deployment import (
            EnvValue,
            EnvVar,
            LeptonContainer,
            Mount,
        )
        from leptonai.api.v1.types.job import LeptonJob, LeptonJobUserSpec
        from leptonai.api.v2.client import APIClient

        client = APIClient()

        # Prepare environment variables (support both direct values and secret references)
        lepton_env_vars = []
        if env_vars:
            for key, value in env_vars.items():
                # Handle both regular dicts and OmegaConf objects
                if isinstance(value, (dict, DictConfig)) and "value_from" in value:
                    # Secret reference: {value_from: {secret_name_ref: "secret_name"}}
                    # Convert OmegaConf to dict if needed
                    value_dict = dict(value) if isinstance(value, DictConfig) else value
                    env_var = EnvVar(
                        name=key,
                        value_from=EnvValue(
                            secret_name_ref=value_dict["value_from"]["secret_name_ref"]
                        ),
                    )
                    lepton_env_vars.append(env_var)
                else:
                    # Direct value
                    lepton_env_vars.append(EnvVar(name=key, value=str(value)))

        # Prepare mounts
        lepton_mounts = []
        if mounts:
            for mount in mounts:
                # Handle both regular dicts and OmegaConf DictConfig objects
                if isinstance(mount, (dict, DictConfig)):
                    try:
                        # Convert DictConfig to regular dict if needed
                        mount_dict: dict[Any, Any] = dict(mount)
                        lepton_mount = Mount(**mount_dict)
                        lepton_mounts.append(lepton_mount)
                    except Exception as e:
                        return False, f"Invalid mount configuration: {e}"
                else:
                    return (  # type: ignore[unreachable]
                        False,
                        f"Mount must be a dictionary or DictConfig, got {type(mount)}",
                    )

        # Handle node group affinity
        affinity = None
        if node_group:
            # Get node groups to find the correct one
            node_groups = client.nodegroup.list_all()
            node_group_map = {ng.metadata.name: ng for ng in node_groups}
            if node_group in node_group_map:
                node_group_obj = node_group_map[node_group]
                valid_node_ids = [
                    node.metadata.id_
                    for node in client.nodegroup.list_nodes(node_group_obj)
                ]
                affinity = LeptonResourceAffinity(
                    allowed_dedicated_node_groups=[node_group_obj.metadata.id_],
                    allowed_nodes_in_node_group=valid_node_ids,
                )

        # Create job specification
        job_spec = LeptonJobUserSpec(
            resource_shape=resource_shape,
            affinity=affinity,
            container=LeptonContainer(image=container_image, command=command),
            envs=lepton_env_vars,
            mounts=lepton_mounts,
            image_pull_secrets=image_pull_secrets or [],
            shared_memory_size=1024,  # 1GB - appropriate for CPU tasks
            completions=1,
            parallelism=1,
            intra_job_communication=False,
        )

        # Create the job
        job = LeptonJob(
            metadata=Metadata(name=job_name, visibility=LeptonVisibility.PRIVATE),
            spec=job_spec,
        )

        response = client.job.create(job)
        logger.info(
            "Successfully created Lepton job",
            job_name=job_name,
            id=response.metadata.id_,
        )
        return True, ""

    except Exception as e:
        error_msg = f"Error creating Lepton job via API: {e}"
        logger.error("Error creating Lepton job via API", err=str(e))
        return False, error_msg


def get_lepton_job_status(job_name_or_id: str) -> dict[Any, Any] | None:
    """Get the status of a Lepton job using the API client.

    Args:
        job_name_or_id: Name or ID of the job.

    Returns:
        Job status dictionary if successful, None otherwise.
    """
    return _get_lepton_job_status_api(job_name_or_id)


def _get_lepton_job_status_api(job_name_or_id: str) -> dict[Any, Any] | None:
    """Get job status using API client (preferred method)."""
    try:
        # Import leptonai dependencies locally
        from leptonai.api.v2.client import APIClient

        client = APIClient()

        # Try to get job by ID first, then by name
        job = None
        try:
            # If it looks like an ID, try that first
            if len(job_name_or_id) > 20:  # Job IDs are longer
                job = client.job.get(job_name_or_id)
        except Exception:
            pass

        # If not found by ID, try by name
        if not job:
            # List all jobs and find by name
            all_jobs = client.job.list_all()
            for j in all_jobs:
                if j.metadata.name == job_name_or_id:
                    job = j
                    break

        if not job:
            logger.warn(
                "Not found when getting job status via API",
                job_name_or_id=job_name_or_id,
            )
            return None

        # Extract status information
        if job.status:
            # Handle enum states
            state_str = str(job.status.state)
            if "." in state_str:
                state = state_str.split(".")[
                    -1
                ]  # Extract "Completed" from "LeptonJobState.Completed"
            else:
                state = state_str

            return {
                "id": job.metadata.id_,
                "name": job.metadata.name,
                "state": state,
                "start_time": getattr(job.status, "start_time", None),
                "end_time": getattr(job.status, "end_time", None),
                "ready": getattr(job.status, "ready", 0),
                "active": getattr(job.status, "active", 0),
                "succeeded": getattr(job.status, "succeeded", 0),
                "failed": getattr(job.status, "failed", 0),
            }
        else:
            return {
                "id": job.metadata.id_,
                "name": job.metadata.name,
                "state": "Unknown",
            }

    except Exception as e:
        logger.error("Error getting job status via API", err=str(e))
        return None


def _get_lepton_job_status_cli(job_name: str) -> dict[Any, Any] | None:
    """Get job status using CLI (fallback method)."""
    try:
        result = subprocess.run(
            ["lep", "job", "get", "--name", job_name],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            job_info: dict[Any, Any] = json.loads(result.stdout)
            # Return the job info which contains status information
            return job_info
        else:
            return None
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        return None


def list_lepton_jobs(prefix: str | None = None) -> List[dict[Any, Any]]:
    """List Lepton jobs, optionally filtered by name prefix.

    Args:
        prefix: Optional prefix to filter job names.

    Returns:
        List of job information dictionaries.
    """
    try:
        result = subprocess.run(
            ["lep", "job", "list"], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            jobs_info: dict[Any, Any] = json.loads(result.stdout)
            jobs: List[dict[Any, Any]] = jobs_info.get("jobs", [])

            if prefix:
                jobs = [job for job in jobs if job.get("name", "").startswith(prefix)]

            return jobs
        else:
            return []
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ):
        return []


def delete_lepton_job(job_name: str) -> bool:
    """Delete/cancel a Lepton job.

    Args:
        job_name: Name of the job to delete.

    Returns:
        True if deletion succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            ["lep", "job", "remove", "--name", job_name],
            capture_output=True,
            text=True,
            timeout=60,
        )

        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def wait_for_lepton_jobs_completion(job_names: List[str], timeout: int = 3600) -> dict:
    """Wait for multiple Lepton jobs to complete.

    Args:
        job_names: List of job names to monitor.
        timeout: Maximum time to wait in seconds.

    Returns:
        Dictionary mapping job names to their final status.
    """

    start_time = time.time()
    job_statuses = {}
    completed_jobs: set[str] = set()

    print(f"⏳ Monitoring {len(job_names)} evaluation jobs...")

    while len(completed_jobs) < len(job_names) and (time.time() - start_time) < timeout:
        for job_name in job_names:
            if job_name in completed_jobs:
                continue

            status = get_lepton_job_status(job_name)
            if status:
                job_state = status.get("state", "Unknown")
                job_statuses[job_name] = status

                if job_state in ["Succeeded", "Failed", "Cancelled"]:
                    completed_jobs.add(job_name)
                    if job_state == "Succeeded":
                        print(f"✅ Job {job_name}: {job_state}")
                    else:
                        print(f"❌ Job {job_name}: {job_state}")
                else:
                    print(f"⏳ Job {job_name}: {job_state}")

        if len(completed_jobs) < len(job_names):
            time.sleep(10)  # Check every 10 seconds

    # Final status check
    for job_name in job_names:
        if job_name not in completed_jobs:
            status = get_lepton_job_status(job_name)
            if status:
                job_statuses[job_name] = status
                print(
                    f"⏰ Job {job_name}: Timeout (still {status.get('state', 'Unknown')})"
                )

    return job_statuses
