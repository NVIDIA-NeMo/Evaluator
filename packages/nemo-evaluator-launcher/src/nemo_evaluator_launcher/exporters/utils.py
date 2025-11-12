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
"""Shared utilities for metrics and configuration handling."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import (
    get_task_from_mapping,
    load_tasks_mapping,
)

# =============================================================================
# ARTIFACTS
# =============================================================================

# Artifacts to be logged by default
REQUIRED_ARTIFACTS = ["results.yml", "eval_factory_metrics.json"]
OPTIONAL_ARTIFACTS = ["omni-info.json"]


def get_relevant_artifacts() -> List[str]:
    """Get relevant artifacts (required + optional)."""
    return REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS


def validate_artifacts(artifacts_dir: Path) -> Dict[str, Any]:
    """Check which artifacts are available."""
    if not artifacts_dir or not artifacts_dir.exists():
        return {
            "can_export": False,
            "missing_required": REQUIRED_ARTIFACTS.copy(),
            "missing_optional": OPTIONAL_ARTIFACTS.copy(),
            "message": "Artifacts directory not found",
        }

    missing_required = [
        f for f in REQUIRED_ARTIFACTS if not (artifacts_dir / f).exists()
    ]
    missing_optional = [
        f for f in OPTIONAL_ARTIFACTS if not (artifacts_dir / f).exists()
    ]
    can_export = len(missing_required) == 0

    message_parts = []
    if missing_required:
        message_parts.append(f"Missing required: {', '.join(missing_required)}")
    if missing_optional:
        message_parts.append(f"Missing optional: {', '.join(missing_optional)}")

    return {
        "can_export": can_export,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "message": (
            ". ".join(message_parts) if message_parts else "All artifacts available"
        ),
    }


def get_available_artifacts(artifacts_dir: Path) -> List[str]:
    """Get list of artifacts available in artifacts directory."""
    if not artifacts_dir or not artifacts_dir.exists():
        return []
    return [
        filename
        for filename in get_relevant_artifacts()
        if (artifacts_dir / filename).exists()
    ]


# =============================================================================
# METRICS EXTRACTION
# =============================================================================


class MetricConflictError(Exception):
    """Raised when attempting to set the same metric key with a different value."""


def extract_accuracy_metrics(
    job_data: JobData, get_job_paths_func: Callable, log_metrics: List[str] = None
) -> Dict[str, float]:
    """Extract accuracy metrics from job results."""
    try:
        paths = get_job_paths_func(job_data)
        artifacts_dir = _get_artifacts_dir(paths)

        if not artifacts_dir or not artifacts_dir.exists():
            logger.warning(f"Artifacts directory not found for job {job_data.job_id}")
            return {}

        # Prefer results.yml, but also merge JSON metrics to avoid missing values
        metrics: Dict[str, float] = {}
        results_yml = artifacts_dir / "results.yml"
        if results_yml.exists():
            yml_metrics = _extract_from_results_yml(results_yml)
            if yml_metrics:
                metrics.update(yml_metrics)

        # Merge in JSON metrics (handles tasks that only emit JSON or extra fields)
        json_metrics = _extract_from_json_files(artifacts_dir)
        for k, v in json_metrics.items():
            metrics.setdefault(k, v)

        # Filter metrics if specified
        if log_metrics:
            filtered_metrics = {}
            for metric_name, metric_value in metrics.items():
                if any(filter_key in metric_name.lower() for filter_key in log_metrics):
                    filtered_metrics[metric_name] = metric_value
            return filtered_metrics

        return metrics

    except Exception as e:
        logger.error(f"Failed to extract metrics for job {job_data.job_id}: {e}")
        return {}


# =============================================================================
# CONFIG EXTRACTION
# =============================================================================


def extract_exporter_config(
    job_data: JobData, exporter_name: str, constructor_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Extract and merge exporter configuration from multiple sources."""
    config = {}

    # root-level `export.<exporter-name>`
    if job_data.config:
        export_block = (job_data.config or {}).get("export", {})
        yaml_config = (export_block or {}).get(exporter_name, {})
        if yaml_config:
            config.update(yaml_config)

    # From webhook metadata (if triggered by webhook)
    if "webhook_metadata" in job_data.data:
        webhook_data = job_data.data["webhook_metadata"]
        webhook_config = {
            "triggered_by_webhook": True,
            "webhook_source": webhook_data.get("webhook_source", "unknown"),
            "source_artifact": f"{webhook_data.get('artifact_name', 'unknown')}:{webhook_data.get('artifact_version', 'unknown')}",
            "config_source": webhook_data.get("config_file", "unknown"),
        }
        if exporter_name == "wandb" and webhook_data.get("webhook_source") == "wandb":
            wandb_specific = {
                "entity": webhook_data.get("entity"),
                "project": webhook_data.get("project"),
                "run_id": webhook_data.get("run_id"),
            }
            webhook_config.update({k: v for k, v in wandb_specific.items() if v})
        config.update(webhook_config)

    # allows CLI overrides
    if constructor_config:
        config.update(constructor_config)

    return config


# =============================================================================
# JOB DATA EXTRACTION
# =============================================================================


def get_task_name(job_data: JobData) -> str:
    """Get task name from job data."""
    if "." in job_data.job_id:
        try:
            idx = int(job_data.job_id.split(".")[-1])
            return job_data.config["evaluation"]["tasks"][idx]["name"]
        except Exception:
            return f"job_{job_data.job_id}"
    return "all_tasks"


def get_model_name(job_data: JobData, config: Dict[str, Any] = None) -> str:
    """Extract model name from config or job data."""
    if config and "model_name" in config:
        return config["model_name"]

    job_config = job_data.config or {}
    model_sources = [
        job_config.get("target", {}).get("api_endpoint", {}).get("model_id"),
        job_config.get("deployment", {}).get("served_model_name"),
        job_data.data.get("served_model_name"),
        job_data.data.get("model_name"),
        job_data.data.get("model_id"),
    ]

    for source in model_sources:
        if source:
            return str(source)

    return f"unknown_model_{job_data.job_id}"


def get_pipeline_id(job_data: JobData) -> str:
    """Get pipeline ID for GitLab jobs."""
    return job_data.data.get("pipeline_id") if job_data.executor == "gitlab" else None


def get_benchmark_info(job_data: JobData) -> Dict[str, str]:
    """Get harness and benchmark info from mapping."""
    try:
        task_name = get_task_name(job_data)
        if task_name in ["all_tasks", f"job_{job_data.job_id}"]:
            return {"harness": "unknown", "benchmark": task_name}

        # Use mapping to get harness info
        mapping = load_tasks_mapping()
        task_definition = get_task_from_mapping(task_name, mapping)
        harness = task_definition.get("harness", "unknown")

        # Extract benchmark name (remove harness prefix)
        if "." in task_name:
            benchmark = ".".join(task_name.split(".")[1:])
        else:
            benchmark = task_name

        return {"harness": harness, "benchmark": benchmark}

    except Exception as e:
        logger.warning(f"Failed to get benchmark info: {e}")
        return {"harness": "unknown", "benchmark": get_task_name(job_data)}


def get_container_from_mapping(job_data: JobData) -> str:
    """Get container from mapping."""
    try:
        task_name = get_task_name(job_data)
        if task_name in ["all_tasks", f"job_{job_data.job_id}"]:
            return None

        mapping = load_tasks_mapping()
        task_definition = get_task_from_mapping(task_name, mapping)
        return task_definition.get("container")

    except Exception as e:
        logger.warning(f"Failed to get container from mapping: {e}")
        return None


def get_artifact_root(job_data: JobData) -> str:
    """Get artifact root from job data."""
    bench = get_benchmark_info(job_data)
    h = bench.get("harness", "unknown")
    b = bench.get("benchmark", get_task_name(job_data))
    return f"{h}.{b}"


# =============================================================================
# GITLAB DOWNLOAD
# =============================================================================


def download_gitlab_artifacts(
    paths: Dict[str, Any], export_dir: Path, extract_specific: bool = False
) -> Dict[str, Path]:
    """Download artifacts from GitLab API.

    Args:
        paths: Dictionary containing pipeline_id and project_id
        export_dir: Local directory to save artifacts
        extract_specific: If True, extract individual files; if False, keep as ZIP files

    Returns:
        Dictionary mapping artifact names to local file paths
    """
    raise NotImplementedError("Downloading from gitlab is not implemented")


# =============================================================================
# SSH UTILS
# =============================================================================


# SSH connections directory
CONNECTIONS_DIR = Path.home() / ".nemo-evaluator" / "connections"


def ssh_setup_masters(jobs: Dict[str, JobData]) -> Dict[Tuple[str, str], str]:
    """Start SSH master connections for remote jobs, returns control_paths."""
    remote_pairs: set[tuple[str, str]] = set()
    for jd in jobs.values():
        try:
            # Preferred: explicit 'paths' from job data
            p = (jd.data or {}).get("paths") or {}
            if (
                p.get("storage_type") == "remote_ssh"
                and p.get("username")
                and p.get("hostname")
            ):
                remote_pairs.add((p["username"], p["hostname"]))
                continue
            # Fallback: common slurm fields (works with BaseExporter.get_job_paths)
            d = jd.data or {}
            if jd.executor == "slurm" and d.get("username") and d.get("hostname"):
                remote_pairs.add((d["username"], d["hostname"]))
        except Exception:
            pass

    if not remote_pairs:
        return {}

    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    control_paths: Dict[Tuple[str, str], str] = {}
    for username, hostname in remote_pairs:
        socket_path = CONNECTIONS_DIR / f"{username}_{hostname}.sock"
        try:
            cmd = [
                "ssh",
                "-N",
                "-f",
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=60",
                "-o",
                f"ControlPath={socket_path}",
                f"{username}@{hostname}",
            ]
            subprocess.run(cmd, check=False, capture_output=True)
            control_paths[(username, hostname)] = str(socket_path)
        except Exception as e:
            logger.warning(f"Failed to start SSH master for {username}@{hostname}: {e}")
    return control_paths


def ssh_cleanup_masters(control_paths: Dict[Tuple[str, str], str]) -> None:
    """Clean up SSH master connections from control_paths."""
    for (username, hostname), socket_path in (control_paths or {}).items():
        try:
            cmd = [
                "ssh",
                "-O",
                "exit",
                "-o",
                f"ControlPath={socket_path}",
                f"{username}@{hostname}",
            ]
            subprocess.run(cmd, check=False, capture_output=True)
        except Exception as e:
            logger.warning(f"Failed to stop SSH master for {username}@{hostname}: {e}")

        # Clean up
        try:
            Path(socket_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up file: {e}")


def ssh_download_artifacts(
    paths: Dict[str, Any],
    export_dir: Path,
    config: Dict[str, Any] | None = None,
    control_paths: Dict[Tuple[str, str], str] | None = None,
) -> List[str]:
    """Download artifacts/logs via SSH with optional connection reuse."""
    exported_files: List[str] = []
    copy_logs = bool((config or {}).get("copy_logs", False))
    copy_artifacts = bool((config or {}).get("copy_artifacts", True))
    only_required = bool((config or {}).get("only_required", True))

    control_path = None
    if control_paths:
        control_path = control_paths.get((paths["username"], paths["hostname"]))
    ssh_opts = ["-o", f"ControlPath={control_path}"] if control_path else []

    def scp_file(remote_path: str, local_path: Path) -> bool:
        cmd = (
            ["scp"]
            + ssh_opts
            + [
                f"{paths['username']}@{paths['hostname']}:{remote_path}",
                str(local_path),
            ]
        )
        return subprocess.run(cmd, capture_output=True).returncode == 0

    export_dir.mkdir(parents=True, exist_ok=True)

    # Artifacts
    if copy_artifacts:
        art_dir = export_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)

        if only_required:
            for artifact in get_relevant_artifacts():
                remote_file = f"{paths['remote_path']}/artifacts/{artifact}"
                local_file = art_dir / artifact
                local_file.parent.mkdir(parents=True, exist_ok=True)
                if scp_file(remote_file, local_file):
                    exported_files.append(str(local_file))
        else:
            # Copy known files individually to avoid subfolders and satisfy tests
            for artifact in get_available_artifacts(paths.get("artifacts_dir", Path())):
                remote_file = f"{paths['remote_path']}/artifacts/{artifact}"
                local_file = art_dir / artifact
                if scp_file(remote_file, local_file):
                    exported_files.append(str(local_file))

    # Logs (top-level only)
    if copy_logs:
        local_logs = export_dir / "logs"
        remote_logs = f"{paths['remote_path']}/logs"
        cmd = (
            ["scp", "-r"]
            + ssh_opts
            + [
                f"{paths['username']}@{paths['hostname']}:{remote_logs}/.",
                str(local_logs),
            ]
        )
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            for p in local_logs.iterdir():
                if p.is_dir():
                    import shutil

                    shutil.rmtree(p, ignore_errors=True)
            exported_files.extend([str(f) for f in local_logs.glob("*") if f.is_file()])

    return exported_files


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================


def _get_artifacts_dir(paths: Dict[str, Any]) -> Path:
    """Get artifacts directory from paths."""
    storage_type = paths.get("storage_type")

    # For SSH-based remote access, artifacts aren't available locally yet
    if storage_type == "remote_ssh":
        return None

    # For all local access (local_filesystem, remote_local, gitlab_ci_local)
    # return the artifacts_dir from paths
    return paths.get("artifacts_dir")


def _extract_metrics_from_results(results: dict) -> Dict[str, float]:
    """Extract metrics from a 'results' dict (with optional 'groups'/'tasks')."""
    metrics: Dict[str, float] = {}
    for section in ["groups", "tasks"]:
        section_data = results.get(section)
        if isinstance(section_data, dict):
            for task_name, task_data in section_data.items():
                task_metrics = _extract_task_metrics(task_name, task_data)
                _safe_update_metrics(
                    target=metrics,
                    source=task_metrics,
                    context=f" while extracting results for task '{task_name}'",
                )
    return metrics


def _extract_from_results_yml(results_yml: Path) -> Dict[str, float]:
    """Extract metrics from results.yml file."""
    try:
        with open(results_yml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "results" not in data:
            return {}
        return _extract_metrics_from_results(data.get("results"))
    except Exception as e:
        logger.warning(f"Failed to parse results.yml: {e}")
        return {}


def _extract_from_json_files(artifacts_dir: Path) -> Dict[str, float]:
    """Extract metrics from individual JSON result files."""
    metrics = {}

    for json_file in artifacts_dir.glob("*.json"):
        if json_file.name in get_relevant_artifacts():
            continue  # Skip known artifact files, focus on task result files

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "score" in data:
                task_name = json_file.stem
                metrics[f"{task_name}_score"] = float(data["score"])

        except Exception as e:
            logger.warning(f"Failed to parse {json_file}: {e}")

    return metrics


def _extract_task_metrics(task_name: str, task_data: dict) -> Dict[str, float]:
    """Extract metrics from a task's metrics data."""
    extracted = {}

    metrics_data = task_data.get("metrics", {})
    if "groups" in task_data:
        for group_name, group_data in task_data["groups"].items():
            group_extracted = _extract_task_metrics(
                f"{task_name}_{group_name}", group_data
            )
            _safe_update_metrics(
                target=extracted,
                source=group_extracted,
                context=f" in task '{task_name}'",
            )

    for metric_name, metric_data in metrics_data.items():
        try:
            for score_type, score_data in metric_data["scores"].items():
                if score_type != metric_name:
                    key = f"{task_name}_{metric_name}_{score_type}"
                else:
                    key = f"{task_name}_{metric_name}"
                _safe_set_metric(
                    container=extracted,
                    key=key,
                    new_value=score_data["value"],
                    context=f" in task '{task_name}'",
                )
                for stat_name, stat_value in metric_data.get("stats", {}).items():
                    stats_key = f"{key}_{stat_name}"
                    _safe_set_metric(
                        container=extracted,
                        key=stats_key,
                        new_value=stat_value,
                        context=f" in task '{task_name}'",
                    )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to extract metric {metric_name} for task {task_name}: {e}"
            )

    return extracted


def _safe_set_metric(
    container: Dict[str, float], key: str, new_value: float, context: str
) -> None:
    """Set a metric into container; raise with details if key exists."""
    if key in container:
        # Allow exact matches; warn and keep existing
        if container[key] == float(new_value):
            logger.warning(
                f"Metric rewrite{context}: '{key}' has identical value; keeping existing. value={container[key]}"
            )
            return
        # Different value is an error we want to surface distinctly
        raise MetricConflictError(
            f"Metric key collision{context}: '{key}' already set. existing={container[key]} new={new_value}"
        )
    container[key] = float(new_value)


def _safe_update_metrics(
    target: Dict[str, float], source: Dict[str, float], context: str
) -> None:
    """Update target from source safely, raising on collisions with detailed values."""
    for k, v in source.items():
        _safe_set_metric(target, k, v, context)


# =============================================================================
# MLFLOW FUNCTIONS
# =============================================================================

# MLflow constants
_MLFLOW_KEY_MAX = 250
_MLFLOW_PARAM_VAL_MAX = 250
_MLFLOW_TAG_VAL_MAX = 5000

_INVALID_KEY_CHARS = re.compile(r"[^/\w.\- ]")
_MULTI_UNDERSCORE = re.compile(r"_+")


def mlflow_sanitize(s: Any, kind: str = "key") -> str:
    """
    Sanitize strings for MLflow logging.

    kind:
      - "key", "metric", "tag_key", "param_key": apply key rules
      - "tag_value": apply tag value rules
      - "param_value": apply param value rules
    """
    s = "" if s is None else str(s)

    if kind in ("key", "metric", "tag_key", "param_key"):
        # common replacements
        s = s.replace("pass@", "pass_at_")
        # drop disallowed chars, collapse underscores, trim
        s = _INVALID_KEY_CHARS.sub("_", s)
        s = _MULTI_UNDERSCORE.sub("_", s).strip()
        return s[:_MLFLOW_KEY_MAX] or "key"

    # values: normalize whitespace, enforce length
    s = s.replace("\n", " ").replace("\r", " ").strip()
    max_len = _MLFLOW_TAG_VAL_MAX if kind == "tag_value" else _MLFLOW_PARAM_VAL_MAX
    return s[:max_len]
