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

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
REQUIRED_ARTIFACTS = [
    "results.yml",
    "eval_factory_metrics.json",
    "run_config.yml",
    "metadata.yaml",
]
OPTIONAL_ARTIFACTS = ["omni-info.json"]

# Glob-style patterns to exclude when only_required=false (applied recursively)
# Matches: cache/, response_stats_cache/, lm_cache_rank0.db/, *.lock, synthetic/, etc.
EXCLUDED_PATTERNS = ["*cache*", "*.db", "*.lock", "synthetic", "debug.json"]


@dataclass
class ExportResult:
    """Result of an export operation."""

    successful_jobs: List[str]
    failed_jobs: List[str]
    skipped_jobs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataForExport:
    artifacts_dir: Path
    logs_dir: Optional[Path]

    config: Dict[str, Any]
    model_id: str
    metrics: Dict[str, float]

    harness: Optional[str]
    task: str
    container: str

    executor: str
    invocation_id: str
    job_id: str
    timestamp: float

    job_data: Optional[Dict[str, Any]] = None


def get_relevant_artifacts() -> List[str]:
    """Get relevant artifacts (required + optional)."""
    return REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS


def should_exclude_artifact(name: str) -> bool:
    """Check if artifact should be excluded based on glob patterns."""
    name_lower = name.lower()
    for pattern in EXCLUDED_PATTERNS:
        p = pattern.lower()
        if p.startswith("*") and p.endswith("*"):
            # *cache* - contains match
            if p[1:-1] in name_lower:
                return True
        elif p.startswith("*"):
            # *.db, *.lock - suffix match
            if name_lower.endswith(p[1:]):
                return True
        elif name_lower == p:
            # exact match at any depth (synthetic, debug.json)
            return True
    return False


def get_copytree_ignore() -> Callable[[str, List[str]], List[str]]:
    """Return ignore function for shutil.copytree() that excludes artifacts recursively."""

    def ignore_func(directory: str, contents: List[str]) -> List[str]:
        return [name for name in contents if should_exclude_artifact(name)]

    return ignore_func


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


RESULTS_FILE = "results.yml"
METADATA_FILE = "metadata.yaml"
METADATA_CONFIG_KEY = "launcher_resolved_config"
NE_CONFIG_FILE = "run_config.yml"


def extract_accuracy_metrics(
    artifacts_dir: Path, log_metrics: List[str] = None
) -> Dict[str, float]:
    """Extract accuracy metrics from job results.
    artifacts_dir: Path to the artifacts directory
    log_metrics: List of metrics to log. If None, all metrics are logged.
    Returns:
        Dict[str, float]: Dictionary of metrics
    """

    if not artifacts_dir or not artifacts_dir.exists():
        raise RuntimeError(f"Artifacts directory {artifacts_dir} not found")

    metrics = _extract_from_results_yml(artifacts_dir / RESULTS_FILE)
    if not log_metrics:
        return metrics

    # Filter metrics if specified
    filtered_metrics = {}
    for metric_name, metric_value in metrics.items():
        if any(filter_key in metric_name.lower() for filter_key in log_metrics):
            filtered_metrics[metric_name] = metric_value
    return filtered_metrics


def get_model_id(artifacts_dir: Path) -> str:
    """Get model name from ne config file."""
    config_file = artifacts_dir / NE_CONFIG_FILE
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Failed to parse {config_file} - it should be a dictionary")
    if (
        "target" not in data
        or "api_endpoint" not in data["target"]
        or "model_id" not in data["target"]["api_endpoint"]
    ):
        raise ValueError(
            f"Failed to parse {config_file} - no target.api_endpoint.model_id found"
        )
    return data["target"]["api_endpoint"]["model_id"]


def load_config_from_metadata(artifacts_dir: Path) -> Dict[str, Any]:
    """Load nel config from artifacts directory."""
    with open(artifacts_dir / METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Failed to parse {METADATA_FILE} - it should be a dictionary")
    if METADATA_CONFIG_KEY not in metadata:
        raise ValueError(
            f"Failed to parse {METADATA_FILE} - no {METADATA_CONFIG_KEY} section found"
        )
    return metadata[METADATA_CONFIG_KEY]


def load_benchmark_info(artifacts_dir: Path) -> Tuple[str, str]:
    """Load benchmark info from ne config file."""
    config_file = artifacts_dir / NE_CONFIG_FILE
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Failed to parse {config_file} - it should be a dictionary")
    harness = data.get("framework_name", None)
    benchmark = data.get("config", {}).get("type", None)
    return harness, benchmark


# =============================================================================
# CONFIG EXTRACTION
# =============================================================================


def extract_exporter_config(
    data_for_export: DataForExport,
    exporter_name: str,
    constructor_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Extract and merge exporter configuration from multiple sources."""
    config = {}

    # root-level `export.<exporter-name>`
    config = (data_for_export.config or {}).get("export", {}).get(exporter_name, {})

    # From webhook metadata (if triggered by webhook)
    if "webhook_metadata" in data_for_export.job_data:
        webhook_data = data_for_export.job_data["webhook_metadata"]
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


def ssh_setup_masters(remotes: List[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    """Start SSH master connections for remote jobs, returns control_paths.
    Args:
        remotes: List of tuples containing username and hostname
    Returns:
        Dictionary mapping username and hostname to socket path
    """
    if not remotes:
        return {}

    CONNECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    control_paths: Dict[Tuple[str, str], str] = {}
    for username, hostname in remotes:
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
    username: str,
    hostname: str,
    remote_path: str,
    export_dir: Path,
    *,
    copy_logs: bool = False,
    copy_artifacts: bool = True,
    only_required: bool = True,
    control_paths: Dict[Tuple[str, str], str] | None = None,
) -> List[str]:
    """Download artifacts/logs via SSH with optional connection reuse."""
    exported_files: List[str] = []

    control_path = None
    if control_paths:
        control_path = control_paths.get((username, hostname))
    ssh_opts = ["-o", f"ControlPath={control_path}"] if control_path else []

    def scp_file(remote_path: str, local_path: Path) -> bool:
        cmd = (
            ["scp"]
            + ssh_opts
            + [
                f"{username}@{hostname}:{remote_path}",
                str(local_path),
            ]
        )
        return subprocess.run(cmd, capture_output=True).returncode == 0

    export_dir.mkdir(parents=True, exist_ok=True)

    # Artifacts

    art_dir = export_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    if only_required:
        for artifact in get_relevant_artifacts():
            remote_file = f"{remote_path}/artifacts/{artifact}"
            local_file = art_dir / artifact
            local_file.parent.mkdir(parents=True, exist_ok=True)
            if scp_file(remote_file, local_file):
                exported_files.append(str(local_file))
            else:
                logger.error(
                    f"Failed to copy required artifact {artifact} from {remote_file} to {local_file}"
                )
    else:
        # Use tar+ssh to bundle many small files into one transfer
        # This is much faster than rsync for directories with thousands of files
        exclude_args = " ".join(f"--exclude={p}" for p in EXCLUDED_PATTERNS)

        # Build SSH command
        ssh_cmd = ["ssh"] + ssh_opts
        remote_tar_cmd = f"cd {remote_path} && tar -czf - {exclude_args} artifacts/"

        # Stream tar from remote, extract locally
        ssh_full = ssh_cmd + [
            f"{username}@{hostname}",
            remote_tar_cmd,
        ]
        with subprocess.Popen(
            ssh_full, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as ssh_proc:
            tar_extract = subprocess.run(
                ["tar", "-xzf", "-", "-C", str(export_dir)],
                stdin=ssh_proc.stdout,
                capture_output=True,
            )
            ssh_proc.wait()
            if ssh_proc.returncode == 0 and tar_extract.returncode == 0:
                exported_files.extend(
                    [str(f) for f in art_dir.rglob("*") if f.is_file()]
                )

    # Logs (top-level only)
    if copy_logs:
        local_logs = export_dir / "logs"
        remote_logs = f"{remote_path}/logs"
        cmd = (
            ["scp", "-r"]
            + ssh_opts
            + [
                f"{username}@{hostname}:{remote_logs}/.",
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


def _extract_from_results_yml(
    results_yml: Path,
) -> Dict[str, float]:
    """Extract metrics and config from results.yml file."""
    with open(results_yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Failed to parse {results_yml} - it should be a dictionary with 'results' section"
        )
    if "results" not in data:
        raise ValueError(f"Failed to parse {results_yml} - no results section found")

    return _extract_metrics_from_results(data["results"])


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
# CONFIG FLATTENING
# =============================================================================


def flatten_config(
    config: Any,
    parent_key: str = "",
    sep: str = ".",
    max_depth: int = 10,
) -> Dict[str, str]:
    """
    Flatten a nested config dict into dot-notation keys.

    Args:
        config: Nested configuration (dict, list, or scalar)
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Flattened dictionary with string values

    Examples:
        >>> flatten_config({"a": {"b": 1}})
        {"a.b": "1"}
        >>> flatten_config({"tasks": [{"name": "foo"}, {"name": "bar"}]})
        {"tasks.0.name": "foo", "tasks.1.name": "bar"}
    """
    if max_depth <= 0:
        return {parent_key: str(config)} if parent_key else {}

    if isinstance(config, dict):
        items: Dict[str, str] = {}
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.update(flatten_config(value, new_key, sep, max_depth - 1))
        return items

    if isinstance(config, list):
        items: Dict[str, str] = {}
        for idx, item in enumerate(config):
            item_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            items.update(flatten_config(item, item_key, sep, max_depth - 1))
        return items

    # Scalar value
    if not parent_key:
        return {}
    if config is None:
        return {parent_key: "null"}
    return {parent_key: str(config)}


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
