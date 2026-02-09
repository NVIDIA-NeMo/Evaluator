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


def get_available_artifacts(artifacts_dir: Path) -> List[str]:
    """Get list of artifacts available in artifacts directory."""
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
    if not copy_artifacts and not copy_logs:
        logger.warning("Both copy_artifacts and copy_logs are False, nothing to copy")
        return []

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
    if copy_artifacts:
        art_dir = export_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)

        if only_required:
            for artifact in get_relevant_artifacts():
                remote_file = f"{remote_path}/artifacts/{artifact}"
                local_file = art_dir / artifact
                local_file.parent.mkdir(parents=True, exist_ok=True)
                if scp_file(remote_file, local_file):
                    exported_files.append(str(local_file))
                elif artifact in REQUIRED_ARTIFACTS:
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
# COPY ARTIFACTS UTILS
# =============================================================================
def copy_local_artifacts(
    job_dir: Path,
    export_dir: Path,
    only_required: bool = True,
    copy_artifacts: bool = True,
    copy_logs: bool = False,
):
    """Copy artifacts between local directories."""

    if not copy_artifacts and not copy_logs:
        logger.warning("Both copy_artifacts and copy_logs are False, nothing to copy")
        return

    export_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    def cp_file(original_path: Path | str, copy_path: Path | str) -> bool:
        cmd = ["cp", str(original_path), str(copy_path)]
        return subprocess.run(cmd, capture_output=True).returncode == 0

    if copy_artifacts:
        art_dir = export_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        if only_required:
            for artifact in get_relevant_artifacts():
                original_file = job_dir / "artifacts" / artifact
                copy_file = art_dir / artifact
                copy_file.parent.mkdir(parents=True, exist_ok=True)
                if cp_file(original_file, copy_file):
                    exported_files.append(str(copy_file))
                elif artifact in REQUIRED_ARTIFACTS:
                    logger.error(
                        f"Failed to copy required artifact {artifact} from {original_file} to {copy_file}"
                    )

        else:
            # Copy all artifacts recursively (except excluded patterns)
            import shutil

            source_artifacts = job_dir / "artifacts"
            if source_artifacts.exists():
                # Use copytree with ignore function for excluded patterns
                try:
                    shutil.copytree(
                        source_artifacts,
                        art_dir,
                        ignore=get_copytree_ignore(),
                        dirs_exist_ok=True,
                    )
                    exported_files.extend(
                        [str(f) for f in art_dir.rglob("*") if f.is_file()]
                    )
                except Exception as e:
                    logger.error(f"Failed to copy artifacts directory: {e}")

    # Logs (top-level only)
    if copy_logs:
        logs_dir = export_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        original_logs = job_dir / "logs"
        # Copy top-level log files only (not subdirectories)
        if original_logs.exists():
            for log_file in original_logs.glob("*"):
                if log_file.is_file():
                    dest_file = logs_dir / log_file.name
                    if cp_file(log_file, dest_file):
                        exported_files.append(str(dest_file))
                    else:
                        logger.warning(
                            f"Failed to copy log file {log_file} to {dest_file}"
                        )

    return exported_files


def copy_artifacts(
    jobs_data: List[JobData],
    export_dir: Path,
    copy_local: bool = False,
    copy_artifacts: bool = True,
    only_required: bool = True,
    copy_logs: bool = False,
) -> Tuple[List[JobData], List[str]]:
    """Copy artifacts to local filesystem. Returns list of failed jobs. Works both for local and remote jobs."""
    if not copy_artifacts and not copy_logs:
        logger.warning("Both copy_artifacts and copy_logs are False, nothing to copy")
        return jobs_data, []

    remote_jobs_to_copy = []
    failed_job_ids = []
    prepared_jobs_data = []

    for job_data in jobs_data:
        if "output_dir" not in job_data.data:
            if "remote_rundir_path" in job_data.data:
                if "username" not in job_data.data or "hostname" not in job_data.data:
                    raise ValueError(
                        f"Username or hostname not found for remote job: {job_data.job_id}"
                    )
                remote_jobs_to_copy.append(job_data)
            else:
                logger.warning(
                    f"Job {job_data.job_id} has no output directory and is not remote. Could not export job."
                )
                continue
        elif copy_local:
            logger.debug(
                f"Copying local artifacts for job {job_data.job_id} (copy_local={copy_local})"
            )
            new_job_dir = export_dir / job_data.job_id
            copy_local_artifacts(
                Path(job_data.data["output_dir"]),
                new_job_dir,
                only_required=only_required,
                copy_artifacts=copy_artifacts,
                copy_logs=copy_logs,
            )
            job_data.data["output_dir"] = str(new_job_dir)
            prepared_jobs_data.append(job_data)
        else:
            logger.debug(
                f"Skipping local artifacts for job {job_data.job_id} (copy_local={copy_local})"
            )
            prepared_jobs_data.append(job_data)

    if not remote_jobs_to_copy:
        return prepared_jobs_data, failed_job_ids

    remotes = {
        (job_data.data["username"], job_data.data["hostname"])
        for job_data in remote_jobs_to_copy
    }
    cp = ssh_setup_masters(list(remotes))
    try:
        for job_data in remote_jobs_to_copy:
            job_local_dir = export_dir / job_data.job_id
            exported_files = ssh_download_artifacts(
                job_data.data["username"],
                job_data.data["hostname"],
                job_data.data["remote_rundir_path"],
                job_local_dir,
                copy_artifacts=copy_artifacts,
                copy_logs=copy_logs,
                only_required=only_required,
                control_paths=cp,
            )
            logger.debug(f"Exported files: {exported_files}")
            if len(exported_files) == 0:
                logger.warning(
                    f"No artifacts copied for remote job {job_data.job_id}. Could not export job."
                )
                failed_job_ids.append(job_data.job_id)
                continue
            job_data.data["output_dir"] = str(job_local_dir)
            prepared_jobs_data.append(job_data)
    finally:
        ssh_cleanup_masters(cp)
    return prepared_jobs_data, failed_job_ids


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================


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
