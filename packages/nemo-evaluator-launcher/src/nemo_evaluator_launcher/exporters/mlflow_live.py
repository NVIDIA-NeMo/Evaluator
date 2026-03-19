#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""MLflow live exporter for NeMo Evaluator.

Standalone script that runs on the cluster inside the MLflow container.
Reads export_config.yml and exports evaluation results to MLflow.

Modes:
  --init              Create an MLflow run with tags (status=deploying), write run_id to file.
  --update [STATUS]   Heartbeat: set heartbeat timestamp (optionally set status).
  --cancel <status>   Set status on cancellation/timeout, log final artifacts.
  (default)           Finalize: update existing run with metrics, artifacts, status=completed.
"""

import argparse
import logging
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]  # standalone script; mlflow available at runtime

import yaml

_MLFLOW_METRIC_RE = re.compile(r"[^a-zA-Z0-9_\-.\s:/]")
RUN_ID_FILE = "mlflow_run_id.txt"

# Must match REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS in exporters/utils.py
REQUIRED_ARTIFACTS = [
    "results.yml",
    "eval_factory_metrics.json",
    "run_config.yml",
    "metadata.yaml",
]
OPTIONAL_ARTIFACTS = ["omni-info.json"]

# Must match EXCLUDED_PATTERNS in exporters/utils.py
EXCLUDED_PATTERNS = ["*cache*", "*.db", "*.lock", "synthetic", "debug.json"]


def _load_config(results_dir: Path) -> dict:
    """Load export_config.yml — flat config with runtime tags already injected.

    The YAML is written by the SLURM executor at job start and mirrors the
    Flat format: tracking_uri, experiment_name, tags, etc.
    """
    config = yaml.safe_load(
        (results_dir / "export_config.yml").read_text(encoding="utf-8")
    )

    tracking_uri = config.get("tracking_uri", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Normalize log_artifacts/log_logs (NEL uses copy_artifacts/copy_logs)
    if "log_artifacts" not in config:
        config["log_artifacts"] = config.get("copy_artifacts", True)
    if "log_logs" not in config:
        config["log_logs"] = config.get("copy_logs", True)

    config.setdefault("tags", {})
    config.setdefault("experiment_name", "nemo-evaluator-launcher")
    config.setdefault("description", "")
    config.setdefault("skip_existing", False)
    config.setdefault(
        "run_name",
        "eval-{}-{}".format(
            config["tags"].get("invocation_id", ""),
            config["tags"].get("task_name", ""),
        ),
    )

    return config


def _sanitize_metric_key(key: str) -> str:
    return _MLFLOW_METRIC_RE.sub("_", key)


def _extract_task_metrics(prefix: str, task_data: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name, metric_data in (task_data.get("metrics") or {}).items():
        for score_type, score_data in (metric_data.get("scores") or {}).items():
            if not isinstance(score_data, dict) or "value" not in score_data:
                continue
            try:
                value = float(score_data["value"])
            except (TypeError, ValueError):
                continue
            if score_type == metric_name:
                key = _sanitize_metric_key(f"{prefix}_{metric_name}")
            else:
                key = _sanitize_metric_key(f"{prefix}_{metric_name}_{score_type}")
            metrics[key] = value
            # Extract stats (stderr, num_samples, etc.)
            for stat_name, stat_value in (metric_data.get("stats") or {}).items():
                try:
                    metrics[_sanitize_metric_key(f"{key}_{stat_name}")] = float(
                        stat_value
                    )
                except (TypeError, ValueError):
                    continue
    for group_name, group_data in (task_data.get("groups") or {}).items():
        metrics.update(_extract_task_metrics(f"{prefix}_{group_name}", group_data))
    return metrics


def _extract_metrics_from_results_yml(results_yml: Path) -> dict[str, float]:
    data = yaml.safe_load(results_yml.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data:
        return {}
    results = data["results"]
    metrics: dict[str, float] = {}
    for section in ("groups", "tasks"):
        for task_name, task_data in (results.get(section) or {}).items():
            metrics.update(_extract_task_metrics(task_name, task_data))
    return metrics


def _check_existing_run(config: dict) -> tuple[bool, str | None]:
    experiment_id = config["tags"]["invocation_id"]
    task = config["tags"]["task_name"]
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    if not experiment:
        return False, None
    filter_str = f"tags.invocation_id = '{experiment_id}' AND tags.task_name = '{task}'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string=filter_str
    )
    if not runs.empty:
        return True, runs.iloc[0].run_id
    return False, None


def _should_exclude(name: str) -> bool:
    """Check if an artifact should be excluded.

    Matches the glob-style EXCLUDED_PATTERNS logic from exporters/utils.py:
      *cache*   → contains match (case-insensitive)
      *.db      → suffix match
      *.lock    → suffix match
      synthetic → exact match
      debug.json → exact match
    """
    name_lower = name.lower()
    for pattern in EXCLUDED_PATTERNS:
        p = pattern.lower()
        if p.startswith("*") and p.endswith("*"):
            if p[1:-1] in name_lower:
                return True
        elif p.startswith("*"):
            if name_lower.endswith(p[1:]):
                return True
        elif name_lower == p:
            return True
    return False


def _get_artifact_path(config: dict) -> str:
    """Build artifact path matching the existing convention: {harness}.{task_name}."""
    harness = config["tags"].get("harness", "")
    task_name = config["tags"].get("task_name", "")
    if harness:
        return f"{harness}.{task_name}"
    return task_name


def _log_config(results_dir: Path) -> None:
    """Upload config.yml at the MLflow root level (matching old exporter)."""
    config_file = results_dir / "artifacts" / "config.yml"
    if config_file.exists():
        mlflow.log_artifact(str(config_file))


def _log_artifacts(results_dir: Path, config: dict) -> None:
    """Upload all artifacts except excluded patterns.

    Uses shutil.copytree with recursive exclusion matching EXCLUDED_PATTERNS
    (*cache*, *.db, *.lock, synthetic, debug.json), then uploads the staged
    directory to MLflow.
    """
    import shutil
    import tempfile

    artifact_path = _get_artifact_path(config)
    artifacts_dir = results_dir / "artifacts"
    if not artifacts_dir.is_dir():
        return

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "artifacts"
        shutil.copytree(
            artifacts_dir,
            staged,
            ignore=lambda d, c: [n for n in c if _should_exclude(n)],
            dirs_exist_ok=True,
        )
        mlflow.log_artifacts(str(staged), artifact_path=f"{artifact_path}/artifacts")


def _log_logs(results_dir: Path, config: dict) -> None:
    logs_dir = results_dir / "logs"
    if not logs_dir.is_dir():
        return
    artifact_path = f"{_get_artifact_path(config)}/logs"
    for log_file in sorted(logs_dir.iterdir()):
        if log_file.is_file():
            mlflow.log_artifact(str(log_file), artifact_path=artifact_path)


def _init_mlflow_run(results_dir: Path, config: dict) -> str:
    mlflow.set_experiment(config["experiment_name"])
    run_id_path = results_dir / RUN_ID_FILE

    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("status", "retrying")
        return run_id

    with mlflow.start_run() as run:
        tags = {
            str(k)[:250]: str(v)[:5000]
            for k, v in config["tags"].items()
            if v is not None
        }
        tags["status"] = "deploying"
        tags["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tags(tags)
        mlflow.set_tag("mlflow.runName", str(config["run_name"])[:5000])
        if config["description"]:
            mlflow.set_tag("mlflow.note.content", str(config["description"])[:5000])
        # Log params
        params = {
            "invocation_id": config["tags"].get("invocation_id", ""),
            "executor": config["tags"].get("executor", ""),
            "timestamp": tags["timestamp"],
        }
        mlflow.log_params({k: str(v)[:250] for k, v in params.items() if v})
        # Upload config files early for dashboard visibility
        artifact_path = _get_artifact_path(config)
        for item in sorted(results_dir.iterdir()):
            if item.is_file() and item.suffix in (".yaml", ".yml"):
                mlflow.log_artifact(
                    str(item), artifact_path=f"{artifact_path}/artifacts"
                )
        run_id_path.write_text(run.info.run_id, encoding="utf-8")
        return run.info.run_id


def _finalize_mlflow_run(results_dir: Path, config: dict) -> str:
    run_id_path = results_dir / RUN_ID_FILE
    results_yml = results_dir / "artifacts" / "results.yml"
    metrics = (
        _extract_metrics_from_results_yml(results_yml) if results_yml.exists() else {}
    )
    mlflow.set_experiment(config["experiment_name"])

    run_id = (
        run_id_path.read_text(encoding="utf-8").strip()
        if run_id_path.exists()
        else None
    )

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("status", "completed")
            mlflow.set_tag(
                "timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            )
            if metrics:
                mlflow.log_metrics(metrics)
            if config["log_artifacts"]:
                _log_config(results_dir)
                _log_artifacts(results_dir, config)
            if config["log_logs"]:
                _log_logs(results_dir, config)
            return run_id

    if config["skip_existing"]:
        exists, existing_run_id = _check_existing_run(config)
        if exists:
            raise RuntimeError(f"MLflow run already exists: {existing_run_id}")
    with mlflow.start_run() as run:
        tags = {
            str(k)[:250]: str(v)[:5000]
            for k, v in config["tags"].items()
            if v is not None
        }
        tags["status"] = "completed"
        tags["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_tags(tags)
        mlflow.set_tag("mlflow.runName", str(config["run_name"])[:5000])
        if config["description"]:
            mlflow.set_tag("mlflow.note.content", str(config["description"])[:5000])
        if metrics:
            mlflow.log_metrics(metrics)
        if config["log_artifacts"]:
            _log_config(results_dir)
            _log_artifacts(results_dir, config)
        if config["log_logs"]:
            _log_logs(results_dir, config)
        return run.info.run_id


def _update_mlflow_run(results_dir: Path, config: dict, status: str = "") -> None:
    run_id_path = results_dir / RUN_ID_FILE
    if not run_id_path.exists():
        return
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_id=run_id):
        if status:
            mlflow.set_tag("status", status)
        mlflow.set_tag(
            "heartbeat", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        )
        if config["log_artifacts"]:
            _log_artifacts(results_dir, config)
        if config["log_logs"]:
            _log_logs(results_dir, config)


def _cancel_mlflow_run(results_dir: Path, config: dict, status: str) -> None:
    run_id_path = results_dir / RUN_ID_FILE
    if not run_id_path.exists():
        return
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("status", status)
        mlflow.set_tag(
            "ended_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        )
        if config["log_artifacts"]:
            _log_artifacts(results_dir, config)
        if config["log_logs"]:
            _log_logs(results_dir, config)


def main() -> None:
    """Read export config and log results to MLflow.

    Wrapped in try/except so export failures never cause the sbatch job
    to exit non-zero.
    """
    parser = argparse.ArgumentParser(description="MLflow exporter for NeMo Evaluator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--init", action="store_true", help="Create MLflow run (status=deploying)"
    )
    group.add_argument(
        "--update",
        nargs="?",
        const="",
        default=None,
        help="Heartbeat: update timestamp (optionally set --update STATUS)",
    )
    group.add_argument(
        "--cancel",
        nargs="?",
        const="cancelled",
        default=None,
        help="Set cancellation status (default: cancelled)",
    )
    args = parser.parse_args()

    try:
        results_dir = Path(__file__).resolve().parent
        config = _load_config(results_dir)

        if args.update is not None:
            _update_mlflow_run(results_dir, config, args.update)
        elif args.cancel is not None:
            _cancel_mlflow_run(results_dir, config, args.cancel)
        elif args.init:
            run_id = _init_mlflow_run(results_dir, config)
            print(f"[export:init] SUCCESS — MLflow run: {run_id}")
        else:
            run_id = _finalize_mlflow_run(results_dir, config)
            print(f"[export] SUCCESS — MLflow run: {run_id}")
    except Exception:  # pylint: disable=broad-exception-caught
        logging.basicConfig(level=logging.WARNING)
        logging.warning(
            "[export] FAILED — results were NOT exported to MLflow. "
            "The evaluation itself completed successfully.\n%s",
            traceback.format_exc(),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
