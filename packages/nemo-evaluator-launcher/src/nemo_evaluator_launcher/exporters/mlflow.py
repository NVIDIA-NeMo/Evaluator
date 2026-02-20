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
"""Evaluation results exporter for MLflow tracking."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import (
    DataForExport,
    flatten_config,
    get_available_artifacts,
    get_copytree_ignore,
    mlflow_sanitize,
)


@register_exporter("mlflow")
class MLflowExporter(BaseExporter):
    """Export accuracy metrics to MLflow tracking server."""

    DEFAULT_EXPERIMENT_NAME = "nemo-evaluator-launcher"

    def is_available(self) -> bool:
        return MLFLOW_AVAILABLE

    def _get_existing_run_id(self, job_id: str, experiment_name: str) -> str | None:
        """Check if MLflow run exists for this job and return the run id if it does."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return None

            # Search for runs with matching invocation_id tag
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.job_id = '{job_id}'",
            )

            if not runs.empty:
                existing_run = runs.iloc[0]
                if len(runs) > 1:
                    logger.warning(
                        f"Multiple runs found for job {job_id}, using the first one: {existing_run.run_id}"
                    )
                else:
                    logger.info(f"Run found for job {job_id}: {existing_run.run_id}")

                return existing_run.run_id

        except Exception as e:
            logger.debug(f"Error checking if run exists for job {job_id}: {e}")
            return None

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export job to MLflow."""
        if not self.is_available():
            logger.error(
                "MLflow package not installed. Install via: pip install nemo-evaluator-launcher[mlflow]"
            )
            return [], [data.job_id for data in data_for_export], []

        successful_jobs = []
        failed_jobs = []
        skipped_jobs = []

        # Extract config using common utility
        tracking_uri = self.config.get("tracking_uri")
        if not tracking_uri:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        # allow env var name
        if tracking_uri and "://" not in tracking_uri:
            tracking_uri = os.getenv(tracking_uri, tracking_uri)

        if not tracking_uri:
            logger.error(
                "MLflow requires 'tracking_uri' to be configured. "
                "Set export.mlflow.tracking_uri field in the config "
                "or MLFLOW_TRACKING_URI environment variable."
            )
            return [], [data.job_id for data in data_for_export], []

        # Set up MLflow
        tracking_uri = tracking_uri.rstrip("/")
        mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        experiment_name = self.config.get(
            "experiment_name", self.DEFAULT_EXPERIMENT_NAME
        )
        mlflow.set_experiment(experiment_name)
        for data in data_for_export:
            try:
                result = self._export_one_job(data)
            except Exception as e:
                logger.error(f"Error exporting job {data.job_id}: {e}")
                failed_jobs.append(data.job_id)
                continue
            if result == "skipped":
                skipped_jobs.append(data.job_id)
            elif result == "failed":
                failed_jobs.append(data.job_id)
            else:
                successful_jobs.append(data.job_id)
        return successful_jobs, failed_jobs, skipped_jobs

    def _export_one_job(self, job_data: DataForExport) -> str:
        # Prepare parameters
        all_params = {
            "invocation_id": job_data.invocation_id,
            "executor": job_data.executor,
            "timestamp": str(job_data.timestamp),
        }

        # Add extra metadata if provided
        if self.config.get("extra_metadata"):
            all_params.update(self.config["extra_metadata"])

        # Add flattened config as params if enabled
        if self.config.get("log_config_params", False):
            config_params = flatten_config(
                job_data.config or {},
                parent_key="config",
                max_depth=self.config.get("log_config_params_max_depth", 10),
            )
            all_params.update(config_params)

        # Add webhook info if available
        if self.config.get("triggered_by_webhook"):
            all_params.update(
                {
                    "webhook_triggered": "true",
                    "webhook_source": self.config.get("webhook_source"),
                    "source_artifact": self.config.get("source_artifact"),
                    "config_source": self.config.get("config_source"),
                }
            )

        # Sanitize params
        safe_params = {
            mlflow_sanitize(k, "param_key"): mlflow_sanitize(v, "param_value")
            for k, v in (all_params or {}).items()
            if v is not None
        }

        # Prepare tags
        tags = {
            "invocation_id": job_data.invocation_id,
            "job_id": job_data.job_id,
            "task_name": job_data.task,
            "benchmark": job_data.task,
            "harness": job_data.harness,
            "executor": job_data.executor,
        }
        if self.config.get("tags"):
            tags.update({k: v for k, v in self.config["tags"].items() if v})

        # Sanitize tags
        safe_tags = {
            mlflow_sanitize(k, "tag_key"): mlflow_sanitize(v, "tag_value")
            for k, v in tags.items()
            if v is not None
        }

        # skip run if it already exists
        existing_run_id = None
        skip_existing = self.config.get("skip_existing", False)
        existing_run_id = self._get_existing_run_id(
            job_data.job_id,
            self.config.get("experiment_name", self.DEFAULT_EXPERIMENT_NAME),
        )
        if existing_run_id and skip_existing:
            return "skipped"

        # run
        try:
            with mlflow.start_run(run_id=existing_run_id):
                # Set tags
                if safe_tags:
                    mlflow.set_tags(safe_tags)

                # Set run name
                run_name = (
                    self.config.get("run_name")
                    or f"eval-{job_data.invocation_id}-{job_data.task}"
                )
                mlflow.set_tag("mlflow.runName", mlflow_sanitize(run_name, "tag_value"))

                # Set description only if provided
                description = self.config.get("description")
                if description:
                    mlflow.set_tag(
                        "mlflow.note.content", mlflow_sanitize(description, "tag_value")
                    )

                # Log parameters
                mlflow.log_params(safe_params)

                # Sanitize metric keys before logging
                safe_metrics = {
                    mlflow_sanitize(k, "metric"): v
                    for k, v in (job_data.metrics or {}).items()
                }
                mlflow.log_metrics(safe_metrics)

                # Log artifacts
                _ = self._log_artifacts(job_data)

            return "success"
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
            return "failed"

    def _log_artifacts(
        self,
        job_data: DataForExport,
    ) -> List[str]:
        """Log evaluation artifacts to MLflow using LocalExporter for transfer."""

        # Check if artifacts should be logged (default: True)
        if not self.config.get("log_artifacts", True):
            return []

        # Log config at root level (or synthesize)
        fname = "config.yml"
        p = job_data.artifacts_dir / fname
        if p.exists():
            mlflow.log_artifact(str(p))
        else:
            # TODO(martas): why we need this? is it even possible?
            with tempfile.TemporaryDirectory() as tmpdir:
                import yaml

                cfg_file = Path(tmpdir) / "config.yaml"
                cfg_file.write_text(
                    yaml.dump(
                        job_data.config or {},
                        default_flow_style=False,
                        sort_keys=False,
                    )
                )
                mlflow.log_artifact(str(cfg_file))

        # Choose files to upload
        artifact_path = f"{job_data.harness}.{job_data.task}"
        logged_names = []
        if self.config.get("only_required", True):
            # Upload only specific required files
            for fname in get_available_artifacts(job_data.artifacts_dir):
                p = job_data.artifacts_dir / fname
                if p.exists():
                    mlflow.log_artifact(
                        str(p),
                        artifact_path=f"{artifact_path}/artifacts",
                    )
                    logged_names.append(fname)
                    logger.debug(f"mlflow upload artifact: {fname}")
        else:
            # Upload all artifacts with recursive exclusion
            # Stage to temp dir with exclusions, then upload
            with tempfile.TemporaryDirectory() as tmp:
                staged = Path(tmp) / "artifacts"
                shutil.copytree(
                    job_data.artifacts_dir,
                    staged,
                    ignore=get_copytree_ignore(),
                    dirs_exist_ok=True,
                )
                # Upload entire staged directory
                mlflow.log_artifacts(
                    str(staged), artifact_path=f"{artifact_path}/artifacts"
                )
                logged_names.extend([p.name for p in staged.iterdir()])
                logger.debug(
                    f"mlflow upload artifacts: {len(logged_names)} items (with exclusions)"
                )

        # Optionally upload logs under "<harness.task>/logs"
        if self.config.get("log_logs", False) and job_data.logs_dir.exists():
            for p in job_data.logs_dir.iterdir():
                if p.is_file():
                    rel = p.name
                    mlflow.log_artifact(str(p), artifact_path=f"{artifact_path}/logs")
                    logged_names.append(f"logs/{rel}")
                    logger.debug(f"mlflow upload log: {rel}")

        logger.info(
            f"MLflow upload summary: files={len(logged_names)}, only_required={self.config.get('only_required', True)}, log_logs={self.config.get('log_logs', False)}"
        )

        return logged_names
