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
import tempfile
from pathlib import Path
from typing import Any, Dict, List

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter, ExportResult
from nemo_evaluator_launcher.exporters.local import LocalExporter
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import (
    extract_accuracy_metrics,
    extract_exporter_config,
    get_artifact_root,
    get_available_artifacts,
    get_benchmark_info,
    get_task_name,
    mlflow_sanitize,
)


@register_exporter("mlflow")
class MLflowExporter(BaseExporter):
    """Export accuracy metrics to MLflow tracking server."""

    def supports_executor(self, executor_type: str) -> bool:
        return True

    def is_available(self) -> bool:
        return MLFLOW_AVAILABLE

    def _get_existing_run_info(
        self, job_data: JobData, config: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Check if MLflow run exists for this invocation/job."""
        try:
            import mlflow

            tracking_uri = config.get("tracking_uri")
            if not tracking_uri:
                return False, None

            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = config.get("experiment_name", "nemo-evaluator-launcher")

            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if not experiment:
                    return False, None

                # Search for runs with matching invocation_id tag
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.invocation_id = '{job_data.invocation_id}'",
                )

                if not runs.empty:
                    existing_run = runs.iloc[0]
                    return True, existing_run.run_id

            except Exception:
                pass

            return False, None
        except ImportError:
            return False, None

    def export_job(self, job_data: JobData) -> ExportResult:
        """Export job to MLflow."""
        if not self.is_available():
            return ExportResult(
                success=False, dest="mlflow", message="mlflow package not installed"
            )

        try:
            # Extract config using common utility
            mlflow_config = extract_exporter_config(job_data, "mlflow", self.config)

            # resolve tracking_uri with fallbacks
            tracking_uri = mlflow_config.get("tracking_uri")
            if not tracking_uri:
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            # allow env var name
            if tracking_uri and "://" not in tracking_uri:
                tracking_uri = os.getenv(tracking_uri, tracking_uri)

            if not tracking_uri:
                return ExportResult(
                    success=False,
                    dest="mlflow",
                    message="tracking_uri is required (set export.mlflow.tracking_uri or MLFLOW_TRACKING_URI)",
                )

            # Stage artifacts locally if remote_ssh (e.g., Slurm), so we can extract metrics
            staged_base_dir = None
            try:
                paths = self.get_job_paths(job_data)
                if paths.get("storage_type") == "remote_ssh":
                    tmp_stage = Path(tempfile.mkdtemp(prefix="mlflow_stage_"))
                    LocalExporter(
                        {
                            "output_dir": str(tmp_stage),
                            "copy_logs": mlflow_config.get(
                                "log_logs", False
                            ),  # log_logs -> copy_logs
                            "only_required": mlflow_config.get("only_required", True),
                        }
                    ).export_job(job_data)
                    staged_base_dir = (
                        tmp_stage / job_data.invocation_id / job_data.job_id
                    )
            except Exception as e:
                logger.warning(f"Failed staging artifacts for {job_data.job_id}: {e}")

            # Extract metrics (prefer staged if available)
            log_metrics = mlflow_config.get("log_metrics", [])
            if staged_base_dir and (staged_base_dir / "artifacts").exists():
                accuracy_metrics = extract_accuracy_metrics(
                    job_data,
                    lambda _: {
                        "artifacts_dir": staged_base_dir / "artifacts",
                        "storage_type": "local_filesystem",
                    },
                    log_metrics,
                )
            else:
                accuracy_metrics = extract_accuracy_metrics(
                    job_data, self.get_job_paths, log_metrics
                )

            if not accuracy_metrics:
                return ExportResult(
                    success=False, dest="mlflow", message="No accuracy metrics found"
                )

            # Set up MLflow
            tracking_uri = tracking_uri.rstrip("/")
            mlflow.set_tracking_uri(tracking_uri)

            # Set experiment
            experiment_name = mlflow_config.get(
                "experiment_name", "nemo-evaluator-launcher"
            )
            mlflow.set_experiment(experiment_name)

            # Prepare parameters
            all_params = {
                "invocation_id": job_data.invocation_id,
                "executor": job_data.executor,
                "timestamp": str(job_data.timestamp),
            }

            # Add extra metadata if provided
            if mlflow_config.get("extra_metadata"):
                all_params.update(mlflow_config["extra_metadata"])

            # Add webhook info if available
            if mlflow_config.get("triggered_by_webhook"):
                all_params.update(
                    {
                        "webhook_triggered": "true",
                        "webhook_source": mlflow_config.get("webhook_source"),
                        "source_artifact": mlflow_config.get("source_artifact"),
                        "config_source": mlflow_config.get("config_source"),
                    }
                )

            # Sanitize params
            safe_params = {
                mlflow_sanitize(k, "param_key"): mlflow_sanitize(v, "param_value")
                for k, v in (all_params or {}).items()
                if v is not None
            }

            # Prepare tags
            tags = {}
            if mlflow_config.get("tags"):
                tags.update({k: v for k, v in mlflow_config["tags"].items() if v})

            bench_info = get_benchmark_info(job_data)
            benchmark = bench_info.get("benchmark", get_task_name(job_data))
            harness = bench_info.get("harness", "unknown")

            # Tag the run with invocation_id and task metadata
            exec_type = (job_data.config or {}).get("execution", {}).get(
                "type"
            ) or job_data.executor
            tags.update(
                {
                    "invocation_id": job_data.invocation_id,
                    "job_id": job_data.job_id,
                    "task_name": benchmark,
                    "benchmark": benchmark,
                    "harness": harness,
                    "executor": exec_type,
                }
            )

            # Sanitize tags
            safe_tags = {
                mlflow_sanitize(k, "tag_key"): mlflow_sanitize(v, "tag_value")
                for k, v in (tags or {}).items()
                if v is not None
            }

            # skip run if it already exists
            exists, existing_run_id = self._get_existing_run_info(
                job_data, mlflow_config
            )
            if exists and mlflow_config.get("skip_existing"):
                return ExportResult(
                    success=True,
                    dest="mlflow",
                    message=f"Run already exists: {existing_run_id}, skipped",
                )

            # run
            with mlflow.start_run() as run:
                # Set tags
                if safe_tags:
                    mlflow.set_tags(safe_tags)

                # Set run name
                run_name = (
                    mlflow_config.get("run_name")
                    or f"eval-{job_data.invocation_id}-{benchmark}"
                )
                mlflow.set_tag("mlflow.runName", mlflow_sanitize(run_name, "tag_value"))

                # Set description only if provided
                description = mlflow_config.get("description")
                if description:
                    mlflow.set_tag(
                        "mlflow.note.content", mlflow_sanitize(description, "tag_value")
                    )

                # Log parameters
                mlflow.log_params(safe_params)

                # Sanitize metric keys before logging
                safe_metrics = {
                    mlflow_sanitize(k, "metric"): v
                    for k, v in (accuracy_metrics or {}).items()
                }
                mlflow.log_metrics(safe_metrics)

                # Log artifacts
                artifacts_logged = self._log_artifacts(
                    job_data, mlflow_config, staged_base_dir
                )

                # Build run URL
                run_url = None
                if tracking_uri.startswith(("http://", "https://")):
                    run_url = f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

                return ExportResult(
                    success=True,
                    dest="mlflow",
                    message=f"Logged {len(accuracy_metrics)} metrics to MLflow",
                    metadata={
                        "run_id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "tracking_uri": tracking_uri,
                        "run_url": run_url,
                        "invocation_id": job_data.invocation_id,
                        "metrics_logged": len(accuracy_metrics),
                        "params_logged": len(safe_params),
                        "artifacts_logged": len(artifacts_logged),
                    },
                )

        except Exception as e:
            logger.error(f"MLflow export failed: {e}")
            return ExportResult(
                success=False, dest="mlflow", message=f"Failed: {str(e)}"
            )

    def _log_artifacts(
        self,
        job_data: JobData,
        mlflow_config: Dict[str, Any],
        pre_staged_dir: Path = None,
    ) -> List[str]:
        """Log evaluation artifacts to MLflow using LocalExporter for transfer."""

        # Check if artifacts should be logged (default: True)
        if not mlflow_config.get("log_artifacts", True):
            return []

        try:
            should_cleanup = False
            # Use pre-staged dir if available; otherwise stage now
            if pre_staged_dir and pre_staged_dir.exists():
                base_dir = pre_staged_dir
            else:
                temp_dir = tempfile.mkdtemp(prefix="mlflow_artifacts_")
                local_exporter = LocalExporter(
                    {
                        "output_dir": str(temp_dir),
                        "copy_logs": mlflow_config.get(
                            "log_logs", mlflow_config.get("copy_logs", False)
                        ),
                        "only_required": mlflow_config.get("only_required", True),
                        "format": mlflow_config.get("format", None),
                        "log_metrics": mlflow_config.get("log_metrics", []),
                        "output_filename": mlflow_config.get("output_filename", None),
                    }
                )
                local_result = local_exporter.export_job(job_data)
                if not local_result.success:
                    logger.error(
                        f"Failed to download artifacts: {local_result.message}"
                    )
                    return []
                base_dir = Path(local_result.dest)
                should_cleanup = True

            artifacts_dir = base_dir / "artifacts"
            logs_dir = base_dir / "logs"
            logged_names: list[str] = []
            artifact_path = get_artifact_root(job_data)  # "<harness>.<benchmark>"

            # Log config at root level (or synthesize)
            cfg_logged = False
            for fname in ("config.yml", "run_config.yml"):
                p = artifacts_dir / fname
                if p.exists():
                    mlflow.log_artifact(str(p))
                    cfg_logged = True
                    break
            if not cfg_logged:
                with tempfile.TemporaryDirectory() as tmpdir:
                    from yaml import dump as ydump

                    cfg_file = Path(tmpdir) / "config.yaml"
                    cfg_file.write_text(
                        ydump(
                            job_data.config or {},
                            default_flow_style=False,
                            sort_keys=False,
                        )
                    )
                    mlflow.log_artifact(str(cfg_file))

            # Choose files to upload
            files_to_upload: list[Path] = []
            if mlflow_config.get("only_required", True):
                for fname in get_available_artifacts(artifacts_dir):
                    p = artifacts_dir / fname
                    if p.exists():
                        files_to_upload.append(p)
            else:
                for p in artifacts_dir.iterdir():  # top-level files only
                    if p.is_file():
                        files_to_upload.append(p)

            # Upload artifacts (with DEBUG per-file)
            for fpath in files_to_upload:
                rel = fpath.relative_to(artifacts_dir).as_posix()
                parent = os.path.dirname(rel)
                mlflow.log_artifact(
                    str(fpath),
                    artifact_path=f"{artifact_path}/artifacts/{parent}".rstrip("/"),
                )
                logged_names.append(rel)
                logger.debug(f"mlflow upload artifact: {rel}")

            # Optionally upload logs under "<harness.task>/logs"
            if mlflow_config.get("log_logs", False) and logs_dir.exists():
                for p in logs_dir.iterdir():
                    if p.is_file():
                        rel = p.name
                        mlflow.log_artifact(
                            str(p), artifact_path=f"{artifact_path}/logs"
                        )
                        logged_names.append(f"logs/{rel}")
                        logger.debug(f"mlflow upload log: {rel}")

            logger.info(
                f"MLflow upload summary: files={len(logged_names)}, only_required={mlflow_config.get('only_required', True)}, log_logs={mlflow_config.get('log_logs', False)}"
            )
            if should_cleanup:
                import shutil

                shutil.rmtree(base_dir, ignore_errors=True)

            return logged_names
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
            return []

    def export_invocation(self, invocation_id: str) -> Dict[str, Any]:
        """Export all jobs in invocation as one MLflow run."""
        if not self.is_available():
            return {"success": False, "error": "mlflow package not installed"}

        jobs = self.db.get_jobs(invocation_id)
        if not jobs:
            return {
                "success": False,
                "error": f"No jobs found for invocation {invocation_id}",
            }

        try:
            # Get first job for config access
            first_job = list(jobs.values())[0]

            # Extract config using common utility
            mlflow_config = extract_exporter_config(first_job, "mlflow", self.config)

            # resolve tracking_uri with fallbacks
            tracking_uri = mlflow_config.get("tracking_uri") or os.getenv(
                "MLFLOW_TRACKING_URI"
            )
            if tracking_uri and "://" not in tracking_uri:
                tracking_uri = os.getenv(tracking_uri, tracking_uri)
            if not tracking_uri:
                return {
                    "success": False,
                    "error": "tracking_uri is required (set export.mlflow.tracking_uri or MLFLOW_TRACKING_URI)",
                }

            # Collect metrics from ALL jobs
            all_metrics = {}
            staged_map: dict[str, Path] = {}
            for job_id, job_data in jobs.items():
                try:
                    paths = self.get_job_paths(job_data)
                    if paths.get("storage_type") == "remote_ssh":
                        tmp_stage = Path(tempfile.mkdtemp(prefix="mlflow_inv_stage_"))
                        LocalExporter(
                            {
                                "output_dir": str(tmp_stage),
                                "copy_logs": mlflow_config.get("log_logs", False),
                                "only_required": mlflow_config.get(
                                    "only_required", True
                                ),
                            }
                        ).export_job(job_data)
                        staged_map[job_id] = (
                            tmp_stage / job_data.invocation_id / job_data.job_id
                        )
                except Exception as e:
                    logger.warning(f"Staging failed for {job_id}: {e}")

            for job_id, job_data in jobs.items():
                log_metrics = mlflow_config.get("log_metrics", [])
                if job_id in staged_map and (staged_map[job_id] / "artifacts").exists():
                    job_metrics = extract_accuracy_metrics(
                        job_data,
                        lambda _: {
                            "artifacts_dir": staged_map[job_id] / "artifacts",
                            "storage_type": "local_filesystem",
                        },
                        log_metrics,
                    )
                else:
                    job_metrics = extract_accuracy_metrics(
                        job_data, self.get_job_paths, log_metrics
                    )
                all_metrics.update(job_metrics)

            if not all_metrics:
                return {
                    "success": False,
                    "error": "No accuracy metrics found in any job",
                }

            # Set up MLflow
            tracking_uri = tracking_uri.rstrip("/")
            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = mlflow_config.get(
                "experiment_name", "nemo-evaluator-launcher"
            )
            mlflow.set_experiment(experiment_name)

            # Prepare parameters for invocation
            inv_exec_type = (first_job.config or {}).get("execution", {}).get(
                "type"
            ) or first_job.executor
            all_params = {
                "invocation_id": invocation_id,
                "executor": inv_exec_type,
                "timestamp": str(first_job.timestamp),
                "jobs_count": str(len(jobs)),
            }

            # Add webhook info if available
            if mlflow_config.get("triggered_by_webhook"):
                all_params.update(
                    {
                        "webhook_triggered": "true",
                        "webhook_source": mlflow_config.get("webhook_source"),
                        "source_artifact": mlflow_config.get("source_artifact"),
                        "config_source": mlflow_config.get("config_source"),
                    }
                )

            if mlflow_config.get("extra_metadata"):
                all_params.update(mlflow_config["extra_metadata"])

            # Prepare tags
            tags = {"invocation_id": invocation_id}
            if mlflow_config.get("tags"):
                tags.update({k: v for k, v in mlflow_config["tags"].items() if v})

            # Truncate
            safe_params = {
                str(k)[:250]: str(v)[:250] for k, v in all_params.items() if v
            }
            safe_tags = {str(k)[:250]: str(v)[:5000] for k, v in tags.items() if v}

            # Check for existing run
            exists, existing_run_id = self._get_existing_run_info(
                first_job, mlflow_config
            )
            if exists and mlflow_config.get("skip_existing"):
                return {
                    "success": True,
                    "invocation_id": invocation_id,
                    "jobs": {
                        job_id: {
                            "success": True,
                            "message": f"Run already exists: {existing_run_id}, skipped",
                        }
                        for job_id in jobs.keys()
                    },
                    "metadata": {"run_id": existing_run_id, "skipped": True},
                }

            # Create MLflow run with ALL metrics
            with mlflow.start_run() as run:
                # Set tags
                if safe_tags:
                    mlflow.set_tags(safe_tags)

                # Set run name
                run_name = mlflow_config.get("run_name") or f"eval-{invocation_id}"
                mlflow.set_tag("mlflow.runName", mlflow_sanitize(run_name, "tag_value"))

                # Set description
                description = mlflow_config.get("description")
                if description:
                    mlflow.set_tag(
                        "mlflow.note.content", mlflow_sanitize(description, "tag_value")
                    )

                # Log parameters
                mlflow.log_params(safe_params)

                # Sanitize metric keys
                safe_all_metrics = {
                    mlflow_sanitize(k, "metric"): v
                    for k, v in (all_metrics or {}).items()
                }
                mlflow.log_metrics(safe_all_metrics)

                # Log artifacts from all jobs
                total_artifacts = 0
                for job_id, job_data in jobs.items():
                    artifacts_logged = self._log_artifacts(
                        job_data, mlflow_config, staged_map.get(job_id)
                    )
                    total_artifacts += len(artifacts_logged)

                # Build run URL
                run_url = None
                if tracking_uri.startswith(("http://", "https://")):
                    run_url = f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

                return {
                    "success": True,
                    "invocation_id": invocation_id,
                    "jobs": {
                        job_id: {
                            "success": True,
                            "message": "Contributed to invocation run",
                        }
                        for job_id in jobs.keys()
                    },
                    "metadata": {
                        "run_id": run.info.run_id,
                        "experiment_id": run.info.experiment_id,
                        "tracking_uri": tracking_uri,
                        "run_url": run_url,
                        "metrics_logged": len(all_metrics),
                        "params_logged": len(safe_params),
                        "artifacts_logged": total_artifacts,
                    },
                }
        except Exception as e:
            logger.error(f"MLflow export failed for invocation {invocation_id}: {e}")
            return {"success": False, "error": f"MLflow export failed: {str(e)}"}
