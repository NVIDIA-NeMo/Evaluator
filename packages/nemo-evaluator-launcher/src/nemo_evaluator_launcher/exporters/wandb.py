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
"""Weights & Biases results exporter."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import (
    DataForExport,
    get_available_artifacts,
    get_copytree_ignore,
)


@register_exporter("wandb")
class WandBExporter(BaseExporter):
    """Export accuracy metrics to W&B.

    Config keys:
      entity (str): W&B entity/team name (required)
      project (str): W&B project name (required)
      log_mode (str): "per_task" (one run per task) or "multi_task" (one run per invocation) (default: "per_task")
      log_artifacts (bool): Whether to log artifacts to W&B (default: True)
      log_logs (bool): Whether to log log files (default: False)
      only_required (bool): Log only required+optional artifacts (default: True)
      name (str): Custom run name (optional)
      group (str): Run group (default: invocation_id)
      job_type (str): W&B job type (default: "evaluation")
      tags (list): List of tags for the run
      description (str): Run description/notes
      extra_metadata (dict): Additional metadata to include in run config
    """

    def is_available(self) -> bool:
        return WANDB_AVAILABLE

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export jobs to W&B."""
        if not self.is_available():
            logger.error(
                "W&B package not installed. "
                "Install via: pip install nemo-evaluator-launcher[wandb]"
            )
            return [], [data.job_id for data in data_for_export], []

        successful_jobs = []
        failed_jobs = []
        skipped_jobs = []

        # Get log_mode from config
        log_mode = self.config.get("log_mode", "per_task")
        if log_mode not in ["per_task", "multi_task"]:
            logger.error(
                f"Invalid log_mode: {log_mode}. Valid modes are: per_task, multi_task"
            )
            return [], [data.job_id for data in data_for_export], []

        try:
            if log_mode == "per_task":
                # Create separate run for each task
                for data in data_for_export:
                    try:
                        if not data.metrics:
                            logger.warning(
                                f"No metrics found for job {data.job_id}, skipping"
                            )
                            skipped_jobs.append(data.job_id)
                            continue

                        identifier = f"{data.invocation_id}-{data.task}"
                        result = self._create_wandb_run(
                            identifier=identifier,
                            metrics=data.metrics,
                            data=data,
                            should_resume=False,
                            existing_run_id=None,
                        )
                        successful_jobs.append(data.job_id)
                        logger.info(
                            f"Exported job {data.job_id} to W&B: {result.get('run_url')}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to export job {data.job_id}: {e}")
                        failed_jobs.append(data.job_id)

            elif log_mode == "multi_task":
                # Create one run for all jobs in the same invocation
                if not data_for_export:
                    return successful_jobs, failed_jobs, skipped_jobs

                # Group by invocation_id
                invocations = {}
                for data in data_for_export:
                    if data.invocation_id not in invocations:
                        invocations[data.invocation_id] = []
                    invocations[data.invocation_id].append(data)

                # Create one run per invocation
                for invocation_id, inv_data_list in invocations.items():
                    try:
                        # Aggregate metrics from all jobs
                        all_metrics = {}
                        for data in inv_data_list:
                            if data.metrics:
                                all_metrics.update(data.metrics)

                        if not all_metrics:
                            logger.warning(
                                f"No metrics found for invocation {invocation_id}, skipping"
                            )
                            skipped_jobs.extend([d.job_id for d in inv_data_list])
                            continue

                        # Check if run exists
                        should_resume, run_id = self._check_existing_run(
                            invocation_id, inv_data_list[0]
                        )

                        # Use first job data as template
                        first_data = inv_data_list[0]
                        result = self._create_wandb_run(
                            identifier=invocation_id,
                            metrics=all_metrics,
                            data=first_data,
                            should_resume=should_resume,
                            existing_run_id=run_id,
                            all_data=inv_data_list,  # Pass all data for multi_task mode
                        )
                        successful_jobs.extend([d.job_id for d in inv_data_list])
                        logger.info(
                            f"Exported invocation {invocation_id} to W&B: {result.get('run_url')}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to export invocation {invocation_id}: {e}"
                        )
                        failed_jobs.extend([d.job_id for d in inv_data_list])

        except Exception as e:
            logger.error(f"W&B export failed: {e}")
            failed_jobs.extend(
                [d.job_id for d in data_for_export if d.job_id not in successful_jobs]
            )

        return successful_jobs, failed_jobs, skipped_jobs

    def _log_artifacts(
        self,
        data: DataForExport,
        artifact,
    ) -> List[str]:
        """Log evaluation artifacts to WandB."""
        if not self.config.get("log_artifacts", True):
            return []

        try:
            artifacts_dir = data.artifacts_dir
            logs_dir = data.logs_dir
            logged_names: List[str] = []

            artifact_root = f"{data.harness}.{data.task}"  # "<harness>.<benchmark>"

            # Log config at root level (or synthesize)
            fname = "config.yml"
            p = data.artifacts_dir / fname
            if p.exists():
                artifact.add_file(str(p), name=f"{artifact_root}/{fname}")
                logged_names.append(fname)
            else:
                # TODO(martas): why we need this? is it even possible?
                with tempfile.NamedTemporaryFile("w", suffix=".yml") as tmp_cfg:
                    import yaml

                    yaml.dump(
                        data.config or {},
                        tmp_cfg,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                    cfg_path = tmp_cfg.name
                    artifact.add_file(cfg_path, name=f"{artifact_root}/{fname}")

                logged_names.append(fname)

            if self.config.get("only_required", True):
                # Upload only specific required files
                for p in get_available_artifacts(artifacts_dir):
                    artifact.add_file(str(p), name=f"{artifact_root}/artifacts/{p}")
                    logged_names.append(p)
            else:
                # Upload all artifacts with recursive exclusion
                with tempfile.TemporaryDirectory() as tmp:
                    staged = Path(tmp) / "artifacts"
                    shutil.copytree(
                        artifacts_dir,
                        staged,
                        ignore=get_copytree_ignore(),
                        dirs_exist_ok=True,
                    )
                    # Add entire staged directory to artifact
                    artifact.add_dir(str(staged), name=f"{artifact_root}/artifacts")
                    # Count items for logging
                    logged_names.extend([p.name for p in staged.iterdir()])

            if self.config.get("log_logs", False) and logs_dir and logs_dir.exists():
                for p in logs_dir.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(logs_dir).as_posix()
                        artifact.add_file(str(p), name=f"{artifact_root}/logs/{rel}")
                        logged_names.append(f"logs/{rel}")

            return logged_names
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
            return []

    def _check_existing_run(
        self, identifier: str, data: DataForExport
    ) -> tuple[bool, Optional[str]]:
        """Check if run exists based on webhook metadata then name patterns."""
        try:
            import wandb

            api = wandb.Api()
            entity = self.config.get("entity")
            project = self.config.get("project")
            if not (entity and project):
                logger.error(
                    "W&B requires 'entity' and 'project' to be configured. "
                    "Set export.wandb.entity and export.wandb.project fields in the config."
                )
                return False, None

            # Check webhook metadata for run_id first
            webhook_meta = (data.job_data or {}).get("webhook_metadata", {})
            if (
                webhook_meta.get("webhook_source") == "wandb"
                and self.config.get("triggered_by_webhook")
                and "run_id" in webhook_meta
            ):
                try:
                    # Verify the run actually exists
                    run = api.run(f"{entity}/{project}/{webhook_meta['run_id']}")
                    return True, run.id
                except Exception:
                    pass

            # Check explicit name first
            if self.config.get("name"):
                runs = api.runs(f"{entity}/{project}")
                for run in runs:
                    if run.display_name == self.config["name"]:
                        return True, run.id

            # Check default pattern
            default_run_name = f"eval-{identifier}"
            runs = api.runs(f"{entity}/{project}")
            for run in runs:
                if run.display_name == default_run_name:
                    return True, run.id

            return False, None
        except Exception:
            return False, None

    def _create_wandb_run(
        self,
        identifier: str,
        metrics: Dict[str, float],
        data: DataForExport,
        should_resume: bool,
        existing_run_id: Optional[str],
        all_data: Optional[List[DataForExport]] = None,
    ) -> Dict[str, Any]:
        """Create or resume W&B run for job(s)."""
        log_mode = self.config.get("log_mode", "per_task")
        task_name = data.task
        benchmark = data.task
        harness = data.harness or "unknown"

        if self.config.get("name"):
            run_name = self.config["name"]
        else:
            run_name = (
                f"eval-{data.invocation_id}-{benchmark}"
                if log_mode == "per_task"
                else f"eval-{identifier}"
            )

        run_args = {
            "entity": self.config.get("entity"),
            "project": self.config.get("project"),
            "name": run_name,
            "group": self.config.get("group", data.invocation_id),
            "job_type": self.config.get("job_type", "evaluation"),
            "tags": self.config.get("tags"),
            "notes": self.config.get("description"),
        }

        # resume for multi_task runs
        if log_mode == "multi_task":
            stable_id = self.config.get("run_id") or identifier  # invocation_id
            run_args["id"] = stable_id
            run_args["resume"] = "allow"
        elif should_resume:
            run_args["id"] = existing_run_id
            run_args["resume"] = "allow"

        # Config metadata
        exec_type = (data.config or {}).get("execution", {}).get(
            "type"
        ) or data.executor
        run_config = {
            "invocation_id": data.invocation_id,
            "executor": exec_type,
        }

        if log_mode == "per_task":
            run_config["job_id"] = data.job_id
            run_config["harness"] = harness
            run_config["benchmark"] = benchmark

        if self.config.get("triggered_by_webhook"):
            run_config.update(
                {
                    "webhook_triggered": True,
                    "webhook_source": self.config.get("webhook_source"),
                    "source_artifact": self.config.get("source_artifact"),
                    "config_source": self.config.get("config_source"),
                }
            )

        run_config.update(self.config.get("extra_metadata", {}))
        run_args["config"] = run_config

        # Initialize
        run = wandb.init(**{k: v for k, v in run_args.items() if v is not None})

        # In multi_task, aggregate lists after init (no overwrite)
        if log_mode == "multi_task" and all_data:
            try:
                benchmarks = list(run.config.get("benchmarks", []))
                harnesses = list(run.config.get("harnesses", []))
                for d in all_data:
                    if d.task and d.task not in benchmarks:
                        benchmarks.append(d.task)
                    if d.harness and d.harness not in harnesses:
                        harnesses.append(d.harness)
                run.config.update(
                    {"benchmarks": benchmarks, "harnesses": harnesses},
                    allow_val_change=True,
                )
            except Exception:
                pass

        # Artifact naming
        artifact_name = (
            f"{data.invocation_id}_{benchmark}"
            if log_mode == "per_task"
            else data.invocation_id
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluation_result",
            description="Evaluation results",
            metadata={
                "invocation_id": data.invocation_id,
                "task": task_name,
                "benchmark": benchmark,
                "harness": harness,
            },
        )

        # Log artifacts from data
        logged_artifacts = self._log_artifacts(data, artifact)

        try:
            run.log_artifact(artifact)
            # charts for each logged metric
            try:
                for k in metrics.keys():
                    run.define_metric(k, summary="last")
            except Exception:
                pass

            # Log metrics with per-task step
            try:
                step_idx = int(data.job_id.split(".")[-1])
            except Exception:
                step_idx = 0
            run.log(metrics, step=step_idx)

            # metrics summary
            try:
                run.summary.update(metrics)
            except Exception:
                pass
        finally:
            try:
                run.finish()
            except Exception:
                pass

        return {
            "run_id": run.id,
            "run_url": run.url,
            "metrics_logged": len(metrics),
            "artifacts_logged": len(logged_artifacts),
        }
