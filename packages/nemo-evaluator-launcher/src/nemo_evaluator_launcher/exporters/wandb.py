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
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter, ExportConfig
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import (
    DataForExport,
    get_available_artifacts,
    get_copytree_ignore,
)

LOG_MODES = ["per_task", "multi_task"]


class WandBExporterConfig(ExportConfig):
    """Configuration for WandBExporter."""

    entity: str
    project: str
    log_mode: Literal["per_task", "multi_task"] = Field(default="per_task")
    name: Optional[str] = Field(default=None)
    group: Optional[str] = Field(default=None)
    job_type: str = Field(default="evaluation")
    tags: Optional[List[str]] = Field(default=None)
    description: Optional[str] = Field(default=None)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = Field(default=None)


@register_exporter("wandb")
class WandBExporter(BaseExporter):
    """Export accuracy metrics to W&B.

    Config keys:
      entity (str): W&B entity/team name (required)
      project (str): W&B project name (required)
      log_mode (str): "per_task" (one run per task) or "multi_task" (one run per invocation) (default: "per_task")
      log_artifacts (bool): Whether to log artifacts to W&B (default: True)
      copy_logs (bool): Whether to log log files (default: False)
      only_required (bool): Log only required+optional artifacts (default: True)
      name (str): Custom run name (optional)
      group (str): Run group (default: invocation_id)
      job_type (str): W&B job type (default: "evaluation")
      tags (list): List of tags for the run
      description (str): Run description/notes
      extra_metadata (dict): Additional metadata to include in run config
    """

    config_class = WandBExporterConfig

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

        log_mode = self.config.log_mode
        if log_mode not in LOG_MODES:
            logger.error(f"Invalid log_mode: {log_mode}. Valid modes are: {LOG_MODES}")
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
                        run_info = self._create_wandb_run(
                            identifier=identifier,
                            metrics=data.metrics,
                            data=[data],
                        )
                        successful_jobs.append(data.job_id)
                        logger.info(
                            f"Exported job {data.job_id} to W&B: {run_info['run_url']}"
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
                    all_metrics = {}
                    try:
                        # Aggregate metrics from all jobs
                        # FIXME(martas): we potentially override metrics here
                        # if they have same name for different tasks

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
                        run_id = self._check_existing_run(
                            invocation_id, inv_data_list[0]
                        )

                        # Use first job data as template
                        result = self._create_wandb_run(
                            identifier=invocation_id,
                            metrics=all_metrics,
                            data=inv_data_list,
                            existing_run_id=run_id,
                        )
                        successful_jobs.extend([d.job_id for d in inv_data_list])
                        logger.info(
                            f"Exported invocation {invocation_id} to W&B: {result['run_url']}"
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

        try:
            artifacts_dir = data.artifacts_dir
            if not artifacts_dir.exists():
                logger.error(f"Artifacts directory {artifacts_dir} does not exist")
                return []
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

            if self.config.only_required and self.config.copy_artifacts:
                # Upload only specific required files
                for p in get_available_artifacts(artifacts_dir):
                    artifact.add_file(str(p), name=f"{artifact_root}/artifacts/{p}")
                    logged_names.append(p)
            elif self.config.copy_artifacts:
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

            if self.config.copy_logs and logs_dir:
                if not logs_dir.exists():
                    logger.error(f"Logs directory {logs_dir} does not exist")
                    return logged_names
                for p in logs_dir.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(logs_dir).as_posix()
                        artifact.add_file(str(p), name=f"{artifact_root}/logs/{rel}")
                        logged_names.append(f"logs/{rel}")

            return logged_names
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
            return []

    def _check_existing_run(self, identifier: str, data: DataForExport) -> str | None:
        """Check if run exists based on name patterns."""
        try:
            import wandb

            api = wandb.Api()
            entity = self.config.entity
            project = self.config.project
            if not (entity and project):
                logger.error(
                    "W&B requires 'entity' and 'project' to be configured. "
                    "Set export.wandb.entity and export.wandb.project fields in the config."
                )
                return None

            # Check explicit name first
            if self.config.name:
                runs = api.runs(f"{entity}/{project}")
                for run in runs:
                    if run.display_name == self.config.name:
                        return run.id

            # Check default pattern
            default_run_name = f"eval-{identifier}"
            runs = api.runs(f"{entity}/{project}")
            for run in runs:
                if run.display_name == default_run_name:
                    return run.id

            return None
        except Exception as e:
            logger.error(f"Error checking existing run {identifier}: {e}")
            return None

    def _create_wandb_run(
        self,
        identifier: str,
        metrics: Dict[str, float],
        data: List[DataForExport],
        existing_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or resume W&B run for job(s)."""
        log_mode = self.config.log_mode
        if log_mode == "per_task" and len(data) > 1:
            raise RuntimeError(
                "Log mode 'per_task' does not support multiple data items"
            )

        invocation_id = {job_data.invocation_id for job_data in data}
        if len(invocation_id) > 1:
            raise RuntimeError(
                f"Trying to create a W&B run with multiple invocation IDs: {list(invocation_id)}"
            )
        elif len(invocation_id) == 0:
            raise RuntimeError("No invocation ID found in data")
        invocation_id = invocation_id.pop()

        executor = {job_data.executor for job_data in data}
        executor = ", ".join(executor)

        if self.config.name:
            run_name = self.config.name
        else:
            run_name = (
                f"eval-{data[0].invocation_id}-{data[0].task}"
                if log_mode == "per_task"
                else f"eval-{identifier}"
            )

        run_args = {
            "entity": self.config.entity,
            "project": self.config.project,
            "name": run_name,
            "group": self.config.group or invocation_id,
            "job_type": self.config.job_type,
            "tags": self.config.tags,
            "notes": self.config.description,
        }

        # resume for multi_task runs
        if log_mode == "multi_task":
            stable_id = self.config.run_id or identifier  # invocation_id
            run_args["id"] = stable_id
            run_args["resume"] = "allow"
        elif existing_run_id:
            run_args["id"] = existing_run_id
            run_args["resume"] = "allow"

        # Config metadata
        run_config = {
            "invocation_id": invocation_id,
            "executor": executor,
        }

        if log_mode == "per_task":
            run_config["job_id"] = data[0].job_id
            run_config["harness"] = data[0].harness
            run_config["benchmark"] = data[0].task

        run_config.update(self.config.extra_metadata)
        run_args["config"] = run_config

        # Initialize
        run = wandb.init(**{k: v for k, v in run_args.items() if v is not None})

        # In multi_task, aggregate lists after init (no overwrite)
        if log_mode == "multi_task":
            try:
                benchmarks = list(run.config.get("benchmarks", []))
                harnesses = list(run.config.get("harnesses", []))
                for d in data:
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
            artifact_name = invocation_id
            artifact_metadata = {
                "invocation_id": invocation_id,
            }
            step_idx = 0
        else:
            artifact_name = f"{invocation_id}_{data[0].task}"
            artifact_metadata = {
                "invocation_id": invocation_id,
                "task": data[0].task,
                "benchmark": data[0].task,
                "harness": data[0].harness,
            }
            step_idx = int(data[0].job_id.split(".")[-1])
        # Artifact naming
        artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluation_result",
            description="Evaluation results",
            metadata=artifact_metadata,
        )

        # Log artifacts from data
        logged_artifacts = []
        for job_data in data:
            logged_artifacts.extend(self._log_artifacts(job_data, artifact))

        try:
            run.log_artifact(artifact)
            # charts for each logged metric
            try:
                for k in metrics.keys():
                    run.define_metric(k, summary="last")
            except Exception as e:
                logger.error(f"Error defining metric {k} for run {run.id}: {e}")

            # NOTE(martas): we use job ID (ie tasks order in the config)as step index here.
            # This doesn't look right but I don't want to change existing behavior
            # Log metrics with per-task step
            run.log(metrics, step=step_idx)

            # metrics summary
            try:
                run.summary.update(metrics)
            except Exception as e:
                logger.error(f"Error updating summary for run {run.id}: {e}")
        finally:
            try:
                run.finish()
            except Exception as e:
                logger.error(f"Error finishing run {run.id}: {e}")

        return {
            "run_id": run.id,
            "run_url": run.url,
            "artifacts_logged": len(logged_artifacts),
        }
