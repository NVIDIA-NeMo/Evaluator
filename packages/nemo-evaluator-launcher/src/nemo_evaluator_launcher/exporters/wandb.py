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

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
)


@register_exporter("wandb")
class WandBExporter(BaseExporter):
    """Export accuracy metrics to W&B."""

    def supports_executor(self, executor_type: str) -> bool:
        return True

    def is_available(self) -> bool:
        return WANDB_AVAILABLE

    def export_job(self, job_data: JobData) -> ExportResult:
        """Export single job - same logic as invocation but for one job."""
        if not self.is_available():
            return ExportResult(
                success=False, dest="wandb", message="wandb package not installed"
            )

        try:
            wandb_config = extract_exporter_config(job_data, "wandb", self.config)
            log_mode = wandb_config.get(
                "log_mode", "per_task"
            )  # Default per_task for immediate export

            # Stage artifacts locally if remote_ssh (e.g., Slurm), so we can extract metrics
            staged_base_dir = None
            try:
                paths = self.get_job_paths(job_data)
                if paths.get("storage_type") == "remote_ssh":
                    tmp_stage = Path(tempfile.mkdtemp(prefix="wandb_stage_"))
                    LocalExporter(
                        {
                            "output_dir": str(tmp_stage),
                            "copy_logs": wandb_config.get("log_logs", False),
                            "only_required": wandb_config.get("only_required", True),
                        }
                    ).export_job(job_data)
                    staged_base_dir = (
                        tmp_stage / job_data.invocation_id / job_data.job_id
                    )
            except Exception as e:
                logger.warning(f"W&B: staging failed for {job_data.job_id}: {e}")

            # Metrics (prefer staged if available)
            log_metrics = wandb_config.get("log_metrics", [])
            if staged_base_dir and (staged_base_dir / "artifacts").exists():
                metrics = extract_accuracy_metrics(
                    job_data,
                    lambda _: {
                        "artifacts_dir": staged_base_dir / "artifacts",
                        "storage_type": "local_filesystem",
                    },
                    log_metrics,
                )
            else:
                metrics = extract_accuracy_metrics(
                    job_data, self.get_job_paths, log_metrics
                )

            if not metrics:
                return ExportResult(
                    success=False, dest="wandb", message="No metrics found"
                )

            # Choose either jobId or invocationId based on log_mode
            if log_mode == "per_task":
                # Create separate run per task
                task_name = get_task_name(job_data)
                identifier = f"{job_data.invocation_id}-{task_name}"
                should_resume = False
                run_id = None
            elif log_mode == "multi_task":
                # Append to shared run by invocation_id
                identifier = job_data.invocation_id
                should_resume, run_id = self._check_existing_run(
                    identifier, job_data, wandb_config
                )
            result = self._create_wandb_run(
                identifier, wandb_config, metrics, job_data, should_resume, run_id
            )
            return ExportResult(
                success=True, dest="wandb", message="Export completed", metadata=result
            )

        except Exception as e:
            logger.error(f"W&B export failed: {e}")
            return ExportResult(
                success=False, dest="wandb", message=f"Failed: {str(e)}"
            )

    def export_invocation(self, invocation_id: str) -> Dict[str, Any]:
        """Export all jobs in invocation as one W&B run."""
        if not self.is_available():
            return {"success": False, "error": "wandb package not installed"}

        jobs = self.db.get_jobs(invocation_id)
        if not jobs:
            return {
                "success": False,
                "error": f"No jobs found for invocation {invocation_id}",
            }

        try:
            first_job = list(jobs.values())[0]
            wandb_config = extract_exporter_config(first_job, "wandb", self.config)

            all_metrics = {}
            for _, job_data in jobs.items():
                log_metrics = wandb_config.get("log_metrics", [])
                job_metrics = extract_accuracy_metrics(
                    job_data, self.get_job_paths, log_metrics
                )
                all_metrics.update(job_metrics)

            if not all_metrics:
                return {
                    "success": False,
                    "error": "No accuracy metrics found in any job",
                }

            should_resume, run_id = self._check_existing_run(
                invocation_id, first_job, wandb_config
            )

            result = self._create_wandb_run(
                invocation_id,
                wandb_config,
                all_metrics,
                first_job,
                should_resume,
                run_id,
            )

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
                "metadata": result,
            }

        except Exception as e:
            logger.error(f"W&B export failed for invocation {invocation_id}: {e}")
            return {"success": False, "error": f"W&B export failed: {str(e)}"}

    def _log_artifacts(
        self,
        job_data: JobData,
        wandb_config: Dict[str, Any],
        artifact,
        register_staging_dir=None,
    ) -> List[str]:
        """Log evaluation artifacts to WandB using LocalExporter for staging."""
        if not wandb_config.get("log_artifacts", True):
            return []
        try:
            temp_dir = tempfile.mkdtemp(prefix="wandb_artifacts_")
            if callable(register_staging_dir):
                register_staging_dir(temp_dir)
            local_exporter = LocalExporter(
                {
                    "output_dir": temp_dir,
                    "copy_logs": wandb_config.get(
                        "log_logs", wandb_config.get("copy_logs", False)
                    ),
                    "only_required": wandb_config.get("only_required", True),
                    "format": wandb_config.get("format"),
                    "log_metrics": wandb_config.get("log_metrics", []),
                    "output_filename": wandb_config.get("output_filename"),
                }
            )
            local_result = local_exporter.export_job(job_data)

            if not local_result.success:
                logger.error(f"Failed to download artifacts: {local_result.message}")
                return []

            base_dir = Path(local_result.dest)
            artifacts_dir = base_dir / "artifacts"
            logs_dir = base_dir / "logs"
            logged_names: list[str] = []

            artifact_root = get_artifact_root(job_data)  # "<harness>.<benchmark>"

            # Add config file only when artifacts logging is enabled
            if wandb_config.get("log_artifacts", True):
                cfg_added = False
                for fname in ("config.yml", "run_config.yml"):
                    p = artifacts_dir / fname
                    if p.exists():
                        artifact.add_file(str(p), name=f"{artifact_root}/{fname}")
                        logged_names.append(fname)
                        cfg_added = True
                        break
                if not cfg_added:
                    with tempfile.NamedTemporaryFile(
                        "w", suffix=".yaml", delete=False
                    ) as tmp_cfg:
                        yaml.dump(
                            job_data.config or {},
                            tmp_cfg,
                            default_flow_style=False,
                            sort_keys=False,
                        )
                        cfg_path = tmp_cfg.name
                    artifact.add_file(cfg_path, name=f"{artifact_root}/config.yaml")
                    os.unlink(cfg_path)
                    logged_names.append("config.yaml")

            files_to_upload: list[Path] = []
            if wandb_config.get("only_required", True):
                for fname in get_available_artifacts(artifacts_dir):
                    p = artifacts_dir / fname
                    if p.exists():
                        files_to_upload.append(p)
            else:
                for p in artifacts_dir.iterdir():
                    if p.is_file():
                        files_to_upload.append(p)

            for fpath in files_to_upload:
                rel = fpath.relative_to(artifacts_dir).as_posix()
                artifact.add_file(str(fpath), name=f"{artifact_root}/artifacts/{rel}")
                logged_names.append(rel)

            if wandb_config.get("log_logs", False) and logs_dir.exists():
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
        self, identifier: str, job_data: JobData, config: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Check if run exists based on webhook metadata then name patterns."""
        try:
            import wandb

            api = wandb.Api()
            entity = config.get("entity")
            project = config.get("project")
            if not (entity and project):
                return False, None

            # Check webhook metadata for run_id first
            webhook_meta = job_data.data.get("webhook_metadata", {})
            if (
                webhook_meta.get("webhook_source") == "wandb"
                and config.get("triggered_by_webhook")
                and "run_id" in webhook_meta
            ):
                try:
                    # Verify the run actually exists
                    run = api.run(f"{entity}/{project}/{webhook_meta['run_id']}")
                    return True, run.id
                except Exception:
                    pass

            # Check explicit name first
            if config.get("name"):
                runs = api.runs(f"{entity}/{project}")
                for run in runs:
                    if run.display_name == config["name"]:
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
        config: Dict[str, Any],
        metrics: Dict[str, float],
        job_data: JobData,
        should_resume: bool,
        existing_run_id: str,
    ) -> Dict[str, Any]:
        """Create or resume W&B run for single job."""
        log_mode = config.get("log_mode", "per_task")
        task_name = get_task_name(job_data)
        bench_info = get_benchmark_info(job_data)
        benchmark = bench_info.get("benchmark", task_name)
        harness = bench_info.get("harness", "unknown")

        if config.get("name"):
            run_name = config["name"]
        else:
            run_name = (
                f"eval-{job_data.invocation_id}-{benchmark}"
                if log_mode == "per_task"
                else f"eval-{identifier}"
            )

        run_args = {
            "entity": config.get("entity"),
            "project": config.get("project"),
            "name": run_name,
            "group": config.get("group", job_data.invocation_id),
            "job_type": config.get("job_type", "evaluation"),
            "tags": config.get("tags"),
            "notes": config.get("description"),
        }

        # resume for multi_task runs
        if log_mode == "multi_task":
            stable_id = config.get("run_id") or identifier  # invocation_id
            run_args["id"] = stable_id
            run_args["resume"] = "allow"
        elif should_resume:
            run_args["id"] = existing_run_id
            run_args["resume"] = "allow"

        # Config metadata
        exec_type = (job_data.config or {}).get("execution", {}).get(
            "type"
        ) or job_data.executor
        run_config = {
            "invocation_id": job_data.invocation_id,
            "executor": exec_type,
        }

        if log_mode == "per_task":
            run_config["job_id"] = job_data.job_id
            run_config["harness"] = harness
            run_config["benchmark"] = benchmark

        if config.get("triggered_by_webhook"):
            run_config.update(
                {
                    "webhook_triggered": True,
                    "webhook_source": config.get("webhook_source"),
                    "source_artifact": config.get("source_artifact"),
                    "config_source": config.get("config_source"),
                }
            )

        run_config.update(config.get("extra_metadata", {}))
        run_args["config"] = run_config

        # Initialize
        run = wandb.init(**{k: v for k, v in run_args.items() if v is not None})

        # Track staging dirs for this run
        staging_dirs: List[str] = []

        def register_staging_dir(path: str) -> None:
            if path and os.path.isdir(path):
                staging_dirs.append(path)

        # In multi_task, aggregate lists after init (no overwrite)
        if log_mode == "multi_task":
            try:
                benchmarks = list(run.config.get("benchmarks", []))
                if benchmark not in benchmarks:
                    benchmarks.append(benchmark)
                harnesses = list(run.config.get("harnesses", []))
                if harness not in harnesses:
                    harnesses.append(harness)
                run.config.update(
                    {"benchmarks": benchmarks, "harnesses": harnesses},
                    allow_val_change=True,
                )
            except Exception:
                pass

        # Artifact naming
        artifact_name = (
            f"{job_data.invocation_id}_{benchmark}"
            if log_mode == "per_task"
            else job_data.invocation_id
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluation_result",
            description="Evaluation results",
            metadata={
                "invocation_id": job_data.invocation_id,
                "task": task_name,
                "benchmark": benchmark,
                "harness": harness,
            },
        )

        logged_artifacts = self._log_artifacts(
            job_data, config, artifact, register_staging_dir=register_staging_dir
        )

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
                step_idx = int(job_data.job_id.split(".")[-1])
            except Exception:
                step_idx = 0
            run.log(metrics, step=step_idx)

            # metrics summary
            try:
                run.summary.update(metrics)
            except Exception:
                pass
        finally:
            for d in staging_dirs:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
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
