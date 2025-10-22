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
"""Google Sheets evaluation results exporter."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

try:
    import gspread

    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter, ExportResult
from nemo_evaluator_launcher.exporters.local import LocalExporter
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import (
    extract_accuracy_metrics,
    extract_exporter_config,
    get_model_name,
    get_task_name,
)


@register_exporter("gsheets")
class GSheetsExporter(BaseExporter):
    """Export accuracy metrics to Google Sheets with multi-invocation support."""

    def supports_executor(self, executor_type: str) -> bool:
        return True

    def is_available(self) -> bool:
        return GSPREAD_AVAILABLE

    def _get_artifacts_locally(self, job_data: JobData) -> tuple[Path, str]:
        """Get artifacts locally using LocalExporter."""
        try:
            temp_dir = tempfile.mkdtemp(prefix="gsheets_")
            local_exporter = LocalExporter({"output_dir": temp_dir})
            local_result = local_exporter.export_job(job_data)

            if not local_result.success:
                logger.warning(f"LocalExporter failed: {local_result.message}")
                shutil.rmtree(temp_dir)
                return None, None

            artifacts_dir = Path(local_result.dest) / "artifacts"
            if not artifacts_dir.exists():
                logger.warning(f"No artifacts directory found in {local_result.dest}")
                shutil.rmtree(temp_dir)
                return None, None

            return artifacts_dir, temp_dir

        except Exception as e:
            logger.error(f"Failed to get artifacts locally: {e}")
            if "temp_dir" in locals() and temp_dir:
                shutil.rmtree(temp_dir)
            return None, None

    def export_invocation(self, invocation_id: str) -> Dict[str, Any]:
        """Export all jobs in an invocation to Google Sheets."""
        if not self.is_available():
            return {"success": False, "error": "gspread package not installed"}

        jobs = self.db.get_jobs(invocation_id)
        if not jobs:
            return {
                "success": False,
                "error": f"No jobs found for invocation {invocation_id}",
            }

        try:
            # Load exporter config from the first job (supports job-embedded config and CLI overrides)
            first_job = next(iter(jobs.values()))
            gsheets_config = extract_exporter_config(first_job, "gsheets", self.config)

            # Connect to Google Sheets
            service_account_file = gsheets_config.get("service_account_file")
            spreadsheet_name = gsheets_config.get(
                "spreadsheet_name", "NeMo Evaluator Launcher Results"
            )

            if service_account_file:
                gc = gspread.service_account(
                    filename=os.path.expanduser(service_account_file)
                )
            else:
                gc = gspread.service_account()

            # Get or create spreadsheet
            spreadsheet_id = gsheets_config.get("spreadsheet_id")
            try:
                if spreadsheet_id:
                    sh = gc.open_by_key(spreadsheet_id)
                else:
                    sh = gc.open(spreadsheet_name)
                logger.info(f"Opened existing spreadsheet: {spreadsheet_name}")
            except gspread.SpreadsheetNotFound:
                if spreadsheet_id:
                    raise  # Can't create with explicit ID
                sh = gc.create(spreadsheet_name)
                logger.info(f"Created new spreadsheet: {spreadsheet_name}")

            worksheet = sh.sheet1
            # Extract metrics from ALL jobs first to determine headers
            all_job_metrics = {}
            results = {}

            for job_id, job_data in jobs.items():
                try:
                    # Get artifacts locally first
                    artifacts_dir, temp_dir = self._get_artifacts_locally(job_data)
                    if not artifacts_dir:
                        results[job_id] = {
                            "success": False,
                            "message": "Failed to get artifacts locally",
                        }
                        all_job_metrics[job_id] = {}
                        continue

                    try:
                        # Extract metrics from local artifacts
                        accuracy_metrics = extract_accuracy_metrics(
                            job_data,
                            lambda jd: {
                                "artifacts_dir": artifacts_dir,
                                "storage_type": "local_filesystem",
                            },
                        )
                        all_job_metrics[job_id] = accuracy_metrics

                        if accuracy_metrics:
                            results[job_id] = {
                                "success": True,
                                "message": f"Extracted {len(accuracy_metrics)} metrics",
                                "metadata": {"metrics_count": len(accuracy_metrics)},
                            }
                        else:
                            results[job_id] = {
                                "success": False,
                                "message": "No accuracy metrics found",
                            }
                    finally:
                        if temp_dir:
                            shutil.rmtree(temp_dir)

                except Exception as e:
                    logger.error(f"Failed to extract metrics for job {job_id}: {e}")
                    results[job_id] = {
                        "success": False,
                        "message": f"Metric extraction failed: {str(e)}",
                    }
                    all_job_metrics[job_id] = {}

            # Get/update headers based on all extracted metrics
            headers = self._get_or_update_headers(worksheet, all_job_metrics)

            # Add rows for jobs with metrics
            rows_added = 0
            for job_id, job_data in jobs.items():
                if results[job_id]["success"]:
                    row_data = self._prepare_row_data(
                        job_data, all_job_metrics[job_id], headers
                    )
                    worksheet.append_row(row_data)
                    rows_added += 1

            return {
                "success": True,
                "invocation_id": invocation_id,
                "jobs": results,
                "metadata": {
                    "spreadsheet_name": spreadsheet_name,
                    "spreadsheet_url": sh.url,
                    "rows_added": rows_added,
                    "total_columns": len(headers),
                    "metric_columns": len(
                        [
                            h
                            for h in headers
                            if h
                            not in ["Timestamp", "Invocation ID", "Job ID", "Executor"]
                        ]
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Sheets export failed for invocation {invocation_id}: {e}")
            return {"success": False, "error": f"Sheets export failed: {str(e)}"}

    def export_job(self, job_data: JobData) -> ExportResult:
        """Export single job to Google Sheets."""
        if not self.is_available():
            return ExportResult(
                success=False, dest="gsheets", message="gspread package not installed"
            )

        try:
            # Extract config from job_data
            gsheets_config = extract_exporter_config(job_data, "gsheets", self.config)

            # Get artifacts locally first
            artifacts_dir, temp_dir = self._get_artifacts_locally(job_data)
            if not artifacts_dir:
                return ExportResult(
                    success=False,
                    dest="gsheets",
                    message="Failed to get artifacts locally",
                )

            try:
                # Connect to Google Sheets
                service_account_file = gsheets_config.get("service_account_file")
                spreadsheet_name = gsheets_config.get(
                    "spreadsheet_name", "NeMo Evaluator Launcher Results"
                )

                if service_account_file:
                    gc = gspread.service_account(
                        filename=os.path.expanduser(service_account_file)
                    )
                else:
                    gc = gspread.service_account()

                # Get or create spreadsheet
                spreadsheet_id = gsheets_config.get("spreadsheet_id")
                try:
                    if spreadsheet_id:
                        sh = gc.open_by_key(spreadsheet_id)
                    else:
                        sh = gc.open(spreadsheet_name)
                except gspread.SpreadsheetNotFound:
                    if spreadsheet_id:
                        raise  # Can't create with explicit ID
                    sh = gc.create(spreadsheet_name)

                worksheet = sh.sheet1

                # Extract metrics from local artifacts
                log_metrics = gsheets_config.get("log_metrics", [])
                accuracy_metrics = extract_accuracy_metrics(
                    job_data,
                    lambda jd: {
                        "artifacts_dir": artifacts_dir,
                        "storage_type": "local_filesystem",
                    },
                    log_metrics,
                )

                if not accuracy_metrics:
                    return ExportResult(
                        success=False,
                        dest="gsheets",
                        message="No accuracy metrics found",
                    )

                # Get/update headers for this job's metrics
                headers = self._get_or_update_headers(
                    worksheet, {job_data.job_id: accuracy_metrics}
                )

                # Prepare and add single row for this job
                row_data = self._prepare_row_data(job_data, accuracy_metrics, headers)
                worksheet.append_row(row_data)

                return ExportResult(
                    success=True,
                    dest="gsheets",
                    message=f"Added 1 row for job {job_data.job_id}",
                    metadata={
                        "spreadsheet_url": sh.url,
                        "job_id": job_data.job_id,
                        "metrics_logged": len(accuracy_metrics),
                    },
                )

            finally:
                if temp_dir:
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logger.error(f"GSheets export failed for job {job_data.job_id}: {e}")
            return ExportResult(
                success=False, dest="gsheets", message=f"Failed: {str(e)}"
            )

    def export_multiple_invocations(self, invocation_ids: List[str]) -> Dict[str, Any]:
        """Export multiple invocations to the same sheet."""
        if not self.is_available():
            return {"success": False, "error": "gspread package not installed"}

        all_results = {}
        total_rows_added = 0
        spreadsheet_url = None

        for invocation_id in invocation_ids:
            result = self.export_invocation(invocation_id)
            all_results[invocation_id] = result

            if result["success"]:
                total_rows_added += result.get("metadata", {}).get("rows_added", 0)
                if not spreadsheet_url:
                    spreadsheet_url = result.get("metadata", {}).get("spreadsheet_url")

        return {
            "success": True,
            "invocations": all_results,
            "metadata": {
                "total_invocations": len(invocation_ids),
                "total_rows_added": total_rows_added,
                "spreadsheet_url": spreadsheet_url,
                "spreadsheet_name": self.config.get(
                    "spreadsheet_name", "NeMo Evaluator Launcher Results"
                ),
            },
        }

    def _get_or_update_headers(
        self, worksheet, all_metrics: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Get existing headers or create/update them dynamically."""

        # Base columns
        base_headers = [
            "Model Name",
            "Task Name",
            "Invocation ID",
            "Job ID",
            "Executor",
        ]

        # Get all unique clean metric names (everything after first underscore)
        all_clean_metrics = set()
        for job_metrics in all_metrics.values():
            for full_name in job_metrics.keys():
                clean_name = (
                    full_name.split("_", 1)[1] if "_" in full_name else full_name
                )
                all_clean_metrics.add(clean_name)

        target_headers = base_headers + sorted(all_clean_metrics)

        # Handle sheet creation/updating
        existing_values = worksheet.get_all_values()
        if not existing_values:
            # Empty sheet - create headers
            worksheet.update("1:1", [target_headers])
            worksheet.format("1:1", {"textFormat": {"bold": True}})
            return target_headers
        else:
            # Sheet exists - just update the entire header row
            existing_headers = existing_values[0]
            new_metrics = [
                m for m in sorted(all_clean_metrics) if m not in existing_headers
            ]
            if new_metrics:
                updated_headers = existing_headers + new_metrics
                worksheet.update("1:1", [updated_headers])
                return updated_headers
            return existing_headers

    def _prepare_row_data(
        self, job_data: JobData, accuracy_metrics: Dict[str, float], headers: List[str]
    ) -> List[str]:
        """Prepare row data dynamically."""

        task_name = get_task_name(job_data)
        model_name = get_model_name(job_data)

        row_data = []
        for header in headers:
            if header == "Model Name":
                row_data.append(model_name)
            elif header == "Task Name":
                row_data.append(task_name)
            elif header == "Invocation ID":
                row_data.append(job_data.invocation_id)
            elif header == "Job ID":
                row_data.append(job_data.job_id)
            elif header == "Executor":
                row_data.append(job_data.executor)
            else:
                # Find metric with this clean name
                full_metric = f"{task_name}_{header}"
                value = accuracy_metrics.get(full_metric, "")
                row_data.append(str(value) if value else "")

        return row_data
