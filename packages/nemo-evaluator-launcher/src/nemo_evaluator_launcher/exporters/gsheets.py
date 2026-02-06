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
from typing import Dict, List, Tuple

try:
    import gspread

    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import DataForExport


@register_exporter("gsheets")
class GSheetsExporter(BaseExporter):
    """Export accuracy metrics to Google Sheets with multi-invocation support."""

    def is_available(self) -> bool:
        return GSPREAD_AVAILABLE

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export jobs to Google Sheets."""
        if not self.is_available():
            logger.error(
                "Google Sheets package not installed. "
                "Install via: pip install nemo-evaluator-launcher[gsheets]"
            )
            return [], [data.job_id for data in data_for_export], []

        if not data_for_export:
            return [], [], []

        successful_jobs = []
        failed_jobs = []
        skipped_jobs = []

        # Collect metrics from all jobs BEFORE connecting to Google Sheets
        all_job_metrics = {}
        for data in data_for_export:
            if not data.metrics:
                logger.warning(f"No metrics found for job {data.job_id}, skipping")
                skipped_jobs.append(data.job_id)
                continue
            all_job_metrics[data.job_id] = data.metrics

        if not all_job_metrics:
            logger.warning("No jobs with metrics to export")
            return [], [], skipped_jobs

        try:
            # Connect to Google Sheets
            service_account_file = self.config.get("service_account_file")
            spreadsheet_name = self.config.get(
                "spreadsheet_name", "NeMo Evaluator Launcher Results"
            )

            if service_account_file:
                gc = gspread.service_account(
                    filename=os.path.expanduser(service_account_file)
                )
            else:
                gc = gspread.service_account()

            # Get or create spreadsheet
            spreadsheet_id = self.config.get("spreadsheet_id")
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

            # Get/update headers based on all extracted metrics
            headers = self._get_or_update_headers(worksheet, all_job_metrics)

            # Add rows for jobs with metrics
            for data in data_for_export:
                if data.job_id in all_job_metrics:
                    try:
                        row_data = self._prepare_row_data(
                            data, all_job_metrics[data.job_id], headers
                        )
                        worksheet.append_row(row_data)
                        successful_jobs.append(data.job_id)
                        logger.info(
                            f"Exported job {data.job_id} to Google Sheets: {sh.url}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to add row for job {data.job_id}: {e}")
                        failed_jobs.append(data.job_id)

        except Exception as e:
            logger.error(f"Google Sheets export failed: {e}")
            failed_jobs.extend(
                [
                    d.job_id
                    for d in data_for_export
                    if d.job_id not in successful_jobs and d.job_id not in skipped_jobs
                ]
            )

        return successful_jobs, failed_jobs, skipped_jobs

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
        self,
        data: DataForExport,
        accuracy_metrics: Dict[str, float],
        headers: List[str],
    ) -> List[str]:
        """Prepare row data dynamically."""

        task_name = data.task or "unknown"
        model_name = data.model_id or "unknown"

        row_data = []
        for header in headers:
            if header == "Model Name":
                row_data.append(model_name)
            elif header == "Task Name":
                row_data.append(task_name)
            elif header == "Invocation ID":
                row_data.append(data.invocation_id)
            elif header == "Job ID":
                row_data.append(data.job_id)
            elif header == "Executor":
                row_data.append(data.executor or "unknown")
            else:
                # Find metric with this clean name
                full_metric = f"{task_name}_{header}"
                value = accuracy_metrics.get(full_metric, "")
                row_data.append(str(value) if value else "")

        return row_data
