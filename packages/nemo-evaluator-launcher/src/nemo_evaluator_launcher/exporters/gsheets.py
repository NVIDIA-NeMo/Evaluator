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
from typing import List, Tuple

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
            logger.warning("No data for export")
            return [], [], []

        successful_jobs = []
        failed_jobs = []
        skipped_jobs = []

        try:
            # Connect to Google Sheets
            service_account_file = self.config.get("service_account_file")

            if service_account_file:
                gc = gspread.service_account(
                    filename=os.path.expanduser(service_account_file)
                )
            else:
                gc = gspread.service_account()

            # Get or create spreadsheet
            spreadsheet_id = self.config.get("spreadsheet_id")
            if spreadsheet_id:
                # NOTE(martas): we don't try-except here because if user-provided spreadsheet_id is invalid
                # we want to raise an exception and fail the export
                sh = gc.open_by_key(spreadsheet_id)
                spreadsheet_name = sh.title
                logger.info(
                    "Opened existing spreadsheet",
                    spreadsheet_id=spreadsheet_id,
                    spreadsheet_name=spreadsheet_name,
                )
            else:
                spreadsheet_name = self.config.get(
                    "spreadsheet_name", "NeMo Evaluator Launcher Results"
                )

                try:
                    sh = gc.open(spreadsheet_name)
                    spreadsheet_id = sh.id
                    logger.info(
                        "Opened existing spreadsheet",
                        spreadsheet_id=spreadsheet_id,
                        spreadsheet_name=spreadsheet_name,
                    )
                except gspread.SpreadsheetNotFound:
                    sh = gc.create(spreadsheet_name)
                    spreadsheet_id = sh.id
                    logger.info(
                        "Created new spreadsheet",
                        spreadsheet_id=spreadsheet_id,
                        spreadsheet_name=spreadsheet_name,
                    )

            worksheet = sh.sheet1

            skipped_jobs.extend(self._update_worksheet(worksheet, data_for_export))
            successful_jobs.extend(
                [d.job_id for d in data_for_export if d.job_id not in skipped_jobs]
            )

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

    def _update_worksheet(
        self, worksheet, data_for_export: List[DataForExport]
    ) -> List[str]:
        """Get existing headers or create/update them dynamically."""

        # Base columns
        # FIXME(martas): the logic for base_cols and for merging with existing results duplicates local.py
        base_cols = [
            "Model Name",
            "Harness",
            "Task Name",
            "Executor",
            "Container",
            "Invocation ID",
            "Job ID",
        ]

        skipped_jobs = []
        all_rows = []
        existing_job_ids = set()
        header_fieldnames = set(base_cols)
        header, *rows = worksheet.get_all_values()
        if len(header) > 0 and len(rows) == 0:
            logger.warning(
                "Ignoring existing header because no rows found", header=header
            )
            header = []
        if rows:
            missing_cols = set(base_cols) - set(header)
            if missing_cols:
                logger.warning(
                    f"Columns {missing_cols} not found in old results, "
                    "which might indicate merging with results from a different format"
                )
            header_fieldnames.update(set(header))
            for row in rows:
                row = dict(zip(header, row))
                all_rows.append(row)
                existing_job_ids.add(row.get("Job ID"))

        # Build new results
        for data in data_for_export:
            if data.job_id in existing_job_ids:
                logger.debug(
                    f"Job {data.job_id} already exists in old results. Skipping."
                )
                skipped_jobs.append(data.job_id)
                continue
            row = {
                "Model Name": data.model_id,
                "Harness": data.harness,
                "Task Name": data.task,
                "Executor": data.executor,
                "Container": data.container,
                "Invocation ID": data.invocation_id,
                "Job ID": data.job_id,
                **(data.metrics or {}),
            }
            header_fieldnames.update(set(row.keys()))
            all_rows.append(row)

        # All columns established
        new_headers = base_cols + sorted(header_fieldnames - set(base_cols))
        prepared_content = [new_headers]
        for row in all_rows:
            prepared_row = []
            for header in new_headers:
                prepared_row.append(row.get(header, ""))
            prepared_content.append(prepared_row)
        worksheet.update(prepared_content)
        return skipped_jobs
