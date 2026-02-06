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
"""Export evaluation artifacts to local filesystem."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import DataForExport


@register_exporter("local")
class LocalExporter(BaseExporter):
    """Export all artifacts to local/remote filesystem with optional JSON/CSV summaries.

    Config keys:
      output_dir (str): Output directory for exported results (default: "./nemo-evaluator-launcher-results")
      copy_logs (bool): Whether to copy logs with artifacts (default: False)
      only_required (bool): Copy only required+optional artifacts (default: True)
      format (str or None): Summary format, one of None, "json", or "csv" (default: None; no summary, only original artifacts)
      log_metrics (list[str]): Filters for metric names; includes full metric name or substring pattern
      output_filename (str): Overrides default processed_results.json/csv filename
    """

    SUPPORTED_FORMATS = ["json", "csv"]

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export job artifacts to local directory."""
        success_jobs = []
        failed_jobs = []
        skipped_jobs = []

        output_dir = Path(
            self.config.get("output_dir", "./nemo-evaluator-launcher-results")
        )
        for data in data_for_export:
            job_export_dir = output_dir / data.invocation_id / data.job_id
            job_export_dir.mkdir(parents=True, exist_ok=True)

            try:
                per_job_skipped_jobs = self._write_summary([data], job_export_dir)
                if per_job_skipped_jobs:
                    skipped_jobs.extend(per_job_skipped_jobs)
                else:
                    success_jobs.append(data.job_id)

            except Exception as e:
                logger.error(f"Failed to export job {data.job_id}: {e}")
                failed_jobs.append(data.job_id)
                continue

        if len(data_for_export) > 1:
            # for muliple jobs, write a file with all results
            # TODO(martas): what should we do about skipped jobs when writing all results?
            _ = self._write_summary(data_for_export, output_dir)

        return success_jobs, failed_jobs, skipped_jobs

    # Summary JSON/CSV helpers
    def _write_summary(
        self, data_for_export: List[DataForExport], output_dir: Path
    ) -> List[str]:
        """Read per-job artifacts, extract metrics, and update invocation-level summary."""

        fmt = self.config.get("format")
        filename = self.config.get(
            "output_filename",
            f"processed_results.{fmt}",
        )
        out_path = (output_dir / filename).resolve()

        if fmt == "json":
            skipped_jobs = self._json_upsert(out_path, data_for_export)
        elif fmt == "csv":
            skipped_jobs = self._csv_upsert(out_path, data_for_export)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return skipped_jobs

    def _json_upsert(
        self,
        out_path: Path,
        data_for_export: List[DataForExport],
    ) -> None:
        if out_path.exists():
            merged_data = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            merged_data = {"benchmarks": {}}

        merged_data["export_timestamp"] = datetime.now().isoformat()

        skipped_jobs = []
        for data in data_for_export:
            benchmark_name = data.task
            model_name = data.model_id
            entry = {
                "invocation_id": data.invocation_id,
                "job_id": data.job_id,
                "harness": data.harness,
                "container": data.container,
                "scores": data.metrics,
                "timestamp": data.timestamp,
                "executor": data.executor,
            }
            if benchmark_name not in merged_data["benchmarks"]:
                # new benchmark
                merged_data["benchmarks"][benchmark_name] = {
                    "models": {model_name: [entry]}
                }
            elif model_name not in merged_data["benchmarks"][benchmark_name]["models"]:
                # new model
                merged_data["benchmarks"][benchmark_name]["models"][model_name] = [
                    entry
                ]
            elif any(
                e.get("job_id") == data.job_id
                for e in merged_data["benchmarks"][benchmark_name]["models"][model_name]
            ):
                # existing results for existing model
                # TODO(martas): consider adding validation if metrics are the same
                logger.debug(
                    f"Job {data.job_id} already exists for model {model_name} in benchmark {benchmark_name}. Skipping."
                )
                skipped_jobs.append(data.job_id)
                continue
            else:
                # new results for existing model
                merged_data["benchmarks"][benchmark_name]["models"][model_name].append(
                    entry
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(merged_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return skipped_jobs

    def _csv_upsert(
        self,
        out_path: Path,
        data_for_export: List[DataForExport],
    ) -> List[str]:
        import csv

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
        existing_rows = []
        existing_job_ids = set()
        existing_fieldnames = set(base_cols)
        # Read old results if exists
        if out_path.exists():
            with out_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.append(row)
                    existing_job_ids.add(row.get("Job ID"))
                    existing_fieldnames.update(row.keys())
            missing_cols = set(base_cols) - existing_fieldnames
            if missing_cols:
                logger.warning(
                    f"Columns {missing_cols} not found in old results, "
                    "which might indicate merging with results from a different format"
                )

        # Build new results
        new_rows = []
        metric_fieldnames = set()
        for data in data_for_export:
            if data.job_id in existing_job_ids:
                logger.debug(
                    f"Job {data.job_id} already exists in old results. Skipping."
                )
                skipped_jobs.append(data.job_id)
                continue
            # Flatten data for CSV
            row = {
                "Model Name": data.model_id,
                "Harness": data.harness,
                "Task Name": data.task,
                "Executor": data.executor,
                "Container": data.container,
                "Invocation ID": data.invocation_id,
                "Job ID": data.job_id,
            }
            # Add metrics (as columns)
            for k, v in (data.metrics or {}).items():
                row[k] = v
                metric_fieldnames.add(k)
            new_rows.append(row)

        # All columns established
        all_fieldnames = base_cols + sorted(
            (set(existing_fieldnames) | metric_fieldnames) - set(base_cols)
        )

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
            for row in new_rows:
                writer.writerow(row)

        return skipped_jobs
