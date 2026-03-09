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
"""Export evaluation results to Hugging Face Hub leaderboards.

Pushes results as `.eval_results/` YAML files to model repos on the Hub,
enabling automatic leaderboard integration for benchmarks that define an
`eval.yaml` with `evaluation_framework: nemo-evaluator`.

See: https://huggingface.co/docs/hub/eval-results
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import Field, model_validator

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter, ExportConfig
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import DataForExport


class HuggingFaceExporterConfig(ExportConfig):
    """Configuration for HuggingFaceExporter."""

    dataset_id: str = Field(
        ...,
        description="HF Hub dataset ID of the benchmark (e.g., 'nvidia/compute-eval'). "
        "Must have an eval.yaml defining it as a benchmark.",
    )
    task_id: str = Field(
        ...,
        description="Task ID as defined in the benchmark's eval.yaml (e.g., 'compute_eval').",
    )
    model_repo: str = Field(
        ...,
        description="HF Hub model repo to push results to (e.g., 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF').",
    )
    metric_name: Optional[str] = Field(
        default=None,
        description="Metric key from results.yml to use as the score value. "
        "If not set, uses the first metric found. Supports substring matching.",
    )
    token: Optional[str] = Field(
        default=None,
        description="HF Hub token. Defaults to HF_TOKEN env var or cached login.",
    )
    create_pr: bool = Field(
        default=True,
        description="If True, submit results as a Pull Request (community contribution). "
        "If False, push directly to main branch (requires write access).",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes about the evaluation setup "
        "(e.g., 'Pass@1, k=1', 'chain-of-thought').",
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Optional URL linking to evaluation traces or logs.",
    )
    source_name: Optional[str] = Field(
        default=None,
        description="Optional display name for the source link.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_token(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("token"):
            data["token"] = os.environ.get("HF_TOKEN")
        return data


def _sanitize_filename(name: str) -> str:
    """Convert a dataset ID or string into a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _select_metric(
    metrics: Dict[str, float], metric_name: Optional[str]
) -> Tuple[str, float]:
    """Select a metric from the results dict.

    If metric_name is provided, finds the first key containing that substring.
    Otherwise, returns the first metric.

    Returns:
        Tuple of (metric_key, metric_value).

    Raises:
        ValueError: If no matching metric is found or metrics dict is empty.
    """
    if not metrics:
        raise ValueError("No metrics found in evaluation results")

    if metric_name is None:
        key = next(iter(metrics))
        return key, metrics[key]

    for key, value in metrics.items():
        if metric_name in key:
            return key, value

    raise ValueError(
        f"Metric '{metric_name}' not found in results. "
        f"Available metrics: {list(metrics.keys())}"
    )


def _build_eval_result_entry(
    dataset_id: str,
    task_id: str,
    value: float,
    date: str,
    notes: Optional[str] = None,
    source_url: Optional[str] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single .eval_results/ entry conforming to the HF spec."""
    entry: Dict[str, Any] = {
        "dataset": {
            "id": dataset_id,
            "task_id": task_id,
        },
        "value": value,
        "date": date,
    }

    if source_url:
        source: Dict[str, str] = {"url": source_url}
        if source_name:
            source["name"] = source_name
        entry["source"] = source

    if notes:
        entry["notes"] = notes

    return entry


def _format_eval_results_yaml(entries: List[Dict[str, Any]]) -> str:
    """Serialize eval result entries to YAML matching the HF spec."""
    return yaml.dump(
        entries,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


@register_exporter("huggingface")
class HuggingFaceExporter(BaseExporter):
    """Export evaluation results to Hugging Face Hub model repos.

    Results are formatted as `.eval_results/` YAML files conforming to the
    HF Evaluation Results specification. The Hub automatically aggregates
    these into benchmark leaderboards.

    Config keys:
        dataset_id (str): HF benchmark dataset ID (e.g., 'nvidia/compute-eval')
        task_id (str): Task ID from the benchmark's eval.yaml
        model_repo (str): HF model repo to push results to
        metric_name (str): Metric key from results.yml to use as score
        token (str): HF Hub API token (defaults to HF_TOKEN env var)
        create_pr (bool): Submit as PR (default True) vs direct push
        notes (str): Optional evaluation setup details
        source_url (str): Optional link to evaluation traces
        source_name (str): Optional display name for source link
    """

    config_class = HuggingFaceExporterConfig

    def export_jobs(
        self, data_for_export: List[DataForExport]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Export job results to HF Hub as .eval_results/ YAML files."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.error(
                "huggingface_hub is required for the HuggingFace exporter. "
                "Install it with: pip install huggingface_hub"
            )
            return [], [d.job_id for d in data_for_export], []

        api = HfApi(token=self.config.token)

        success_jobs = []
        failed_jobs = []
        skipped_jobs = []

        entries = []
        job_ids_for_entries = []

        for data in data_for_export:
            try:
                metric_key, metric_value = _select_metric(
                    data.metrics, self.config.metric_name
                )
                logger.info(
                    f"Job {data.job_id}: selected metric '{metric_key}' = {metric_value}"
                )

                eval_date = datetime.fromtimestamp(
                    data.timestamp, tz=timezone.utc
                ).strftime("%Y-%m-%d")

                notes_parts = []
                if self.config.notes:
                    notes_parts.append(self.config.notes)
                if data.container:
                    notes_parts.append(f"container: {data.container}")
                notes = ", ".join(notes_parts) if notes_parts else None

                entry = _build_eval_result_entry(
                    dataset_id=self.config.dataset_id,
                    task_id=self.config.task_id,
                    value=round(metric_value, 4),
                    date=eval_date,
                    notes=notes,
                    source_url=self.config.source_url,
                    source_name=self.config.source_name,
                )
                entries.append(entry)
                job_ids_for_entries.append(data.job_id)

            except Exception as e:
                logger.error(f"Failed to prepare result for job {data.job_id}: {e}")
                failed_jobs.append(data.job_id)

        if not entries:
            return success_jobs, failed_jobs, skipped_jobs

        yaml_content = _format_eval_results_yaml(entries)
        filename = f".eval_results/{_sanitize_filename(self.config.dataset_id)}.yaml"

        commit_message = (
            f"Add {self.config.task_id} evaluation results (via NeMo Evaluator)"
        )

        try:
            api.upload_file(
                path_or_fileobj=yaml_content.encode("utf-8"),
                path_in_repo=filename,
                repo_id=self.config.model_repo,
                repo_type="model",
                commit_message=commit_message,
                create_pr=self.config.create_pr,
            )

            action = "PR submitted" if self.config.create_pr else "pushed"
            logger.info(
                f"Results {action} to {self.config.model_repo}/{filename} "
                f"({len(entries)} entries)"
            )
            success_jobs.extend(job_ids_for_entries)

        except Exception as e:
            logger.error(f"Failed to upload results to {self.config.model_repo}: {e}")
            failed_jobs.extend(job_ids_for_entries)

        return success_jobs, failed_jobs, skipped_jobs
