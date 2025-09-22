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
"""Export evaluation results to specified target."""

from dataclasses import dataclass
from typing import Any, List, Optional

from simple_parsing import field


@dataclass
class ExportCmd:
    """Export evaluation results."""

    # Short usage examples will show up in -h as the class docstring:
    # Examples:
    #   nemo-evaluator-launcher export 8abcd123 --dest local --format json -o .
    #   nemo-evaluator-launcher export 8abcd123.0 9ef01234 --dest local --format csv -o results/ -fname processed_results.csv
    #   nemo-evaluator-launcher export 8abcd123 --dest jet

    invocation_ids: List[str] = field(
        positional=True,
        help="IDs to export (space-separated). Accepts invocation IDs (xxxxxxxx) and job IDs (xxxxxxxx.n); mixture of both allowed.",
    )
    dest: str = field(
        default="local",
        alias=["--dest"],
        choices=["local", "wandb", "mlflow", "gsheets", "jet"],
        help="Export destination.",
    )
    output_dir: Optional[str] = field(
        default=".",
        alias=["--output-dir", "-o"],
        help="Output directory (default: current directory).",
    )
    output_filename: Optional[str] = field(
        default=None,
        alias=["--output-filename", "-fname"],
        help="Summary filename (default: processed_results.json/csv based on --format).",
    )
    format: Optional[str] = field(
        default=None,
        alias=["--format"],
        choices=["json", "csv"],
        help="Summary format for --dest local. Omit to only copy artifacts.",
    )
    copy_logs: bool = field(
        default=False,
        alias=["--copy-logs"],
        help="Include logs when copying locally (default: False).",
    )
    log_metrics: List[str] = field(
        default_factory=list,
        alias=["--log-metrics"],
        help="Filter metrics by name (repeatable). Examples: score, f1, mmlu_score_micro.",
    )
    only_required: bool = field(
        default=True,
        alias=["--only-required"],
        help="Copy only required+optional artifacts (default: True). Set to False to copy all available artifacts.",
    )

    def execute(self) -> None:
        """Execute export."""
        # Import heavy dependencies only when needed
        from nemo_evaluator_launcher.api.functional import export_results

        config: dict[str, Any] = {
            "copy_logs": self.copy_logs,
            "only_required": self.only_required,
        }

        # Output handling
        if self.output_dir:
            config["output_dir"] = self.output_dir
        if self.output_filename:
            config["output_filename"] = self.output_filename

        # Format and filters
        if self.format:
            config["format"] = self.format
        if self.log_metrics:
            config["log_metrics"] = self.log_metrics

        if self.format and self.dest != "local":
            print(
                "Note: --format is only used by --dest local. It will be ignored for other destinations."
            )

        # Execute
        print(
            f"Exporting {len(self.invocation_ids)} {'invocations' if len(self.invocation_ids) > 1 else 'invocation'} to {self.dest}..."
        )

        result = export_results(self.invocation_ids, self.dest, config)

        if not result["success"]:
            print(f"Export failed: {result.get('error', 'Unknown error')}")
            return

        # Success path
        if len(self.invocation_ids) == 1:
            # Single invocation
            invocation_id = self.invocation_ids[0]
            print(f"Export completed for {invocation_id}")

            for job_id, job_result in result["jobs"].items():
                if job_result.get("success"):
                    print(f"  {job_id}: {job_result.get('message', '')}")
                    metadata = job_result.get("metadata", {})
                    if metadata.get("run_url"):
                        print(f"    URL: {metadata['run_url']}")
                    if metadata.get("summary_path"):
                        print(f"    Summary: {metadata['summary_path']}")
                else:
                    print(f"  {job_id} failed: {job_result.get('message', '')}")
        else:
            # Multiple invocations
            metadata = result.get("metadata", {})
            print(
                f"Export completed: {metadata.get('successful_invocations', 0)}/{metadata.get('total_invocations', 0)} successful"
            )

            # Show summary path if available
            if metadata.get("summary_path"):
                print(f"Summary: {metadata['summary_path']}")

            # Show per-invocation status
            for invocation_id, inv_result in result["invocations"].items():
                if inv_result.get("success"):
                    job_count = len(inv_result.get("jobs", {}))
                    print(f"  {invocation_id}: {job_count} jobs")
                else:
                    print(
                        f"  {invocation_id}: failed, {inv_result.get('error', 'Unknown error')}"
                    )
