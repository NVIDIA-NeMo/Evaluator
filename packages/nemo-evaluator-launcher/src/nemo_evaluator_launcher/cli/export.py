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
from typing import List, Optional

from omegaconf import OmegaConf
from simple_parsing import field

from nemo_evaluator_launcher.exporters.config import (
    apply_export_overrides,
    load_export_config_from_file,
)


@dataclass
class ExportCmd:
    """Export evaluation results."""

    # Short usage examples will show up in -h as the class docstring:
    # Examples:
    #   nemo-evaluator-launcher export 8abcd123 --dest local --format json --out .
    #   nemo-evaluator-launcher export 8abcd123.0 9ef01234 --dest local --format csv --out results/ -fname processed_results.csv
    #   nemo-evaluator-launcher export 8abcd123 --dest jet
    #   nemo-evaluator-launcher export 8abcd123 --config export_config.yaml --dest wandb

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
    config: Optional[str] = field(
        default=None,
        alias=["--config"],
        help="Path to export config file. The config should contain exporter settings (e.g., output_dir, format, copy_logs). "
        "CLI arguments override config file values.",
    )

    # overrides for exporter config; use -o similar to run command
    override: List[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-o", "--override"],
        help="Hydra-style overrides for exporter config. Use `export.<dest>.key=value` (e.g., -o export.wandb.entity=org-name).",
    )
    output_dir: Optional[str] = field(
        default=None,
        alias=["--output-dir", "-out"],
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
    copy_logs: Optional[bool] = field(
        default=None,
        alias=["--copy-logs"],
        help="Export log files (if exporter allows it) (default: False).",
    )
    copy_artifacts: Optional[bool] = field(
        default=None,
        alias=["--copy-artifacts"],
        help="Export artifact files (if exporter allows it) (default: True).",
    )
    log_metrics: Optional[List[str]] = field(
        default=None,
        alias=["--log-metrics"],
        help="Filter metrics by name (repeatable). Examples: score, f1, mmlu_score_micro.",
    )
    only_required: Optional[bool] = field(
        default=None,
        alias=["--only-required"],
        help="Copy only required artifacts. Set to False to copy all available artifacts. "
        "This flag is ignored if --copy-artifacts is False.",
    )

    job_dirs: Optional[List[str]] = field(
        default=None,
        alias=["--job-dirs"],
        help="Directories used to search for job artifacts and logs. Allows to export results for jobs not stored in the "
        "local database (e.g. during auto-export on a remote machine or executed by a different user). "
        "If privided, it is used to search for sub-directories with results for specified invocation ID "
        "and the jobs information is inferred from the metadata stored in the job artifacts directories. "
        "It should match the value passed as execution.output_dir in the run config. "
        "Can be specified multiple times to search in multiple directories.",
    )

    def execute(self) -> None:
        """Execute export."""
        # Import heavy dependencies only when needed

        from nemo_evaluator_launcher.api.functional import export_results

        # Validation: ensure IDs are provided
        if not self.invocation_ids:
            print("Error: No IDs provided. Specify one or more invocation or job IDs.")
            print(
                "Usage: nemo-evaluator-launcher export <id> [<id>...] --dest <destination>"
            )
            return

        # Load configuration from file if provided
        if self.config:
            config = load_export_config_from_file(self.config, self.dest)
        else:
            config = OmegaConf.create({})

        # CLI arguments override config file values
        # We default to None to avoid overriding the config file values if not explicitly passed via CLI
        if self.copy_logs is not None:
            config["copy_logs"] = self.copy_logs
        if self.copy_artifacts is not None:
            config["copy_artifacts"] = self.copy_artifacts
        if self.only_required is not None:
            config["only_required"] = self.only_required

        # Output handling
        if self.output_dir is not None:
            config["output_dir"] = self.output_dir
        if self.output_filename is not None:
            config["output_filename"] = self.output_filename

        # Format and filters
        if self.format and self.dest == "local":
            config["format"] = self.format

        if self.log_metrics is not None:
            config["log_metrics"] = self.log_metrics

        # Add job_dirs if explicitly passed via CLI
        if self.job_dirs is not None:
            config["job_dirs"] = self.job_dirs

        # Parse and validate overrides
        apply_export_overrides(config, self.dest, self.override)

        if self.format and self.dest != "local":
            print(
                "Note: --format is only used by --dest local. It will be ignored for other destinations."
            )

        print(
            f"Exporting {len(self.invocation_ids)} {'invocations' if len(self.invocation_ids) > 1 else 'invocation'} to {self.dest}..."
        )

        result = export_results(
            self.invocation_ids, self.dest, OmegaConf.to_container(config)
        )

        # Success path
        print(f"Export completed: {result['metadata']}")
        if not result.get("success", False):
            print("Some jobs failed to export. See logs above for more details.")
            return
