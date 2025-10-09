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
    #   nemo-evaluator-launcher export 8abcd123 --dest local --format json --out .
    #   nemo-evaluator-launcher export 8abcd123.0 9ef01234 --dest local --format csv --out results/ -fname processed_results.csv
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
    # overrides for exporter config; use -o similar to run command
    override: List[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-o", "--override"],
        help="Hydra-style overrides for exporter config. Use `export.<dest>.key=value` (e.g., -o export.wandb.entity=org-name).",
    )
    output_dir: Optional[str] = field(
        default=".",
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
    only_required: Optional[bool] = field(
        default=None,
        alias=["--only-required"],
        help="Copy only required+optional artifacts (default: True). Set to False to copy all available artifacts.",
    )

    def execute(self) -> None:
        """Execute export."""
        # Import heavy dependencies only when needed
        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.api.functional import export_results

        # Validation: ensure IDs are provided
        if not self.invocation_ids:
            print("Error: No IDs provided. Specify one or more invocation or job IDs.")
            print(
                "Usage: nemo-evaluator-launcher export <id> [<id>...] --dest <destination>"
            )
            return

        config: dict[str, Any] = {
            "copy_logs": self.copy_logs,
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

        # Add only_required if explicitly passed via CLI
        if self.only_required is not None:
            config["only_required"] = self.only_required

        # Parse and validate overrides
        if self.override:
            # Flatten possible list-of-lists from parser
            flat_overrides: list[str] = []
            for item in self.override:
                if isinstance(item, list):
                    flat_overrides.extend(str(x) for x in item)
                else:
                    flat_overrides.append(str(item))

            try:
                self._validate_overrides(flat_overrides, self.dest)
            except ValueError as e:
                print(f"Error: {e}")
                return

            # Expand env vars in override vals ($VAR / ${VAR})
            import os

            from omegaconf import OmegaConf

            expanded_overrides: list[str] = []
            for ov in flat_overrides:
                if "=" in ov:
                    k, v = ov.split("=", 1)
                    expanded_overrides.append(f"{k}={os.path.expandvars(v)}")
                else:
                    expanded_overrides.append(os.path.expandvars(ov))

            dot_cfg = OmegaConf.from_dotlist(expanded_overrides)
            as_dict = OmegaConf.to_container(dot_cfg, resolve=True) or {}
            if isinstance(as_dict, dict) and "export" in as_dict:
                export_map = as_dict.get("export") or {}
                if isinstance(export_map, dict) and self.dest in export_map:
                    config.update(export_map[self.dest] or {})
                else:
                    config.update(as_dict)
            else:
                config.update(as_dict)

        if self.format and self.dest != "local":
            print(
                "Note: --format is only used by --dest local. It will be ignored for other destinations."
            )

        if "only_required" in config and self.only_required is True:
            config.pop("only_required", None)

        print(
            f"Exporting {len(self.invocation_ids)} {'invocations' if len(self.invocation_ids) > 1 else 'invocation'} to {self.dest}..."
        )

        result = export_results(self.invocation_ids, self.dest, config)

        if not result.get("success", False):
            err = result.get("error", "Unknown error")
            print(f"\nExport failed: {err}")
            # Provide actionable guidance for common configuration issues
            if self.dest == "mlflow":
                if "tracking_uri" in str(err).lower():
                    print("\nMLflow requires 'tracking_uri' to be configured.")
                    print(
                        "Set it via: -o export.mlflow.tracking_uri=http://mlflow-server:5000"
                    )
                elif "not installed" in str(err).lower():
                    print("\nMLflow package not installed.")
                    print("Install via: pip install nemo-evaluator-launcher[mlflow]")
            elif self.dest == "wandb":
                if "entity" in str(err).lower() or "project" in str(err).lower():
                    print("\nW&B requires 'entity' and 'project' to be configured.")
                    print(
                        "Set via: -o export.wandb.entity=my-org -o export.wandb.project=my-proj"
                    )
                elif "not installed" in str(err).lower():
                    print("\nW&B package not installed.")
                    print("Install via: pip install nemo-evaluator-launcher[wandb]")
            elif self.dest == "gsheets":
                if "not installed" in str(err).lower():
                    print("\nGoogle Sheets package not installed.")
                    print("Install via: pip install nemo-evaluator-launcher[gsheets]")
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
                    path_hint = job_result.get("dest") or metadata.get("output_dir")
                    if self.dest == "local" and path_hint:
                        print(f"    Path: {path_hint}")
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

    def _validate_overrides(self, overrides: List[str], dest: str) -> None:
        """Validate override list for destination consistency.

        Raises:
            ValueError: If overrides specify wrong destination or have other issues.
        """
        if not overrides:
            return  # nothing to validate

        # Check each override for destination mismatch
        for override_str in overrides:
            if override_str.startswith(
                "export."
            ):  # check if override starts with export.
                # Extract destination from override path
                try:
                    key_part = override_str.split("=")[0]  # Get left side before =
                    parts = key_part.split(".")
                    if len(parts) >= 2:
                        override_dest = parts[1]
                        if override_dest != dest:
                            raise ValueError(
                                f"Override destination mismatch: override specifies 'export.{override_dest}' but --dest is '{dest}'. "
                                f"Either change --dest to '{override_dest}' or use 'export.{dest}' in overrides."
                            )
                except (IndexError, AttributeError):
                    # miconstructed override -> OmegaConf handles this
                    pass
