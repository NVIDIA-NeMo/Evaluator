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

    job_dirs: Optional[List[str]] = field(
        default=None,
        alias=["--job-dirs"],
        help="Directories used to search for job artifacts and logs. Allows to export results for jobs not stored in the "
        "local database (e.g. during auto-export on a remote machine or executed by a different user). "
        "If privided, it is used to search for sub-directories with results for specified invocation ID "
        "and the jobs information is inferred from the metadata stored in the job artifacts directories. "
        "Can be specified multiple times to search in multiple directories.",
    )

    def execute(self) -> None:
        """Execute export."""
        # Import heavy dependencies only when needed
        import os

        from omegaconf import OmegaConf

        from nemo_evaluator_launcher.api.functional import export_results

        # Validation: ensure IDs are provided
        if not self.invocation_ids:
            print("Error: No IDs provided. Specify one or more invocation or job IDs.")
            print(
                "Usage: nemo-evaluator-launcher export <id> [<id>...] --dest <destination>"
            )
            return

        # Load configuration from file if provided
        config: dict[str, Any] = {}
        if self.config:
            config = self._load_config_from_file(self.config)

        # CLI arguments override config file values
        # Always set copy_logs from CLI
        config["copy_logs"] = self.copy_logs

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

        # Add job_dirs if explicitly passed via CLI
        if self.job_dirs:
            config["job_dirs"] = self.job_dirs

        # Parse and validate overrides
        if self.override:
            # FIXME(martas): this looks over-complicated.
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

        # FIXME(martas): why is this needed? why we added it to the config?
        if "only_required" in config and self.only_required is True:
            config.pop("only_required", None)

        print(
            f"Exporting {len(self.invocation_ids)} {'invocations' if len(self.invocation_ids) > 1 else 'invocation'} to {self.dest}..."
        )

        result = export_results(self.invocation_ids, self.dest, config)

        # Success path
        print(f"Export completed: {result['metadata']}")
        if not result.get("success", False):
            print("Some jobs failed to export. See logs above for more details.")
            return

    def _load_config_from_file(self, config_path: str) -> dict[str, Any]:
        """Load export configuration from a file.

        Args:
            config_path: Path to the config file

        Returns:
            Dictionary containing the export configuration
        """
        import yaml

        # Load config file directly
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Config file {config_path} must contain a dictionary at the root level"
            )

        # If config has an 'export.<dest>' structure, extract the relevant section
        if "export" in config_dict and isinstance(config_dict["export"], dict):
            if self.dest in config_dict["export"]:
                # Use destination-specific config
                dest_config = config_dict["export"][self.dest] or {}
                # Also merge in any top-level export keys (as fallback)
                result = {
                    k: v
                    for k, v in config_dict.items()
                    if k != "export" and not k.startswith("_")
                }
                result.update(dest_config)
                return result
            else:
                # No destination-specific config, use top-level
                return {
                    k: v
                    for k, v in config_dict.items()
                    if k != "export" and not k.startswith("_")
                }
        else:
            # Flat config structure
            return config_dict

    def _validate_overrides(self, overrides: List[str], dest: str) -> None:
        """Validate override list for destination consistency.

        Raises:
            ValueError: If overrides specify wrong destination or have other issues.
        """
        # FIXME(martas): we should just let hydra handle this
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
