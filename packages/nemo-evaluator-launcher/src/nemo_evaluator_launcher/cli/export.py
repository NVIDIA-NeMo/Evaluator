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

from nemo_evaluator_launcher.common.logging_utils import logger


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
    copy_logs: bool = field(
        default=False,
        alias=["--copy-logs"],
        help="Export log files (if exporter allows it) (default: False).",
    )
    copy_artifacts: bool = field(
        default=True,
        alias=["--copy-artifacts"],
        help="Export artifact files (if exporter allows it) (default: True).",
    )
    log_metrics: List[str] = field(
        default_factory=list,
        alias=["--log-metrics"],
        help="Filter metrics by name (repeatable). Examples: score, f1, mmlu_score_micro.",
    )
    only_required: bool = field(
        default=True,
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
            config = self._load_config_from_file(self.config)
        else:
            config = OmegaConf.create({})

        # CLI arguments override config file values
        # Always set copy_logs from CLI
        config["copy_logs"] = self.copy_logs
        config["copy_artifacts"] = self.copy_artifacts
        config["only_required"] = self.only_required

        # Output handling
        if self.output_dir:
            config["output_dir"] = self.output_dir
        if self.output_filename:
            config["output_filename"] = self.output_filename

        # Format and filters
        if self.format and self.dest == "local":
            config["format"] = self.format
        if self.log_metrics:
            config["log_metrics"] = self.log_metrics

        # Add job_dirs if explicitly passed via CLI
        if self.job_dirs:
            config["job_dirs"] = self.job_dirs

        # Parse and validate overrides
        self._apply_overrides(config)

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

    def _load_config_from_file(self, config_path: str) -> OmegaConf:
        """Load export configuration from a file.

        Args:
            config_path: Path to the config file

        Returns:
            OmegaConf object containing the export configuration
        """
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ValueError(
                f"Config file {config_path} must contain a dictionary at the root level"
            )

        # If config has an 'export.<dest>' structure, extract the relevant section
        export_config = config_dict.get("export", {})
        if export_config:
            if self.dest in export_config:
                # Use destination-specific config
                return OmegaConf.create(export_config[self.dest])

            else:
                raise ValueError(
                    f"Export destination {self.dest} not found in config file {config_path}"
                )
        else:
            # Flat config structure
            logger.warning(
                f"Config file {config_path} does not have an 'export' section. Using top-level config."
            )
            return OmegaConf.create(config_dict)

    def _apply_overrides(self, config: OmegaConf) -> None:
        import re

        # matches ~export.dest.key and +export.dest.key and ++export.dest.key
        hydra_pattern = re.compile(rf"^(?:~|\+\+|\+)?export\.{self.dest}\.(.+)$")
        if not self.override:
            return

        to_del = []
        fileterd_overrides = []
        for override in self.override:
            match = hydra_pattern.match(override)
            if not match:
                logger.debug("Ignoring non-applicable override", override=override)
                continue
            rest = match.group(1)
            if override.startswith("~"):
                to_del.append(rest)
                logger.debug(
                    "Adding to delete list", config_key=rest, override=override
                )
                continue
            logger.debug("Adding to update list", config_key=rest, override=override)
            fileterd_overrides.append(rest)

        config_updates = OmegaConf.from_dotlist(fileterd_overrides)

        if config_updates:
            config.update(config_updates)

        for key_to_remove in to_del:
            *parent_path, leaf_key = key_to_remove.rsplit(".", 1)

            if len(parent_path) == 0:
                # top-level key
                config.pop(leaf_key, None)
            elif len(parent_path) == 1:
                # key deeper in the config
                parent = OmegaConf.select(config, parent_path[0])
                if parent is not None:
                    del parent[leaf_key]
                else:
                    logger.warning(
                        f"Could not remove {key_to_remove} from the config (key not found)"
                    )
            else:
                # impossible unless someone changes the line where we split the key
                raise RuntimeError(f"Invalid key to remove: {key_to_remove}")

        logger.debug("Final config after applying overrides", config=config)
