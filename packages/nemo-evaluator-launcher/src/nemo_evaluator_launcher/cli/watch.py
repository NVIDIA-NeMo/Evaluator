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
"""CLI entrypoint for nel-watch — continuously evaluate new checkpoints."""

import pathlib
from dataclasses import dataclass
from typing import List

from simple_parsing import ArgumentParser, field

from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class WatchCmd:
    """Watch checkpoint directories and trigger evaluations for new checkpoints."""

    config: str = field(
        alias=["--config"],
        metadata={"help": "Path to the watch config YAML file."},
    )
    # TODO: add cli parameters: directories, output_dir, ready_markers, checkpoint_patterns
    interval: str | None = field(
        default=None,
        alias=["--interval"],
        metadata={
            "help": "Polling interval in seconds. Use 'null' to scan once and exit."
        },
    )
    order: str | None = field(
        default=None,
        alias=["--order"],
        metadata={
            "help": "Processing order: 'last' (highest step first) or 'first' (lowest step first)."
        },
    )
    resubmit_previous_sessions: bool = field(
        default=False,
        alias=["--resubmit-previous-sessions"],
        metadata={
            "help": "Resubmit checkpoints that were already submitted in previous sessions."
        },
    )
    dry_run: bool = field(
        default=False,
        alias=["-n", "--dry-run"],
        metadata={"help": "Show what would be submitted, do not actually submit."},
    )
    override: List[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-o", "--override"],
        metadata={
            "help": "Dot-path overrides for the watch config in the form key.subkey=value "
            "(pass multiple -o for multiple overrides). Overrides to evaluation_configs are not supported."
        },
    )

    def execute(self) -> None:
        from nemo_evaluator_launcher.common.printing_utils import bold, cyan
        from nemo_evaluator_launcher.watcher.configs import WatchConfig
        from nemo_evaluator_launcher.watcher.run import watch_and_evaluate

        config_path = pathlib.Path(self.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Watch config file does not exist: {self.config}")

        watch_config = WatchConfig.from_hydra(path=config_path, overrides=self.override)

        # Apply explicit CLI flag overrides (take precedence over -o overrides)
        if self.order is not None:
            if self.order not in ("last", "first"):
                raise ValueError(
                    f"--order must be 'last' or 'first', got '{self.order}'."
                )
            watch_config.monitoring_config.order = self.order

        if self.interval is not None:
            if self.interval == "null":
                watch_config.monitoring_config.interval = None
            else:
                try:
                    watch_config.monitoring_config.interval = int(self.interval)
                except ValueError:
                    raise ValueError(
                        f"--interval must be an integer, got '{self.interval}'."
                    )

        logger.info(
            "Starting watch mode",
            config=self.config,
            overrides=self.override,
            directories=watch_config.monitoring_config.directories,
            interval=watch_config.monitoring_config.interval,
            order=watch_config.monitoring_config.order,
            dry_run=self.dry_run,
        )

        submissions = watch_and_evaluate(
            watch_config=watch_config,
            resubmit_previous_sessions=self.resubmit_previous_sessions,
            dry_run=self.dry_run,
        )

        # Print summary
        print(bold(cyan("\n--- Watch Summary ---")))
        if not submissions:
            print("No checkpoints were submitted.")
        else:
            print(f"Submitted {len(submissions)} checkpoint evaluation(s):")
            for s in submissions:
                inv_str = s.invocation_id or "(dry-run)"
                print(f"  {s.checkpoint} -> {inv_str}")
        print(bold(cyan("--- End Summary ---")))


def main():
    parser = ArgumentParser()
    parser.add_arguments(WatchCmd, dest="cmd")
    args = parser.parse_args()
    args.cmd.execute()
