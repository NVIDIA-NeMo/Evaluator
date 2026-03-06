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

from simple_parsing import ArgumentParser, field

from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class Cmd:
    """Watch a checkpoint directory and trigger evaluations for new checkpoints."""

    config: str = field(
        alias=["--config"],
        metadata={"help": "Full path to eval config file (same as nel run)."},
    )
    watch_dir: str | None = field(
        default=None,
        alias=["--watch-dir"],
        metadata={"help": "Directory to watch for new checkpoint subdirectories."},
    )
    watch_config: str | None = field(
        default=None,
        alias=["--watch-config"],
        metadata={"help": "Path to YAML file defining multiple watch directories."},
    )
    interval: int = field(
        default=300,
        alias=["--interval"],
        metadata={"help": "Polling interval in seconds (default: 300)."},
    )
    ready_markers: str = field(
        default="metadata.json,config.yaml",
        alias=["--ready-markers"],
        metadata={
            "help": "Comma-separated list of marker files. A checkpoint is ready if ANY exist (default: metadata.json,config.yaml)."
        },
    )
    checkpoint_patterns: str = field(
        default="iter_*,step_*",
        alias=["--checkpoint-patterns"],
        metadata={
            "help": "Comma-separated glob patterns for checkpoint directory names (default: iter_*,step_*)."
        },
    )
    checkpoint_field: str = field(
        default="deployment.hf_model_handle",
        alias=["--checkpoint-field"],
        metadata={
            "help": "Dot-separated config field to override with checkpoint path (default: deployment.hf_model_handle)."
        },
    )
    order: str = field(
        default="newest",
        alias=["--order"],
        metadata={
            "help": "Processing order: 'newest' (highest step first) or 'oldest' (lowest step first). Default: newest."
        },
    )
    state_file: str | None = field(
        default=None,
        alias=["--state-file"],
        metadata={
            "help": "Path to state file for tracking submitted checkpoints. Auto-generated if not provided."
        },
    )
    dry_run: bool = field(
        default=False,
        alias=["-n", "--dry-run"],
        metadata={"help": "Show what would be submitted, do not actually submit."},
    )
    once: bool = field(
        default=False,
        alias=["--once"],
        metadata={"help": "Scan once and exit, do not loop."},
    )
    override: list[str] = field(
        default_factory=list,
        action="append",
        nargs="?",
        alias=["-o"],
        metadata={
            "help": "Hydra override in the form some.param.path=value (pass multiple -o for multiple overrides).",
        },
    )

    def execute(self) -> None:
        from dotenv import load_dotenv

        from nemo_evaluator_launcher.api.functional import RunConfig
        from nemo_evaluator_launcher.api.watch import (
            _load_watch_config,
            watch_checkpoints,
        )
        from nemo_evaluator_launcher.common.printing_utils import bold, cyan

        # Validate mutually exclusive args
        if not self.watch_dir and not self.watch_config:
            raise ValueError("Either --watch-dir or --watch-config is required.")
        if self.watch_dir and self.watch_config:
            raise ValueError("--watch-dir and --watch-config are mutually exclusive.")
        if self.order not in ("newest", "oldest"):
            raise ValueError(
                f"--order must be 'newest' or 'oldest', got '{self.order}'."
            )

        # Auto-load .env from cwd if present
        env_path = pathlib.Path.cwd() / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)

        # Load configuration via Hydra
        config = RunConfig.from_hydra(
            config=self.config,
            hydra_overrides=self.override,
        )

        # Parse comma-separated args
        ready_markers = [m.strip() for m in self.ready_markers.split(",") if m.strip()]
        checkpoint_patterns = [
            p.strip() for p in self.checkpoint_patterns.split(",") if p.strip()
        ]

        state_file_path = pathlib.Path(self.state_file) if self.state_file else None

        # Determine watch mode
        watch_dir = None
        watch_dirs = None
        if self.watch_config:
            watch_config_path = pathlib.Path(self.watch_config)
            if not watch_config_path.is_file():
                raise FileNotFoundError(
                    f"Watch config file does not exist: {self.watch_config}"
                )
            watch_dirs, global_field = _load_watch_config(watch_config_path)
            # Use global field from watch config if user didn't override on CLI
            if self.checkpoint_field == "deployment.hf_model_handle":
                checkpoint_field = global_field
            else:
                checkpoint_field = self.checkpoint_field
        else:
            watch_dir = pathlib.Path(self.watch_dir)
            if not watch_dir.is_dir():
                raise FileNotFoundError(
                    f"Watch directory does not exist: {self.watch_dir}"
                )
            checkpoint_field = self.checkpoint_field

        logger.info(
            "Starting watch mode",
            watch_dir=self.watch_dir or "(multi-dir)",
            interval=self.interval,
            ready_markers=self.ready_markers,
            checkpoint_patterns=self.checkpoint_patterns,
            checkpoint_field=checkpoint_field,
            order=self.order,
            dry_run=self.dry_run,
            once=self.once,
        )

        submissions = watch_checkpoints(
            config=config,
            watch_dir=watch_dir,
            watch_dirs=watch_dirs,
            interval=self.interval,
            ready_markers=ready_markers,
            checkpoint_patterns=checkpoint_patterns,
            checkpoint_field=checkpoint_field,
            order=self.order,
            dry_run=self.dry_run,
            once=self.once,
            state_file=state_file_path,
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
    parser.add_arguments(Cmd, dest="cmd")
    args = parser.parse_args()
    args.cmd.execute()
