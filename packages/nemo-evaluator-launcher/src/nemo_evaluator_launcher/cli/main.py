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
"""Main CLI module using simple-parsing with subcommands."""

import os

from simple_parsing import ArgumentParser

import nemo_evaluator_launcher.cli.export as export
import nemo_evaluator_launcher.cli.info as info
import nemo_evaluator_launcher.cli.kill as kill
import nemo_evaluator_launcher.cli.ls_runs as ls_runs
import nemo_evaluator_launcher.cli.ls_tasks as ls_tasks
import nemo_evaluator_launcher.cli.run as run
import nemo_evaluator_launcher.cli.status as status
import nemo_evaluator_launcher.cli.version as version
from nemo_evaluator_launcher.common.logging_utils import logger

VERSION_HELP = "Show version information"


def is_verbose_enabled(args) -> bool:
    """Check if verbose flag is enabled in any subcommand."""
    # Check global verbose flag
    if hasattr(args, "verbose") and args.verbose:
        return True

    # Check subcommand verbose flags
    subcommands = [
        "run",
        "status",
        "info",
        "kill",
        "tasks_alias",
        "tasks",
        "runs",
        "export",
    ]
    for subcmd in subcommands:
        if hasattr(args, subcmd) and hasattr(getattr(args, subcmd), "verbose"):
            if getattr(getattr(args, subcmd), "verbose"):
                return True

    return False


def create_parser() -> ArgumentParser:
    """Create and configure the CLI argument parser with subcommands."""
    parser = ArgumentParser()

    # Add --version flag at the top level
    parser.add_argument("--version", action="store_true", help=VERSION_HELP)

    # Add --verbose/-v flag for debug logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets LOG_LEVEL=DEBUG)",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Version subcommand
    version_parser = subparsers.add_parser(
        "version",
        help=VERSION_HELP,
        description=VERSION_HELP,
    )
    version_parser.add_arguments(version.Cmd, dest="version")

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run evaluation", description="Run evaluation"
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets LOG_LEVEL=DEBUG)",
    )
    run_parser.add_arguments(run.Cmd, dest="run")

    # Status subcommand
    status_parser = subparsers.add_parser(
        "status", help="Check job status", description="Check job status"
    )
    status_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets LOG_LEVEL=DEBUG)",
    )
    status_parser.add_arguments(status.Cmd, dest="status")

    # Kill subcommand
    kill_parser = subparsers.add_parser(
        "kill",
        help="Kill a job or invocation",
        description="Kill a job (e.g., aefc4819.0) or entire invocation (e.g., aefc4819) by its ID",
    )
    kill_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets LOG_LEVEL=DEBUG)",
    )
    kill_parser.add_arguments(kill.Cmd, dest="kill")

    # Ls subcommand (with nested subcommands)
    ls_parser = subparsers.add_parser(
        "ls", help="List resources", description="List tasks or runs"
    )
    ls_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets LOG_LEVEL=DEBUG)",
    )
    # Add arguments from `ls tasks` so that they work with `ls` as default alias
    ls_parser.add_arguments(ls_tasks.Cmd, dest="tasks_alias")

    ls_sub = ls_parser.add_subparsers(dest="ls_command", required=False)

    # ls tasks (default)
    ls_tasks_parser = ls_sub.add_parser(
        "tasks", help="List available tasks", description="List available tasks"
    )
    ls_tasks_parser.add_arguments(ls_tasks.Cmd, dest="tasks")

    # ls runs (invocations summary)
    ls_runs_parser = ls_sub.add_parser(
        "runs",
        help="List invocations (runs)",
        description="Show a concise table of invocations from the exec DB",
    )
    ls_runs_parser.add_arguments(ls_runs.Cmd, dest="runs")

    # Export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export evaluation results",
        description="Export evaluation results takes a List of invocation ids and a list of destinations(local, gitlab, wandb)",
    )
    export_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (sets LOG_LEVEL=DEBUG)",
    )
    export_parser.add_arguments(export.ExportCmd, dest="export")

    # Info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Display evaluation job information",
        description="Info functionalities for nemo-evaluator-launcher",
    )
    info_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    info_parser.add_arguments(info.InfoCmd, dest="info")

    return parser


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --verbose flag
    if is_verbose_enabled(args):
        os.environ["LOG_LEVEL"] = "DEBUG"

    # Handle --version flag
    if hasattr(args, "version") and args.version:
        version_cmd = version.Cmd()
        version_cmd.execute()
        return

    # Handle case where no command is provided but --version wasn't used
    if not hasattr(args, "command") or args.command is None:
        parser.print_help()
        return

    logger.debug("Parsed arguments", args=args)
    if args.command == "version":
        args.version.execute()
    elif args.command == "run":
        args.run.execute()
    elif args.command == "status":
        args.status.execute()
    elif args.command == "kill":
        args.kill.execute()
    elif args.command == "ls":
        # Dispatch nested ls subcommands
        if args.ls_command is None or args.ls_command == "tasks":
            # Default to tasks when no subcommand specified
            if hasattr(args, "tasks_alias"):
                args.tasks_alias.execute()
            else:
                args.tasks.execute()
        elif args.ls_command == "runs":
            args.runs.execute()
    elif args.command == "export":
        args.export.execute()
    elif args.command == "info":
        args.info.execute()


if __name__ == "__main__":
    main()
