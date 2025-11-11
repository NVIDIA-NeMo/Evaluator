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
"""CLI command for listing task details."""

import json
from dataclasses import dataclass

import yaml
from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.task_ir import (
    TaskIntermediateRepresentation,
    _load_tasks_from_frameworks,
)


@dataclass
class Cmd:
    """List task command configuration."""

    task_identifier: str = field(
        default="",
        positional=True,
        help="Task identifier in format '[harness.]task_name'. If empty, shows all tasks.",
    )
    json: bool = field(
        default=False,
        action="store_true",
        help="Print output as JSON instead of formatted text",
    )
    frameworks_file: str = field(
        default="",
        help="Path to all_frameworks.yaml file (default: auto-detect)",
    )

    def execute(self) -> None:
        """Execute the ls task command."""
        import pathlib

        # Load frameworks file path
        frameworks_path = None
        if self.frameworks_file:
            frameworks_path = pathlib.Path(self.frameworks_file)
            if not frameworks_path.exists():
                logger.error("Frameworks file not found", path=str(frameworks_path))
                return

        # Load tasks
        try:
            tasks = _load_tasks_from_frameworks(frameworks_path)
        except Exception as e:
            print(f"Error loading tasks: {e}")
            import traceback

            traceback.print_exc()
            logger.error("Failed to load tasks", error=str(e), exc_info=True)
            return

        if not tasks:
            print("No tasks found.")
            if frameworks_path:
                print(f"  Frameworks file: {frameworks_path}")
            else:
                print(
                    "  Note: Make sure all_frameworks.yaml exists and contains valid framework definitions."
                )
            return

        # Parse task identifier
        harness_filter = None
        task_filter = None
        if self.task_identifier:
            if "." in self.task_identifier:
                parts = self.task_identifier.split(".", 1)
                harness_filter = parts[0]
                task_filter = parts[1]
            else:
                task_filter = self.task_identifier

        # Filter tasks
        filtered_tasks = []
        for task in tasks:
            if harness_filter and task.harness.lower() != harness_filter.lower():
                continue
            if task_filter and task.name.lower() != task_filter.lower():
                continue
            filtered_tasks.append(task)

        if not filtered_tasks:
            print(f"No tasks found matching: {self.task_identifier}")
            if self.task_identifier:
                # Show available tasks for debugging
                print("\nAvailable tasks (showing first 10):")
                for i, task in enumerate(tasks[:10]):
                    print(f"  - {task.harness}.{task.name}")
                if len(tasks) > 10:
                    print(f"  ... and {len(tasks) - 10} more")
            return

        # Display tasks
        if self.json:
            self._print_json(filtered_tasks)
        else:
            self._print_formatted(filtered_tasks)

    def _print_json(self, tasks: list[TaskIntermediateRepresentation]) -> None:
        """Print tasks as JSON."""
        tasks_dict = [task.to_dict() for task in tasks]
        print(json.dumps({"tasks": tasks_dict}, indent=2))

    def _print_formatted(self, tasks: list[TaskIntermediateRepresentation]) -> None:
        """Print tasks in formatted text."""
        for i, task in enumerate(tasks):
            if i > 0:
                print()  # Spacing between tasks
                print("=" * 80)

            print(f"Task: {task.name}")
            if task.description:
                print(f"Description: {task.description}")
            print(f"Harness: {task.harness}")
            print(f"Container: {task.container}")
            if task.container_digest:
                print(f"Container Digest: {task.container_digest}")

            # Print defaults as YAML
            if task.defaults:
                print("\nDefaults:")
                defaults_yaml = yaml.dump(
                    task.defaults, default_flow_style=False, sort_keys=False
                )
                # Indent defaults
                for line in defaults_yaml.split("\n"):
                    if line.strip():
                        print(f"  {line}")
                    else:
                        print()

            print("-" * 80)

        print(f"\nTotal: {len(tasks)} task{'s' if len(tasks) != 1 else ''}")
