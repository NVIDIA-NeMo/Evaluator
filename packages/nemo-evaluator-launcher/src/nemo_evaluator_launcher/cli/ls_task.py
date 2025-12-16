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

from nemo_evaluator_launcher.common.container_metadata import (
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.mapping import load_tasks_mapping
from nemo_evaluator_launcher.common.printing_utils import (
    bold,
    cyan,
    magenta,
    yellow,
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
    tasks_file: str = field(
        default="",
        help="Path to all_tasks_irs.yaml file (default: auto-detect)",
    )
    from_container: str = field(
        default="",
        help="Load tasks from container image (e.g., nvcr.io/nvidia/eval-factory/simple-evals:25.10). "
        "If provided, extracts framework.yml from container and loads tasks on-the-fly instead of using all_tasks_irs.yaml",
    )

    def execute(self) -> None:
        """Execute the ls task command."""
        import pathlib

        # Initialize tasks_path to None - it will be set when loading from file
        tasks_path = None

        # If --from is provided, load tasks from container
        if self.from_container:
            from nemo_evaluator_launcher.common.container_metadata import (
                load_tasks_from_container,
            )

            try:
                tasks = load_tasks_from_container(self.from_container)
            except ValueError as e:
                print(f"Error: {e}")
                return
            except Exception as e:
                logger.error(
                    "Failed to load tasks from container",
                    container=self.from_container,
                    error=str(e),
                    exc_info=True,
                )
                return

            if not tasks:
                logger.error(
                    "No tasks found in container",
                    container=self.from_container,
                )
                return

            logger.debug(
                "Loaded tasks from container",
                container=self.from_container,
                num_tasks=len(tasks),
                containers=set(task.container for task in tasks),
            )
            mapping_verified = True  # Tasks from container are always verified
        else:
            # Default behavior: load from all_tasks_irs.yaml
            if self.tasks_file:
                tasks_path = pathlib.Path(self.tasks_file)
                if not tasks_path.exists():
                    logger.error("Tasks file not found", path=str(tasks_path))
                    return

            # Load tasks
            try:
                tasks, mapping_verified = load_tasks_from_tasks_file(tasks_path)
            except Exception as e:
                print(f"Error loading tasks: {e}")
                import traceback

                traceback.print_exc()
                logger.error("Failed to load tasks", error=str(e), exc_info=True)
                return

            # Display warning if mapping is not verified
            if not mapping_verified:
                print(
                    yellow(
                        "⚠ Warning: Tasks are from unverified mapping (mapping.toml checksum mismatch)"
                    )
                )
                print(
                    yellow(
                        "  Consider regenerating all_tasks_irs.yaml if mapping.toml has changed"
                    )
                )
                print()

            # Override containers from mapping.toml (which has the latest containers)
            # This ensures ls task shows the same containers as ls tasks
            # Only do this when NOT using --from (when loading from all_tasks_irs.yaml)
            try:
                mapping = load_tasks_mapping()
                # Create a lookup: (normalized_harness, normalized_task_name) -> container
                # Use case-insensitive keys for matching
                container_lookup = {}
                for (harness, task_name), task_data in mapping.items():
                    container = task_data.get("container")
                    if container:
                        # Normalize harness name for lookup (frameworks.yaml uses hyphens)
                        normalized_harness = harness.replace("_", "-").lower()
                        normalized_task = task_name.lower()
                        container_lookup[(normalized_harness, normalized_task)] = (
                            container
                        )

                # Update task containers from mapping.toml
                for task in tasks:
                    # Defensive checks: ensure task has required attributes
                    if not hasattr(task, "harness") or not task.harness:
                        logger.warning(
                            "Task missing harness attribute, skipping container override",
                            task_name=getattr(task, "name", "unknown"),
                        )
                        continue
                    if not hasattr(task, "name") or not task.name:
                        logger.warning(
                            "Task missing name attribute, skipping container override",
                            harness=getattr(task, "harness", "unknown"),
                        )
                        continue

                    # Normalize both harness and task name for case-insensitive lookup
                    normalized_harness = task.harness.lower()
                    normalized_task = task.name.lower()
                    lookup_key = (normalized_harness, normalized_task)
                    if lookup_key in container_lookup:
                        task.container = container_lookup[lookup_key]
            except Exception as e:
                logger.debug(
                    "Failed to override containers from mapping.toml",
                    error=str(e),
                )
                # Continue with containers from all_tasks_irs.yaml if mapping load fails

        if not tasks:
            print("No tasks found.")
            if tasks_path:
                print(f"  Tasks file: {tasks_path}")
            else:
                print(
                    "  Note: Make sure all_tasks_irs.yaml exists and contains valid task definitions."
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
            self._print_formatted(filtered_tasks, mapping_verified)

    def _print_json(self, tasks: list[TaskIntermediateRepresentation]) -> None:
        """Print tasks as JSON."""
        tasks_dict = [task.to_dict() for task in tasks]
        print(json.dumps({"tasks": tasks_dict}, indent=2))

    def _print_formatted(
        self, tasks: list[TaskIntermediateRepresentation], mapping_verified: bool = True
    ) -> None:
        """Print tasks in formatted text with colorized output."""
        for i, task in enumerate(tasks):
            if i > 0:
                print()  # Spacing between tasks
                print(bold("=" * 80))

            # Task name - bold and magenta key, cyan value (matching logging utils)
            print(f"{bold(magenta('Task:'))} {bold(cyan(str(task.name)))}")

            # Description - magenta key, cyan value
            if task.description:
                print(f"{magenta('Description:')} {cyan(str(task.description))}")

            # Harness - magenta key, cyan value
            print(f"{magenta('Harness:')} {cyan(str(task.harness))}")

            # Container - magenta key, cyan value
            print(f"{magenta('Container:')} {cyan(str(task.container))}")

            # Container Digest - magenta key, cyan value
            if task.container_digest:
                print(
                    f"{magenta('Container Digest:')} {cyan(str(task.container_digest))}"
                )

            # Print defaults as YAML
            if task.defaults:
                print(f"\n{bold(magenta('Defaults:'))}")
                defaults_yaml = yaml.dump(
                    task.defaults, default_flow_style=False, sort_keys=False
                )
                # Indent defaults - use cyan for YAML content (FDF values)
                for line in defaults_yaml.split("\n"):
                    if line.strip():
                        print(f"  {cyan(line)}")
                    else:
                        print()

            print(bold("-" * 80))

        # Total count - bold
        task_word = "task" if len(tasks) == 1 else "tasks"
        print(f"\n{bold(f'Total: {len(tasks)} {task_word}')}")
