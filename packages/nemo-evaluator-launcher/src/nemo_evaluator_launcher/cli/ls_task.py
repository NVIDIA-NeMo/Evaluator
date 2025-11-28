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
from nemo_evaluator_launcher.common.mapping import load_tasks_mapping
from nemo_evaluator_launcher.common.task_ir import (
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
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

        # If --from is provided, load tasks from container
        if self.from_container:
            logger.debug(
                "Loading tasks from container",
                container=self.from_container,
            )
            tasks = self._load_tasks_from_container(self.from_container)
            if not tasks:
                logger.error(
                    "Failed to load tasks from container",
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
            # Don't override containers when loading from --from - the container IS the source
            # Verify all tasks are from the specified container (safeguard)
            expected_container = self.from_container
            original_count = len(tasks)
            tasks = [
                task
                for task in tasks
                if task.container == expected_container
            ]
            if len(tasks) != original_count:
                logger.warning(
                    "Filtered out tasks from different containers",
                    expected_container=expected_container,
                    original_count=original_count,
                    filtered_count=len(tasks),
                )
            if not tasks:
                logger.warning(
                    "No tasks found from specified container after filtering",
                    container=expected_container,
                )
                return
        else:
            # Default behavior: load from all_tasks_irs.yaml
            tasks_path = None
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
                from nemo_evaluator_launcher.common.printing_utils import yellow

                print(yellow("⚠ Warning: Tasks are from unverified mapping (mapping.toml checksum mismatch)"))
                print(yellow("  Consider regenerating all_tasks_irs.yaml if mapping.toml has changed"))
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
                        container_lookup[(normalized_harness, normalized_task)] = container

                # Update task containers from mapping.toml
                for task in tasks:
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

    def _load_tasks_from_container(
        self, container: str
    ) -> list[TaskIntermediateRepresentation]:
        """Load tasks from container by extracting and parsing framework.yml.

        Args:
            container: Container image identifier (e.g., nvcr.io/nvidia/eval-factory/simple-evals:25.10)

        Returns:
            List of TaskIntermediateRepresentation objects
        """
        # Import framework extraction functions from load_framework_definitions.py script
        # Note: These functions are in a script, so we need to import them directly
        import sys
        import pathlib

        # Add scripts directory to path temporarily
        # Path: cli -> nemo_evaluator_launcher -> src -> nemo-evaluator-launcher -> packages
        # Then go to nemo-evaluator-launcher/scripts
        scripts_dir = (
            pathlib.Path(__file__).parent.parent.parent.parent
            / "scripts"
        )
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from load_framework_definitions import (
                extract_framework_yml,
                parse_framework_to_irs,
            )
            from nemo_evaluator_launcher.common.task_ir import (
                _extract_harness_from_container,
            )

            # Extract harness name from container
            harness_name = _extract_harness_from_container(container)

            # Extract framework.yml from container (uses existing cache)
            framework_content, container_digest = extract_framework_yml(
                container=container,
                harness_name=harness_name,
                max_layer_size=100 * 1024,  # 100KB default
                use_cache=True,  # Use existing cache from ~/.nemo-evaluator/docker-meta/
            )

            if not framework_content:
                logger.error(
                    "Could not extract framework.yml from container",
                    container=container,
                )
                return []

            # Parse framework.yml to IRs
            try:
                harness_ir, task_irs = parse_framework_to_irs(
                    framework_content=framework_content,
                    container_id=container,
                    container_digest=container_digest,
                    harness_name=harness_name,
                )
                logger.info(
                    "Loaded tasks from container",
                    container=container,
                    num_tasks=len(task_irs),
                )
                return task_irs
            except Exception as e:
                logger.error(
                    "Failed to parse framework.yml from container",
                    container=container,
                    error=str(e),
                    exc_info=True,
                )
                return []
        finally:
            # Remove scripts directory from path
            if str(scripts_dir) in sys.path:
                sys.path.remove(str(scripts_dir))
