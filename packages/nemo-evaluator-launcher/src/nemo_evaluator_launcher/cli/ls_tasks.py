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
from collections import defaultdict
from dataclasses import dataclass

from simple_parsing import field

from nemo_evaluator_launcher.common.printing_utils import (
    bold,
    cyan,
    grey,
    magenta,
)


@dataclass
class Cmd:
    """List command configuration."""

    json: bool = field(
        default=False,
        action="store_true",
        help="Print output as JSON instead of table format",
    )
    from_container: str = field(
        default="",
        help="Load tasks from container image (e.g., nvcr.io/nvidia/eval-factory/simple-evals:25.10). "
        "If provided, extracts framework.yml from container and lists tasks on-the-fly instead of using mapping.toml",
    )

    def execute(self) -> None:
        # Import heavy dependencies only when needed
        import json

        if self.from_container:
            # Load tasks from container
            from nemo_evaluator_launcher.common.container_metadata import (
                load_tasks_from_container,
            )

            try:
                tasks = load_tasks_from_container(self.from_container)
            except ValueError as e:
                from nemo_evaluator_launcher.common.logging_utils import logger

                logger.error(
                    "Failed to load tasks from container",
                    container=self.from_container,
                    error=str(e),
                )
                return
            except Exception as e:
                from nemo_evaluator_launcher.common.logging_utils import logger

                logger.error(
                    "Failed to load tasks from container",
                    container=self.from_container,
                    error=str(e),
                    exc_info=True,
                )
                return

            if not tasks:
                from nemo_evaluator_launcher.common.logging_utils import logger

                logger.error(
                    "No tasks found in container",
                    container=self.from_container,
                )
                return

            # Convert TaskIntermediateRepresentation to format expected by get_tasks_list()
            # Build data structure matching get_tasks_list() output format
            data = []
            for task in tasks:
                # Extract endpoint types from defaults
                endpoint_types = (
                    task.defaults.get("target", {})
                    .get("api_endpoint", {})
                    .get("type", "chat")
                )
                if isinstance(endpoint_types, str):
                    endpoint_types = [endpoint_types]

                task_type = task.defaults.get("config", {}).get("type", "")

                data.append(
                    [
                        task.name,  # task
                        ",".join(endpoint_types)
                        if isinstance(endpoint_types, list)
                        else endpoint_types,  # endpoint_type
                        task.harness,  # harness
                        task.container,  # container
                        task.description,  # description
                        task_type,  # type
                    ]
                )
        else:
            # Default behavior: load from mapping.toml via get_tasks_list()
            from nemo_evaluator_launcher.api.functional import get_tasks_list

            # TODO(dfridman): modify `get_tasks_list` to return a list of dicts in the first place
            data = get_tasks_list()

        headers = [
            "task",
            "endpoint_type",
            "harness",
            "container",
            "description",
            "type",
        ]
        supported_benchmarks = []
        for task_data in data:
            assert len(task_data) == len(headers)
            supported_benchmarks.append(dict(zip(headers, task_data)))

        if self.json:
            print(json.dumps({"tasks": supported_benchmarks}, indent=2))
        else:
            self._print_table(supported_benchmarks)

    def _print_table(self, tasks: list[dict]) -> None:
        """Print tasks grouped by harness and container in table format with colorized output."""
        if not tasks:
            print("No tasks found.")
            return

        # Group tasks by harness and container
        grouped = defaultdict(lambda: defaultdict(list))
        for task in tasks:
            harness = task["harness"]
            container = task["container"]
            grouped[harness][container].append(task)

        # Print grouped tables
        for i, (harness, containers) in enumerate(grouped.items()):
            if i > 0:
                print()  # Extra spacing between harnesses

            for j, (container, container_tasks) in enumerate(containers.items()):
                if j > 0:
                    print()  # Spacing between containers

                # Prepare task table first to get column widths
                task_header = "task"
                rows = []
                for task in container_tasks:
                    task_name = task["task"]
                    endpoint_type = task["endpoint_type"]
                    task_type = task.get("type", "")
                    description = task.get("description", "")
                    # Format: task_name (endpoint_type, task_type) - first 30 chars...
                    description_preview = description[:30] if description else ""
                    if len(description) > 30:
                        description_preview += "..."

                    # Build the display name
                    type_part = f"{endpoint_type}"
                    if task_type:
                        type_part += f", {task_type}"
                    display_name = f"{task_name} ({type_part})"
                    if description_preview:
                        display_name = f"{display_name} - {description_preview}"
                    rows.append(display_name)

                # Sort tasks alphabetically for better readability
                rows.sort()

                # Calculate column width
                max_task_width = (
                    max(len(task_header), max(len(str(row)) for row in rows)) + 2
                )

                # Calculate required width for header content
                harness_line = f"harness: {harness}"
                container_line = f"container: {container}"
                header_content_width = (
                    max(len(harness_line), len(container_line)) + 4
                )  # +4 for "| " and " |"

                # Use the larger of the two widths
                table_width = max(max_task_width, header_content_width)

                # Limit separator width to prevent overflow on small terminals
                # Use terminal width if available, otherwise cap at 120 characters
                import shutil

                try:
                    terminal_width = shutil.get_terminal_size().columns
                    separator_width = min(
                        table_width, terminal_width - 2
                    )  # -2 for safety margin
                except Exception:
                    # Fallback if terminal size can't be determined
                    separator_width = min(table_width, 120)

                # Print combined header with harness and container info - colorized
                # Keys: magenta, Values: cyan (matching logging utils)
                print(bold("=" * separator_width))
                print(f"{magenta('harness:')} {cyan(str(harness))}")
                print(f"{magenta('container:')} {cyan(str(container))}")

                # Print task table header separator
                print(" " * table_width)
                print(bold(f"{task_header:<{table_width}}"))
                print(bold("-" * separator_width))

                # Print task rows - use grey for task descriptions
                for row in rows:
                    print(f"{grey(str(row)):<{table_width}}")

                print(bold("-" * separator_width))
                # Show task count - grey for count text
                task_count = len(rows)
                task_word = "task" if task_count == 1 else "tasks"
                print(f"  {grey(f'{task_count} {task_word} available')}")
                print(bold("=" * separator_width))

                print()
