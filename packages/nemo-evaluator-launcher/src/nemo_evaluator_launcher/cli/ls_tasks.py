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


@dataclass
class Cmd:
    """List command configuration."""

    json: bool = field(
        default=False,
        action="store_true",
        help="Print output as JSON instead of table format",
    )

    def execute(self) -> None:
        # Import heavy dependencies only when needed
        import json

        from nemo_evaluator_launcher.api.functional import get_tasks_list

        # TODO(dfridman): modify `get_tasks_list` to return a list of dicts in the first place
        data = get_tasks_list()
        headers = ["task", "endpoint_type", "harness", "container"]
        supported_benchmarks = []
        for task_data in data:
            assert len(task_data) == len(headers)
            supported_benchmarks.append(dict(zip(headers, task_data)))

        if self.json:
            print(json.dumps({"tasks": supported_benchmarks}, indent=2))
        else:
            self._print_table(supported_benchmarks)

    def _print_table(self, tasks: list[dict]) -> None:
        """Print tasks grouped by harness and container in table format."""
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
                task_headers = ["task", "endpoint_type"]
                rows = []
                for task in container_tasks:
                    rows.append([task["task"], task["endpoint_type"]])

                # Sort tasks alphabetically for better readability
                rows.sort(key=lambda x: x[0])

                # Calculate column widths with some padding
                widths = [
                    max(len(task_headers[i]), max(len(str(row[i])) for row in rows)) + 2
                    for i in range(len(task_headers))
                ]

                # Calculate minimum table width based on task columns
                min_table_width = sum(widths) + len(widths) + 1

                # Calculate required width for header content
                harness_line = f"harness: {harness}"
                container_line = f"container: {container}"
                header_content_width = (
                    max(len(harness_line), len(container_line)) + 4
                )  # +4 for "| " and " |"

                # Use the larger of the two widths
                table_width = max(min_table_width, header_content_width)

                # Print combined header with harness and container info
                print("=" * table_width)
                print(f"{harness_line}")
                print(f"{container_line}")

                # Adjust column widths to fill the full table width
                available_width = table_width
                # Give more space to the first column (task names can be long)
                adjusted_widths = [
                    max(
                        widths[0], available_width * 2 // 3
                    ),  # 2/3 of available width for task
                    0,  # Will be calculated as remainder
                ]
                adjusted_widths[1] = (
                    available_width - adjusted_widths[0]
                )  # Remainder for endpoint_type

                # Print task table header separator
                print(" " * table_width)
                header_row = f"{task_headers[0]:<{adjusted_widths[0]}}{task_headers[1]:<{adjusted_widths[1]}}"
                print(header_row)
                print("-" * table_width)

                # Print task rows
                for row in rows:
                    data_row = f"{str(row[0]):<{adjusted_widths[0]}}{str(row[1]):<{adjusted_widths[1]}}"
                    print(data_row)

                print("-" * table_width)
                # Show task count
                task_count = len(rows)
                print(f"  {task_count} task{'s' if task_count != 1 else ''} available")
                print("=" * table_width)

                print()
