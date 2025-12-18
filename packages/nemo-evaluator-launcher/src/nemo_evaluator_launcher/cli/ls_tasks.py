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

                data.append(
                    [
                        task.name,  # task
                        ",".join(endpoint_types)
                        if isinstance(endpoint_types, list)
                        else endpoint_types,  # endpoint_type
                        task.harness,  # harness
                        task.container,  # container
                        getattr(task, "container_arch", "") or "",  # arch
                        task.description,  # description
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
            "arch",
            "description",
        ]
        supported_benchmarks = []
        for task_data in data:
            if len(task_data) < len(headers):
                raise ValueError(
                    f"Invalid task row shape: expected at least {len(headers)} columns, got {len(task_data)}"
                )
            # Backwards/forwards compat: allow extra columns and ignore them.
            supported_benchmarks.append(dict(zip(headers, task_data[: len(headers)])))

        if self.json:
            print(json.dumps({"tasks": supported_benchmarks}, indent=2))
        else:
            self._print_table(supported_benchmarks)

    def _print_table(self, tasks: list[dict]) -> None:
        """Print tasks grouped by harness and container in table format with colorized output."""
        if not tasks:
            print("No tasks found.")
            return

        def _truncate(s: str, max_len: int) -> str:
            s = s or ""
            if max_len <= 0:
                return ""
            if len(s) <= max_len:
                return s
            if max_len <= 3:
                return s[:max_len]
            return s[: max_len - 3] + "..."

        def _infer_arch(container: str, container_tasks: list[dict]) -> str:
            # Prefer explicit arch from task IRs.
            for t in container_tasks:
                a = (t.get("arch") or "").strip()
                if a:
                    return a

            # Heuristic fallback: look for common suffixes in tag.
            c = (container or "").lower()
            if "arm64" in c or "aarch64" in c:
                return "arm"
            if "amd64" in c or "x86_64" in c:
                return "amd"
            return "unknown"

        def _infer_registry(container: str) -> str:
            try:
                from nemo_evaluator_launcher.common.container_metadata.utils import (
                    parse_container_image,
                )

                registry_type, _registry_url, _repo, _ref = parse_container_image(
                    container
                )
                return str(registry_type)
            except Exception:
                # Best-effort fallback for unknown formats.
                c = (container or "").lower()
                if "nvcr.io/" in c or c.startswith("nvcr.io"):
                    return "nvcr"
                if "gitlab" in c:
                    return "gitlab"
                return ""

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

                rows = []
                for task in container_tasks:
                    rows.append(
                        {
                            "task": str(task.get("task", "")),
                            "endpoint": str(task.get("endpoint_type", "")),
                            "description": str(task.get("description", "")),
                        }
                    )
                rows.sort(key=lambda r: r["task"].lower())

                # Calculate required width for header content
                harness_line = f"harness: {harness}"
                container_line = f"container: {container}"
                arch_line = f"arch: {_infer_arch(container, container_tasks)}"
                registry_line = f"registry: {_infer_registry(container)}"
                header_content_width = (
                    max(
                        len(harness_line),
                        len(container_line),
                        len(arch_line),
                        len(registry_line),
                    )
                    + 4
                )  # +4 for "| " and " |"

                # Limit separator width to prevent overflow on small terminals
                # Use terminal width if available, otherwise cap at 120 characters
                import shutil

                try:
                    terminal_width = shutil.get_terminal_size().columns
                    separator_width = min(terminal_width - 2, 160)  # -2 safety margin
                except Exception:
                    # Fallback if terminal size can't be determined
                    separator_width = 120

                separator_width = max(separator_width, min(header_content_width, 160))

                # Table columns (keep compact and stable).
                col_task = 36
                col_endpoint = 14
                sep = "  "
                fixed = col_task + col_endpoint + len(sep) * 2
                col_desc = max(20, separator_width - fixed)

                # Print combined header with harness and container info - colorized
                # Keys: magenta, Values: cyan (matching logging utils)
                print(bold("=" * separator_width))
                print(f"{magenta('harness:')} {cyan(str(harness))}")
                print(f"{magenta('container:')} {cyan(str(container))}")
                arch = _infer_arch(container, container_tasks)
                registry = _infer_registry(container)
                print(f"{magenta('arch:')} {cyan(str(arch))}")
                if registry:
                    print(f"{magenta('registry:')} {cyan(str(registry))}")

                # Print task table header separator
                print()
                print(
                    bold(
                        f"{'task':<{col_task}}{sep}"
                        f"{'endpoint':<{col_endpoint}}{sep}"
                        f"{'description':<{col_desc}}"
                    )
                )
                print(bold("-" * separator_width))

                # Print task rows - use grey for task descriptions
                for r in rows:
                    line = (
                        f"{_truncate(r['task'], col_task):<{col_task}}{sep}"
                        f"{_truncate(r['endpoint'], col_endpoint):<{col_endpoint}}{sep}"
                        f"{_truncate(r['description'], col_desc):<{col_desc}}"
                    )
                    print(grey(line))

                print(bold("-" * separator_width))
                # Show task count - grey for count text
                task_count = len(rows)
                task_word = "task" if task_count == 1 else "tasks"
                print(f"  {grey(f'{task_count} {task_word} available')}")
                print(bold("=" * separator_width))

                print()
