#!/usr/bin/env python3
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
"""Script to autogenerate task YAML files from intermediate representations."""

import argparse
import copy
import pathlib
import re
import shutil
import sys
from collections import defaultdict

import yaml

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.task_ir import (
    _load_harnesses_from_frameworks,
    _load_tasks_from_frameworks,
)


def find_repo_root(start_path: pathlib.Path) -> pathlib.Path:
    """Find the repository root by looking for .git directory.

    Args:
        start_path: Starting path to search from

    Returns:
        Path to repository root
    """
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # Fallback: return the starting path's parent if .git not found
    return start_path.parent.parent


def normalize_filename(name: str) -> str:
    """Normalize a name for use in filename.

    Args:
        name: Name to normalize

    Returns:
        Normalized name safe for filenames
    """
    # Replace special characters with underscores
    normalized = re.sub(r"[^\w\-.]", "_", name)
    # Remove multiple consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized


def normalize_id(name: str) -> str:
    """Normalize a name for use in markdown anchor IDs.

    Args:
        name: Name to normalize

    Returns:
        Normalized name safe for anchor IDs (lowercase, hyphens)
    """
    # Convert to lowercase
    normalized = name.lower()
    # Replace spaces and underscores with hyphens
    normalized = re.sub(r"[\s_]+", "-", normalized)
    # Remove any remaining special characters except hyphens
    normalized = re.sub(r"[^\w\-]", "", normalized)
    # Remove multiple consecutive hyphens
    normalized = re.sub(r"-+", "-", normalized)
    # Remove leading/trailing hyphens
    normalized = normalized.strip("-")
    return normalized


class _TaskAutogen:
    """Handles autogeneration of documentation for a single task."""

    def __init__(self, task_ir):
        """Initialize with a TaskIntermediateRepresentation.

        Args:
            task_ir: TaskIntermediateRepresentation object
        """
        self.task_ir = task_ir

    def generate_yaml(self) -> dict:
        """Generate YAML dictionary from task IR.

        Returns:
            Dictionary ready for YAML serialization
        """
        return {
            "name": self.task_ir.name,
            "description": self.task_ir.description,
            "harness": self.task_ir.harness,
            "container": self.task_ir.container,
            "container_digest": self.task_ir.container_digest,
            "defaults": self.task_ir.defaults,
        }

    def generate_markdown_section(self, harness_id: str) -> list[str]:
        """Generate markdown section for this task.

        Args:
            harness_id: Normalized harness ID for anchor

        Returns:
            List of markdown lines
        """
        lines = []
        task_id = f"{harness_id}-{normalize_id(self.task_ir.name)}"

        # Use MyST label syntax for explicit anchor IDs
        lines.append(f"({task_id})=")
        lines.append(f"## {self.task_ir.name}")
        lines.append("")

        if self.task_ir.description:
            lines.append(f"{self.task_ir.description}")
            lines.append("")

        # Extract command and defaults separately
        command = None
        defaults_without_command = {}
        if self.task_ir.defaults:
            defaults_without_command = copy.deepcopy(self.task_ir.defaults)
            # Check for command at top level
            if "command" in defaults_without_command:
                command = defaults_without_command.pop("command")
            # Check for command in config
            elif "config" in defaults_without_command and isinstance(
                defaults_without_command["config"], dict
            ):
                config = defaults_without_command["config"]
                if "command" in config:
                    command = config.pop("command")
                # Also check nested config.params if command not found in config
                elif "params" in config and isinstance(config["params"], dict):
                    params = config["params"]
                    if "command" in params:
                        command = params.pop("command")

        # Unfoldable details section using HTML details tag
        lines.append("<details>")
        lines.append("<summary><strong>View task details</strong></summary>")
        lines.append("")

        lines.append(f"**Harness:** `{self.task_ir.harness}`")
        lines.append("")
        lines.append("**Container:**")
        lines.append("```")
        lines.append(self.task_ir.container)
        lines.append("```")
        lines.append("")
        if self.task_ir.container_digest:
            lines.append("**Container Digest:**")
            lines.append("```")
            lines.append(self.task_ir.container_digest)
            lines.append("```")
            lines.append("")

        # Show command section if present
        if command:
            lines.append("**Command:**")
            lines.append("```bash")
            if isinstance(command, str):
                # Output command as-is, let CSS handle wrapping
                # Simple fallback: break at existing newlines or spaces if very long
                lines.append(command)
            else:
                lines.append(str(command))
            lines.append("```")
            lines.append("")

        # Show defaults section (without command)
        if defaults_without_command:
            lines.append("**Defaults:**")
            lines.append("```yaml")
            defaults_yaml = yaml.dump(
                defaults_without_command, default_flow_style=False, sort_keys=False
            )
            lines.append(defaults_yaml)
            lines.append("```")
            lines.append("")

        lines.append("</details>")
        lines.append("")
        lines.append("")

        return lines


class _HarnessAutogen:
    """Handles autogeneration of documentation for a harness and its tasks."""

    def __init__(self, harness_ir, tasks: list):
        """Initialize with harness IR and list of tasks.

        Args:
            harness_ir: HarnessIntermediateRepresentation object
            tasks: List of TaskIntermediateRepresentation objects for this harness
        """
        self.harness_ir = harness_ir
        self.harness_name = harness_ir.name
        self.tasks = sorted(tasks, key=lambda t: t.name)
        self.harness_id = normalize_id(self.harness_name)
        self.harness_filename = normalize_filename(self.harness_name)

    def generate_markdown_page(self) -> str:
        """Generate complete markdown page for this harness.

        Returns:
            Complete markdown content as string
        """
        lines = []
        lines.append(f"# {self.harness_name}")
        lines.append("")
        lines.append(
            f"This page contains all evaluation tasks for the **{self.harness_name}** harness."
        )
        lines.append("")

        # Create a table listing all tasks in this harness
        lines.append("```{list-table}")
        lines.append(":header-rows: 1")
        lines.append(":widths: 30 70")
        lines.append("")
        lines.append("* - Task")
        lines.append("  - Description")

        for task in self.tasks:
            task_autogen = _TaskAutogen(task)
            task_id = f"{self.harness_id}-{normalize_id(task.name)}"
            description = task.description if task.description else ""
            # Use regular markdown link with explicit anchor IDs
            lines.append(f"* - [{task.name}](#{task_id})")
            lines.append(f"  - {description}")

        lines.append("```")
        lines.append("")

        # Generate detailed sections for each task
        for task in self.tasks:
            task_autogen = _TaskAutogen(task)
            task_lines = task_autogen.generate_markdown_section(self.harness_id)
            lines.extend(task_lines)

        return "\n".join(lines)

    def generate_card_markdown(self, harnesses_dir: str) -> str:
        """Generate card markdown for this harness in the catalog.

        Args:
            harnesses_dir: Relative path to harnesses directory (from docs/, e.g., ../_resources/harnesses_autogen)

        Returns:
            Card markdown content
        """
        # Omit .md extension - Sphinx/MyST handles it automatically
        harness_file = f"{harnesses_dir}/{self.harness_filename}"
        task_count = len(self.tasks)
        task_text = f"{task_count} task{'s' if task_count != 1 else ''}"

        # Use harness description if available, otherwise use task count
        description = (
            self.harness_ir.description if self.harness_ir.description else task_text
        )
        # If we have description, append task count
        if self.harness_ir.description:
            description = f"{description}\n\n{task_text}"

        # Generate card using MyST card syntax (3 colons for grid-item-card)
        card_lines = []
        card_lines.append(f":::{{grid-item-card}} {self.harness_name}")
        card_lines.append(f":link: {harness_file}")
        card_lines.append(":link-type: doc")
        card_lines.append(f"{description}")
        card_lines.append(":::")

        return "\n".join(card_lines)


def generate_tasks_catalog_markdown(
    harnesses: list[_HarnessAutogen], harnesses_dir: str
) -> str:
    """Generate markdown catalog with cards linking to harness pages.

    Args:
        harnesses: List of _HarnessAutogen objects
        harnesses_dir: Relative path to harnesses directory (from docs/)

    Returns:
        Markdown content as string
    """
    lines = []
    lines.append("# Tasks Catalog")
    lines.append("")
    lines.append(
        "This page provides an overview of all available evaluation harnesses. "
        "Click on a harness card to view its tasks."
    )
    lines.append("")

    # Generate harness cards in a grid (3 columns)
    if not harnesses:
        lines.append("No harnesses found.")
        lines.append("")
    else:
        lines.append("::::{grid} 1 2 2 2")
        lines.append(":gutter: 1 1 1 2")
        lines.append("")

        for harness in harnesses:
            card_content = harness.generate_card_markdown(harnesses_dir)
            lines.append(card_content)
            lines.append("")

    lines.append("::::")
    lines.append("")

    # Add toctree for navigation hierarchy
    if harnesses:
        lines.append(":::{toctree}")
        lines.append(":caption: Harnesses")
        lines.append(":hidden:")
        lines.append("")
        for harness in harnesses:
            # Use relative path from tasks-catalog.md to harness pages
            harness_path = f"../_resources/harnesses_autogen/{harness.harness_filename}"
            lines.append(f"{harness.harness_name} <{harness_path}>")
        lines.append(":::")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Autogenerate task YAML files from intermediate representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all task YAMLs
  python scripts/autogen_task_yamls.py

  # Custom frameworks file and output directory
  python scripts/autogen_task_yamls.py --frameworks-file custom.yaml --output-dir docs/tasks/

  # Dry run to preview
  python scripts/autogen_task_yamls.py --dry-run

  # Filter by harness
  python scripts/autogen_task_yamls.py --harness simple-evals

  # Filter by task name
  python scripts/autogen_task_yamls.py --task mmlu
        """,
    )
    parser.add_argument(
        "--frameworks-file",
        type=pathlib.Path,
        default=None,
        help="Path to all_frameworks.yaml file (default: auto-detect)",
    )
    parser.add_argument(
        "--catalog-file",
        type=pathlib.Path,
        default=None,
        help="Path to tasks catalog markdown file (default: repo_root/docs/evaluation/tasks-catalog.md)",
    )
    parser.add_argument(
        "--harnesses-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for generated harness markdown files (default: repo_root/docs/_resources/harnesses_autogen)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be generated without writing files",
    )
    parser.add_argument(
        "--harness",
        type=str,
        default=None,
        help="Filter by harness name (optional)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task name (optional)",
    )

    args = parser.parse_args()

    # Find repository root
    repo_root = find_repo_root(pathlib.Path(__file__))

    # Set default paths relative to repo root if not provided
    if args.catalog_file is None:
        args.catalog_file = repo_root / "docs" / "evaluation" / "tasks-catalog.md"
    if args.harnesses_dir is None:
        args.harnesses_dir = repo_root / "docs" / "_resources" / "harnesses_autogen"

    logger.info(
        "Starting documentation autogeneration",
        repo_root=str(repo_root),
        frameworks_file=str(args.frameworks_file)
        if args.frameworks_file
        else "auto-detect",
        catalog_file=str(args.catalog_file),
        dry_run=args.dry_run,
        harness_filter=args.harness,
        task_filter=args.task,
    )

    # Load tasks
    tasks = _load_tasks_from_frameworks(args.frameworks_file)

    if not tasks:
        logger.warning("No tasks found")
        sys.exit(1)

    # Load harness IRs (use same frameworks_file path as tasks)
    harness_irs = _load_harnesses_from_frameworks(args.frameworks_file)

    logger.info(
        "Loaded harness IRs",
        count=len(harness_irs),
        harnesses=list(harness_irs.keys())[:5] if harness_irs else [],
    )

    # Filter tasks
    filtered_tasks = []
    for task in tasks:
        if args.harness and task.harness != args.harness:
            continue
        if args.task and task.name != args.task:
            continue
        filtered_tasks.append(task)

    if not filtered_tasks:
        logger.warning(
            "No tasks match filters",
            harness_filter=args.harness,
            task_filter=args.task,
        )
        sys.exit(1)

    logger.info(
        "Tasks to process",
        total=len(filtered_tasks),
        filtered_from=len(tasks),
    )

    # Create output directories if not dry run
    if not args.dry_run:
        # Remove existing harnesses directory for clean regeneration
        if args.harnesses_dir.exists():
            logger.info(
                "Removing existing harnesses directory",
                path=str(args.harnesses_dir),
            )
            shutil.rmtree(args.harnesses_dir)

        # Create fresh directory
        args.harnesses_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created harnesses directory", path=str(args.harnesses_dir))

    # Summary
    if args.dry_run:
        logger.info(
            "Dry run complete",
            total_tasks=len(filtered_tasks),
        )
        print(f"\nWould generate catalog: {args.catalog_file}")
    else:
        # Group tasks by harness and create harness autogen objects
        tasks_by_harness = defaultdict(list)
        for task in filtered_tasks:
            tasks_by_harness[task.harness].append(task)

        # Create harness autogen objects with harness IRs (already loaded above)
        harnesses = []
        for harness_name, tasks in sorted(tasks_by_harness.items()):
            harness_ir = harness_irs.get(harness_name)
            if not harness_ir:
                # Create a minimal harness IR if not found
                from nemo_evaluator_launcher.common.task_ir import (
                    HarnessIntermediateRepresentation,
                )

                container = tasks[0].container if tasks else ""
                container_digest = tasks[0].container_digest if tasks else None
                harness_ir = HarnessIntermediateRepresentation(
                    name=harness_name,
                    description="",
                    full_name=None,
                    url=None,
                    source=None,
                    container=container,
                    container_digest=container_digest,
                )
            harnesses.append(_HarnessAutogen(harness_ir, tasks))

        # Generate harness markdown pages
        harnesses_successful = 0
        harnesses_failed = 0
        harnesses_dir_rel = "../_resources/harnesses_autogen"

        for harness in harnesses:
            harness_file = args.harnesses_dir / f"{harness.harness_filename}.md"
            try:
                harness_content = harness.generate_markdown_page()
                with open(harness_file, "w", encoding="utf-8") as f:
                    f.write(harness_content)
                harnesses_successful += 1
                logger.debug(
                    "Generated harness page",
                    harness=harness.harness_name,
                    path=str(harness_file),
                )
            except Exception as e:
                harnesses_failed += 1
                logger.error(
                    "Failed to generate harness page",
                    harness=harness.harness_name,
                    path=str(harness_file),
                    error=str(e),
                    exc_info=True,
                )

        # Generate tasks catalog markdown
        try:
            catalog_content = generate_tasks_catalog_markdown(
                harnesses, harnesses_dir_rel
            )
            args.catalog_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.catalog_file, "w", encoding="utf-8") as f:
                f.write(catalog_content)
            logger.info(
                "Generated tasks catalog",
                path=str(args.catalog_file),
            )
            print(f"Generated tasks catalog: {args.catalog_file}")
        except Exception as e:
            logger.error(
                "Failed to generate tasks catalog",
                path=str(args.catalog_file),
                error=str(e),
                exc_info=True,
            )
            print(f"Warning: Failed to generate catalog: {e}")

        logger.info(
            "Documentation generation complete",
            total_tasks=len(filtered_tasks),
            harnesses_generated=harnesses_successful,
        )
        if harnesses_successful > 0:
            print(
                f"\nGenerated {harnesses_successful} harness page(s) in {args.harnesses_dir}"
            )
        if harnesses_failed > 0:
            print(f"Failed to generate {harnesses_failed} harness page(s)")
            sys.exit(1)


if __name__ == "__main__":
    main()
