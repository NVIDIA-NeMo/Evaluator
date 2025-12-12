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

from nemo_evaluator_launcher.common.container_metadata import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
)
from nemo_evaluator_launcher.common.logging_utils import logger


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

    def __init__(self, task_ir: TaskIntermediateRepresentation) -> None:
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
            # Handle None or empty description safely
            desc = str(self.task_ir.description).strip()
            if desc:
                lines.append(desc)
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

        # Extract task type from defaults if available
        task_type = None
        if self.task_ir.defaults:
            config = self.task_ir.defaults.get("config", {})
            if isinstance(config, dict):
                task_type = config.get("type")

        # Create tab set with three tabs: Container, Command, Defaults
        lines.append("::::{tab-set}")
        lines.append("")

        # Tab 1: Container (harness name, container, digest, task type)
        lines.append(":::{tab-item} Container")
        lines.append("")
        lines.append(f"**Harness:** `{self.task_ir.harness}`")
        lines.append("")
        lines.append("**Container:**")
        lines.append("```")
        lines.append(str(self.task_ir.container))
        lines.append("```")
        lines.append("")
        if self.task_ir.container_digest:
            lines.append("**Container Digest:**")
            lines.append("```")
            lines.append(str(self.task_ir.container_digest))
            lines.append("```")
            lines.append("")
        if task_type:
            lines.append(f"**Task Type:** `{task_type}`")
            lines.append("")
        lines.append(":::")
        lines.append("")

        # Tab 2: Command (if present)
        if command:
            lines.append(":::{tab-item} Command")
            lines.append("")
            lines.append("```bash")
            if isinstance(command, str):
                # Output command as-is, let CSS handle wrapping
                lines.append(command)
            else:
                lines.append(str(command))
            lines.append("```")
            lines.append("")
            lines.append(":::")
            lines.append("")

        # Tab 3: Defaults
        if defaults_without_command:
            lines.append(":::{tab-item} Defaults")
            lines.append("")
            lines.append("```yaml")
            # Use better YAML formatting with proper width and allow_unicode
            defaults_yaml = yaml.dump(
                defaults_without_command,
                default_flow_style=False,
                sort_keys=False,
                width=120,  # Reasonable width for better readability
                allow_unicode=True,  # Allow unicode characters
            )
            lines.append(defaults_yaml.rstrip())  # Remove trailing newline
            lines.append("```")
            lines.append("")
            lines.append(":::")
            lines.append("")

        lines.append("::::")
        lines.append("")
        lines.append("")

        return lines


class _HarnessAutogen:
    """Handles autogeneration of documentation for a harness and its tasks."""

    def __init__(
        self,
        harness_ir: HarnessIntermediateRepresentation,
        tasks: list[TaskIntermediateRepresentation],
    ) -> None:
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
            # Handle None or empty description safely
            description = str(task.description).strip() if task.description else ""
            # Use regular markdown link with explicit anchor IDs
            lines.append(f"* - [{task.name}](#{task_id})")
            lines.append(f"  - {description}")

        lines.append("```")
        lines.append("")

        # Generate detailed sections for each task
        for i, task in enumerate(self.tasks):
            # Add horizontal separator between tasks (but not before the first one)
            if i > 0:
                lines.append("---")
                lines.append("")

            task_autogen = _TaskAutogen(task)
            task_lines = task_autogen.generate_markdown_section(self.harness_id)
            lines.extend(task_lines)

        return "\n".join(lines)

    def generate_card_markdown(self, harnesses_dir: str) -> str:
        """Generate card markdown for this harness in the catalog.

        Args:
            harnesses_dir: Relative path to harnesses directory (from index.md, e.g., harnesses)

        Returns:
            Card markdown content
        """
        # Omit .md extension - Sphinx/MyST handles it automatically
        harness_file = f"{harnesses_dir}/{self.harness_filename}"
        task_count = len(self.tasks)
        task_text = f"{task_count} task{'s' if task_count != 1 else ''}"

        # Use harness description if available, otherwise try to derive from tasks
        description = self.harness_ir.description
        if not description:
            # Try to derive description from task descriptions (use first non-empty task description)
            for task in self.tasks:
                if task.description:
                    # Use first sentence or first 100 chars of first task description
                    desc = task.description.strip()
                    # Extract first sentence if it ends with period
                    first_sentence = (
                        desc.split(".", 1)[0] + "." if "." in desc[:100] else desc[:100]
                    )
                    if len(desc) > 100:
                        first_sentence += "..."
                    description = first_sentence
                    break

        # Use task count as fallback if still no description
        if not description:
            description = task_text
        else:
            # Append task count if we have a description
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
        harnesses_dir: Relative path to harnesses directory (from index.md, e.g., harnesses)

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
        # Sort harnesses alphabetically for consistent ordering
        sorted_harnesses = sorted(harnesses, key=lambda h: h.harness_name.lower())

        lines.append("::::{grid} 1 2 2 2")
        lines.append(":gutter: 1 1 1 2")
        lines.append("")

        for harness in sorted_harnesses:
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
        # Sort harnesses alphabetically for consistent ordering
        sorted_harnesses = sorted(harnesses, key=lambda h: h.harness_name.lower())
        for harness in sorted_harnesses:
            # Use relative path from index.md to harness pages
            harness_path = f"{harnesses_dir}/{harness.harness_filename}"
            lines.append(f"{harness.harness_name} <{harness_path}>")
        lines.append(":::")
        lines.append("")

    return "\n".join(lines)


def extract_container_name_and_version(container: str) -> tuple[str, str]:
    """Extract container name and version from container image identifier.

    Args:
        container: Container image identifier (e.g., nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:25.11)

    Returns:
        Tuple of (container_name, version) where container_name is the part after eval-factory/
    """
    if not container:
        return "", ""

    # Extract version if present
    version = ""
    if ":" in container:
        container, version = container.rsplit(":", 1)

    # Extract container name (part after eval-factory/)
    container_name = ""
    if "eval-factory/" in container:
        container_name = container.split("eval-factory/")[-1]
    elif "/" in container:
        container_name = container.split("/")[-1]

    return container_name, version


def generate_benchmarks_table_markdown(
    harnesses: list[_HarnessAutogen],
    checksum: str | None = None,
) -> str:
    """Generate benchmarks table markdown for benchmarks.md.

    Args:
        harnesses: List of _HarnessAutogen objects
        checksum: Checksum from all_tasks_irs.yaml metadata (optional)

    Returns:
        Markdown table content as string
    """
    lines = []
    # Add comment indicating this is autogenerated
    lines.append("<!--")
    lines.append("This file is automatically generated by autogen_task_yamls.py")
    lines.append("DO NOT EDIT MANUALLY - changes will be overwritten")
    if checksum:
        lines.append(f"Generated from mapping.toml with checksum: {checksum}")
    lines.append("-->")
    lines.append("")

    # Generate toctree for sidebar navigation
    lines.append(":::{toctree}")
    lines.append(":caption: Harnesses")
    lines.append(":hidden:")
    lines.append("")
    # Sort harnesses alphabetically for toctree
    sorted_harnesses_for_toc = sorted(harnesses, key=lambda h: h.harness_name.lower())
    for harness in sorted_harnesses_for_toc:
        lines.append(
            f"{harness.harness_name} <all/harnesses/{harness.harness_filename}>"
        )
    lines.append(":::")
    lines.append("")

    lines.append("```{list-table}")
    lines.append(":header-rows: 1")
    lines.append(":widths: 20 25 15 15 25")
    lines.append("")
    lines.append("* - Container")
    lines.append("  - Description")
    lines.append("  - NGC Catalog")
    lines.append("  - Latest Tag")
    lines.append("  - Tasks")

    # Sort harnesses alphabetically for consistent ordering
    sorted_harnesses = sorted(harnesses, key=lambda h: h.harness_name.lower())

    for harness in sorted_harnesses:
        # Get container from harness IR or first task
        container = harness.harness_ir.container
        if not container and harness.tasks:
            container = harness.tasks[0].container

        container_name, version = extract_container_name_and_version(container)

        # Generate NGC catalog link
        if container_name:
            ngc_url = f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/{container_name}"
            if version:
                ngc_url += f"?version={version}"
            ngc_link = f"[NGC]({ngc_url})"
        else:
            ngc_link = "N/A"

        # Get description
        description = harness.harness_ir.description or ""
        if not description and harness.tasks:
            # Try to get description from first task
            description = harness.tasks[0].description or ""

        # Generate harness page link
        # benchmarks-table.md is at docs/evaluation/benchmarks/catalog/all/benchmarks-table.md
        # harness page is at docs/evaluation/benchmarks/catalog/all/harnesses/{filename}.md
        # Need relative path from benchmarks-table.md (same directory): harnesses/{filename}
        # But when included from benchmarks/index.md, need: catalog/all/harnesses/{filename}
        # We'll use the path relative to catalog/all/ since that's where benchmarks-table.md lives
        harness_page_path = f"all/harnesses/{harness.harness_filename}"
        # Link to harness page with harness anchor (harness_id is the normalized harness name)
        # The harness page heading creates an anchor with the harness_id
        harness_anchor = harness.harness_id
        container_display = f'<a href="{harness_page_path}.html#{harness_anchor}"><strong>{harness.harness_name}</strong></a>'

        # Generate task links with anchors to specific tasks on the harness page
        # Use same relative path format
        task_links = []
        for task in harness.tasks:
            task_id = f"{harness.harness_id}-{normalize_id(task.name)}"
            # HTML link with anchor - use relative path from benchmarks.md
            task_link = f'<a href="{harness_page_path}.html#{task_id}">{task.name}</a>'
            task_links.append(task_link)

        # Join task links with commas
        tasks_display = ", ".join(task_links)

        # Use version from container tag (sourced from container image identifier)
        # If no version found, use placeholder as fallback
        latest_tag = version if version else "{{ docker_compose_latest }}"

        # Escape special characters in markdown (but preserve links)
        description_display = description.replace("|", "\\|").replace("\n", " ")

        lines.append(f"* - {container_display}")
        lines.append(f"  - {description_display}")
        lines.append(f"  - {ngc_link}")
        lines.append(f"  - {latest_tag}")
        lines.append(f"  - {tasks_display}")

    lines.append("```")

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

  # Dry run to preview
  python scripts/autogen_task_yamls.py --dry-run

  # Filter by harness
  python scripts/autogen_task_yamls.py --harness simple-evals

  # Filter by task name
  python scripts/autogen_task_yamls.py --task mmlu
        """,
    )
    parser.add_argument(
        "--catalog-file",
        type=pathlib.Path,
        default=None,
        help="Path to tasks catalog markdown file (default: repo_root/docs/evaluation/benchmarks/catalog/index.md)",
    )
    parser.add_argument(
        "--harnesses-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for generated harness markdown files (default: repo_root/docs/evaluation/benchmarks/catalog/all/harnesses)",
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
        args.catalog_file = (
            repo_root / "docs" / "evaluation" / "benchmarks" / "catalog" / "index.md"
        )
    if args.harnesses_dir is None:
        args.harnesses_dir = (
            repo_root
            / "docs"
            / "evaluation"
            / "benchmarks"
            / "catalog"
            / "all"
            / "harnesses"
        )

    logger.info(
        "Starting documentation autogeneration",
        repo_root=str(repo_root),
        catalog_file=str(args.catalog_file),
        dry_run=args.dry_run,
        harness_filter=args.harness,
        task_filter=args.task,
    )

    # Load tasks (checksum validation happens automatically)
    tasks, mapping_verified = load_tasks_from_tasks_file()  # Uses default path

    # Check if mapping is verified (checksum mismatch)
    if not mapping_verified:
        logger.error(
            "mapping.toml checksum mismatch - all_tasks_irs.yaml is out of sync",
            message=(
                "all_tasks_irs.yaml is out of sync with mapping.toml. "
                "Please regenerate all_tasks_irs.yaml by running: "
                "python scripts/container_metadata_controller.py update"
            ),
        )
        sys.exit(1)

    if not tasks:
        logger.warning("No tasks found")
        sys.exit(1)

    # Derive harnesses from tasks
    harness_irs: dict[str, HarnessIntermediateRepresentation] = {}
    for task in tasks:
        if not task.harness:
            logger.warning(
                "Task missing harness name, skipping",
                task_name=task.name,
            )
            continue

        if task.harness not in harness_irs:
            # Use first task's container info for harness
            container = str(task.container) if task.container else ""
            container_digest = (
                str(task.container_digest) if task.container_digest else None
            )

            harness_irs[task.harness] = HarnessIntermediateRepresentation(
                name=task.harness,
                description="",  # Description not available from tasks alone
                full_name=None,
                url=None,
                container=container,
                container_digest=container_digest,
            )

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
            if not tasks:
                logger.warning(
                    "No tasks found for harness, skipping",
                    harness=harness_name,
                )
                continue

            harness_ir = harness_irs.get(harness_name)
            if not harness_ir:
                # This shouldn't happen since we derive harnesses from tasks
                # But create a minimal harness IR as fallback
                container = str(tasks[0].container) if tasks[0].container else ""
                container_digest = (
                    str(tasks[0].container_digest)
                    if tasks[0].container_digest
                    else None
                )
                harness_ir = HarnessIntermediateRepresentation(
                    name=harness_name,
                    description="",
                    full_name=None,
                    url=None,
                    container=container,
                    container_digest=container_digest,
                )
            harnesses.append(_HarnessAutogen(harness_ir, tasks))

        # Generate harness markdown pages
        harnesses_successful = 0
        harnesses_failed = 0

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

        # Note: task_catalog generation removed - everything is now in benchmarks table

        # Load checksum from all_tasks_irs.yaml for inclusion in generated table
        checksum = None
        try:
            import importlib.resources

            tasks_content = importlib.resources.read_text(
                "nemo_evaluator_launcher.resources",
                "all_tasks_irs.yaml",
                encoding="utf-8",
            )
            tasks_data = yaml.safe_load(tasks_content)
            checksum = tasks_data.get("metadata", {}).get("mapping_toml_checksum")
        except Exception as e:
            logger.warning(
                "Could not load checksum from all_tasks_irs.yaml", error=str(e)
            )

        # Generate benchmarks table file
        benchmarks_table_file = (
            repo_root
            / "docs"
            / "evaluation"
            / "benchmarks"
            / "catalog"
            / "all"
            / "benchmarks-table.md"
        )
        try:
            table_content = generate_benchmarks_table_markdown(
                harnesses, checksum=checksum
            )
            benchmarks_table_file.parent.mkdir(parents=True, exist_ok=True)
            with open(benchmarks_table_file, "w", encoding="utf-8") as f:
                f.write(table_content)
            logger.info("Generated benchmarks table", path=str(benchmarks_table_file))
            print(f"Generated benchmarks table: {benchmarks_table_file}")
        except Exception as e:
            logger.error(
                "Failed to generate benchmarks table",
                path=str(benchmarks_table_file),
                error=str(e),
                exc_info=True,
            )
            print(f"Warning: Failed to generate benchmarks table: {e}")

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
