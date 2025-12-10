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
"""Script to update the README.md table with harness and task information.

Workflow diagram:
mapping.toml
(container = "..")
(# container-digest:sha256:....)              <---- CI: check digests are relevant

   |
   |
   v
scripts/load_framework_definitions.py
(updates the toml
 AND
creates the resources/all_tasks_irs,
records TOML checksum)                        <---- pre-commit guard: checks TOML checksum
   |          \
   |           \
   |            ------------------->    make docs-build
   |                                  (builds docs on the fly)
   v
scripts/update_readme.py
(updates README, records checksum)           <----- pre-commit guard: checks TOML checksum
"""

import argparse
import pathlib
import re
import sys

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.task_ir import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
    load_tasks_from_tasks_file,
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
    return start_path.parent.parent


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


def generate_readme_table(
    harnesses: dict[
        str,
        tuple[HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation]],
    ],
    checksum: str,
) -> str:
    """Generate README.md table markdown inside HTML comments.

    Args:
        harnesses: Dictionary mapping harness name to (harness_ir, tasks) tuple
        checksum: Mapping.toml checksum

    Returns:
        Markdown table content as string (table wrapped in single HTML comment block)
    """
    lines = []
    lines.append("<!-- BEGIN AUTOGENERATION -->")
    lines.append(f"<!-- mapping toml checksum: {checksum} -->")

    # Generate table content first
    table_lines = []
    table_lines.append(
        "| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |"
    )
    table_lines.append(
        "|-----------|-------------|-------------|------------| ------------|"
    )

    # Sort harnesses alphabetically for consistent ordering
    sorted_harnesses = sorted(harnesses.items(), key=lambda x: x[0].lower())

    for harness_name, (harness_ir, tasks) in sorted_harnesses:
        # Get container from harness IR or first task
        container = harness_ir.container
        if not container and tasks:
            container = tasks[0].container

        container_name, version = extract_container_name_and_version(container)

        # Generate NGC catalog link
        if container_name:
            ngc_url = f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/{container_name}"
            if version:
                ngc_url += f"?version={version}"
            ngc_link = f"[Link]({ngc_url})"
        else:
            ngc_link = "N/A"

        # Get description
        description = harness_ir.description or ""
        if not description and tasks:
            # Try to get description from first task
            description = tasks[0].description or ""

        # Generate task names list (just names, comma-separated)
        task_names = [task.name for task in tasks]
        tasks_display = ", ".join(task_names)

        # Use version from container tag (sourced from container image identifier)
        # If no version found, use placeholder as fallback
        latest_tag = version if version else "25.11"

        # Escape special characters in markdown (but preserve links)
        description_display = description.replace("|", "\\|").replace("\n", " ")

        # Format container name as bold
        container_display = f"**{harness_name}**"

        table_lines.append(
            f"| {container_display} | {description_display} | {ngc_link} | `{latest_tag}` | {tasks_display} |"
        )

    # Wrap entire table in a single HTML comment block
    lines.append("<!--")
    for table_line in table_lines:
        lines.append(table_line)
    lines.append("-->")
    lines.append("<!-- END AUTOGENERATION -->")

    return "\n".join(lines)


def update_readme_table(readme_path: pathlib.Path, table_content: str) -> None:
    """Update the README.md file with new table content.

    Args:
        readme_path: Path to README.md file
        table_content: New table content to insert
    """
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the autogeneration section
    begin_pattern = r"<!-- BEGIN AUTOGENERATION -->"
    end_pattern = r"<!-- END AUTOGENERATION -->"

    begin_match = re.search(begin_pattern, content)
    end_match = re.search(end_pattern, content)

    if not begin_match:
        raise ValueError(
            "README.md does not contain '<!-- BEGIN AUTOGENERATION -->' marker"
        )
    if not end_match:
        raise ValueError(
            "README.md does not contain '<!-- END AUTOGENERATION -->' marker"
        )
    if begin_match.end() > end_match.start():
        raise ValueError("BEGIN marker appears after END marker in README.md")

    # Replace the content between markers
    # Extract everything BEFORE the BEGIN tag (not including it)
    before = content[: begin_match.start()]
    # Extract everything AFTER the END tag (not including it)
    # table_content already includes BEGIN and END tags, so we exclude them from before/after
    after = content[end_match.end() :]

    # table_content already includes BEGIN and END tags, so just insert it
    new_content = before + table_content + after

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info("Updated README.md table", path=str(readme_path))


def calculate_mapping_checksum(mapping_file: pathlib.Path) -> str:
    """Calculate SHA256 checksum of mapping.toml file.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        SHA256 checksum as string (format: "sha256:...")
    """
    import hashlib

    try:
        with open(mapping_file, "rb") as f:
            file_content = f.read()

        checksum = hashlib.sha256(file_content).hexdigest()
        return f"sha256:{checksum}"
    except Exception as e:
        logger.error(
            "Failed to calculate mapping.toml checksum", error=str(e), exc_info=True
        )
        raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Update README.md table with harness and task information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--readme-file",
        type=pathlib.Path,
        default=None,
        help="Path to README.md file (default: repo_root/README.md)",
    )
    parser.add_argument(
        "--mapping-file",
        type=pathlib.Path,
        default=None,
        help="Path to mapping.toml file (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be generated without writing files",
    )

    args = parser.parse_args()

    # Find repository root
    repo_root = find_repo_root(pathlib.Path(__file__))

    # Set default paths relative to repo root if not provided
    if args.readme_file is None:
        args.readme_file = repo_root / "README.md"
    if args.mapping_file is None:
        args.mapping_file = (
            repo_root
            / "packages"
            / "nemo-evaluator-launcher"
            / "src"
            / "nemo_evaluator_launcher"
            / "resources"
            / "mapping.toml"
        )

    logger.info(
        "Starting README.md table update",
        repo_root=str(repo_root),
        readme_file=str(args.readme_file),
        mapping_file=str(args.mapping_file),
        dry_run=args.dry_run,
    )

    # Calculate mapping.toml checksum
    if not args.mapping_file.exists():
        logger.error("Mapping file not found", path=str(args.mapping_file))
        sys.exit(1)

    checksum = calculate_mapping_checksum(args.mapping_file)
    logger.info("Calculated mapping.toml checksum", checksum=checksum)

    # Load tasks
    tasks, mapping_verified = load_tasks_from_tasks_file()
    if not tasks:
        logger.error("No tasks found")
        sys.exit(1)

    if not mapping_verified:
        logger.warning(
            "mapping.toml checksum mismatch - all_tasks_irs.yaml may be out of sync"
        )

    # Group tasks by harness (harnesses are created from tasks)
    harnesses: dict[
        str,
        tuple[HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation]],
    ] = {}
    for task in tasks:
        if not task.harness:
            logger.warning(
                "Task missing harness name, skipping",
                task_name=task.name,
            )
            continue

        if task.harness not in harnesses:
            # Create harness IR from task
            container = str(task.container) if task.container else ""
            container_digest = (
                str(task.container_digest) if task.container_digest else None
            )

            harness_ir = HarnessIntermediateRepresentation(
                name=task.harness,
                description="",  # Description not available from tasks alone
                full_name=None,
                url=None,
                source=None,
                container=container,
                container_digest=container_digest,
            )
            harnesses[task.harness] = (harness_ir, [])

        harnesses[task.harness][1].append(task)

    logger.info(
        "Loaded harnesses",
        count=len(harnesses),
        harnesses=list(harnesses.keys())[:5] if harnesses else [],
    )

    # Generate table
    table_content = generate_readme_table(harnesses, checksum)

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Would update README.md with:")
        print("=" * 80)
        print(table_content)
        print("=" * 80)
    else:
        # Update README.md
        update_readme_table(args.readme_file, table_content)
        print(f"✓ Updated README.md table (checksum: {checksum})")


if __name__ == "__main__":
    main()
