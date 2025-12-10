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
"""Script to check that all container entries in mapping.toml have digest comments.

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

# Maximum number of lines to check after container declaration for digest comment
MAX_DIGEST_COMMENT_OFFSET_LINES = 4


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


def check_container_digests(mapping_file: pathlib.Path) -> tuple[bool, list[str]]:
    """Check that all container entries in mapping.toml have digest comments.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        Tuple of (all_valid: bool, errors: list[str])
    """
    if not mapping_file.exists():
        return False, [f"Mapping file not found: {mapping_file}"]

    # Read file as text to check comments
    with open(mapping_file, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    lines = content.split("\n")

    # Find all container entries and their line numbers
    container_entries = []
    for i, line in enumerate(lines, start=1):
        # Match container = "..." lines (harness-level sections only)
        # Pattern: container = "value" (with optional whitespace)
        match = re.match(r'^\s*container\s*=\s*"([^"]+)"', line)
        if match:
            container = match.group(1)
            # Check if this is a harness-level section (not a task-level)
            # Look backwards to find the section header
            # i is 1-indexed (from enumerate start=1), convert to 0-indexed for array access
            section_line_idx = i - 1  # Convert to 0-indexed
            while section_line_idx >= 0 and not lines[
                section_line_idx
            ].strip().startswith("["):
                section_line_idx -= 1
            if section_line_idx >= 0:
                section_line = lines[section_line_idx].strip()
                # Harness-level sections don't have dots (e.g., [lm-evaluation-harness])
                # Task-level sections have dots (e.g., [lm-evaluation-harness.tasks.chat.mmlu])
                if "." not in section_line.strip("[]"):
                    container_entries.append((i, container, section_line))

    # Check each container entry has a digest comment on the next line
    for line_num, container, section in container_entries:
        # Check if next line (or lines after) has digest comment
        digest_found = False
        # Look at next few lines for digest comment (allow for blank lines)
        for offset in range(1, MAX_DIGEST_COMMENT_OFFSET_LINES + 1):
            if line_num + offset - 1 >= len(lines):
                break
            next_line = lines[line_num + offset - 1].strip()
            # Pattern: # container-digest:sha256:... or # container-digest: sha256:...
            if re.match(
                r"#\s*container-digest:\s*sha256:[a-f0-9]+", next_line, re.IGNORECASE
            ):
                digest_found = True
                break
            # Stop if we hit another section or key
            if next_line.startswith("[") or (
                next_line and not next_line.startswith("#") and "=" in next_line
            ):
                break

        if not digest_found:
            errors.append(
                f"Line {line_num}: Container '{container}' in section '{section}' "
                f"missing digest comment. Expected: # container-digest:sha256:..."
            )

    return len(errors) == 0, errors


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Check that all container entries in mapping.toml have digest comments"
    )
    parser.add_argument(
        "--mapping-file",
        type=pathlib.Path,
        default=None,
        help="Path to mapping.toml file (default: auto-detect)",
    )

    args = parser.parse_args()

    # Find repository root
    repo_root = find_repo_root(pathlib.Path(__file__))

    # Set default path if not provided
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

    # Check container digests
    all_valid, errors = check_container_digests(args.mapping_file)

    if not all_valid:
        print("ERROR: Container digest validation failed:", file=sys.stderr)
        print("", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Please run: python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print("✓ All container entries have digest comments")
    sys.exit(0)


if __name__ == "__main__":
    main()
