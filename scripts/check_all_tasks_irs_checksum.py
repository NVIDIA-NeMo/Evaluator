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
"""Script to check that mapping.toml checksum in all_tasks_irs.yaml matches the actual checksum.

Workflow diagram:
mapping.toml
(container = "....")
(# container-digest:sha256:....)              <---- CI: check digests are relevant

   |
   |
   v
scripts/load_framework_definitions.py
(updates the toml
 AND
creates the resources/all_tasks_irs,
records TOML checksum)                        <---- pre-commit guard: checks TOML checksum
                                                      pre-commit guard: checks all_tasks_irs checksum
   |          \
   |           \
   |            ------------------->    make docs-build
   |                                  (builds docs on the fly)
   v
scripts/update_readme.py
(updates README, records checksum)           <----- pre-commit guard: checks TOML checksum
"""

import argparse
import hashlib
import pathlib
import sys

import yaml


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


def calculate_mapping_checksum(mapping_file: pathlib.Path) -> str:
    """Calculate SHA256 checksum of mapping.toml file.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        SHA256 checksum as string (format: "sha256:...")
    """
    try:
        with open(mapping_file, "rb") as f:
            file_content = f.read()

        checksum = hashlib.sha256(file_content).hexdigest()
        return f"sha256:{checksum}"
    except Exception as e:
        print(f"ERROR: Failed to calculate mapping.toml checksum: {e}", file=sys.stderr)
        sys.exit(1)


def extract_checksum_from_all_tasks_irs(all_tasks_irs_file: pathlib.Path) -> str | None:
    """Extract mapping_toml_checksum from all_tasks_irs.yaml metadata.

    Args:
        all_tasks_irs_file: Path to all_tasks_irs.yaml file

    Returns:
        Checksum string (format: "sha256:...") or None if not found
    """
    try:
        with open(all_tasks_irs_file, "r", encoding="utf-8") as f:
            content = f.read()

        yaml_data = yaml.safe_load(content)
        if not yaml_data:
            return None

        metadata = yaml_data.get("metadata", {})
        return metadata.get("mapping_toml_checksum")
    except Exception as e:
        print(
            f"ERROR: Failed to read all_tasks_irs.yaml: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Check that mapping.toml checksum in all_tasks_irs.yaml matches the actual checksum"
    )
    parser.add_argument(
        "--all-tasks-irs-file",
        type=pathlib.Path,
        default=None,
        help="Path to all_tasks_irs.yaml file (default: auto-detect)",
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

    # Set default paths if not provided
    if args.all_tasks_irs_file is None:
        args.all_tasks_irs_file = (
            repo_root
            / "packages"
            / "nemo-evaluator-launcher"
            / "src"
            / "nemo_evaluator_launcher"
            / "resources"
            / "all_tasks_irs.yaml"
        )

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

    # Check files exist
    if not args.all_tasks_irs_file.exists():
        print(
            f"ERROR: all_tasks_irs.yaml not found: {args.all_tasks_irs_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.mapping_file.exists():
        print(
            f"ERROR: mapping.toml not found: {args.mapping_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract stored checksum from all_tasks_irs.yaml
    stored_checksum = extract_checksum_from_all_tasks_irs(args.all_tasks_irs_file)
    if not stored_checksum:
        print(
            "ERROR: Could not find mapping_toml_checksum in all_tasks_irs.yaml metadata",
            file=sys.stderr,
        )
        sys.exit(1)

    # Calculate current checksum of mapping.toml
    current_checksum = calculate_mapping_checksum(args.mapping_file)

    # Compare checksums
    if stored_checksum != current_checksum:
        print("ERROR: Checksum mismatch detected:", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            f"  Stored in all_tasks_irs.yaml: {stored_checksum}",
            file=sys.stderr,
        )
        print(
            f"  Current mapping.toml:         {current_checksum}",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print(
            "all_tasks_irs.yaml is out of sync with mapping.toml.",
            file=sys.stderr,
        )
        print(
            "Please regenerate it by running:",
            file=sys.stderr,
        )
        print(
            "  python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print("✓ Checksum in all_tasks_irs.yaml matches mapping.toml")
    sys.exit(0)


if __name__ == "__main__":
    main()
