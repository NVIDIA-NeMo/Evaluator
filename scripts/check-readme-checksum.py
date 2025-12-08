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
"""Script to check that mapping.toml checksum in README.md matches the actual checksum.

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
scripts/update-readme.py
(updates README, records checksum)           <----- pre-commit guard: checks TOML checksum
"""

import argparse
import hashlib
import pathlib
import re
import sys


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


def extract_checksum_from_readme(readme_path: pathlib.Path) -> str | None:
    """Extract checksum from README.md comment.

    Args:
        readme_path: Path to README.md file

    Returns:
        Checksum string if found, None otherwise
    """
    try:
        content = readme_path.read_text(encoding="utf-8")

        # Look for checksum comment (allow flexible whitespace)
        # Match both the full comment line and extract the checksum value
        # Pattern matches: sha256:hexdigest or sha256:PLACEHOLDER
        pattern = r"<!--\s*mapping\s+toml\s+checksum:\s*(sha256:[a-f0-9]+|sha256:PLACEHOLDER)\s*-->"
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            checksum = match.group(1)
            # Return None if placeholder, so we can detect it
            if "PLACEHOLDER" in checksum.upper():
                return None
            return checksum

        # Try a simpler pattern as fallback (without HTML comment markers)
        pattern2 = r"mapping\s+toml\s+checksum:\s*(sha256:[a-f0-9]+|sha256:PLACEHOLDER)"
        match2 = re.search(pattern2, content, re.IGNORECASE)
        if match2:
            checksum = match2.group(1)
            if "PLACEHOLDER" in checksum.upper():
                return None
            return checksum

        return None
    except Exception as e:
        print(f"ERROR: Failed to read README.md: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Check that mapping.toml checksum in README.md matches actual checksum"
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

    # Check files exist
    if not args.readme_file.exists():
        print(f"ERROR: README.md not found: {args.readme_file}", file=sys.stderr)
        sys.exit(1)
    if not args.mapping_file.exists():
        print(f"ERROR: mapping.toml not found: {args.mapping_file}", file=sys.stderr)
        sys.exit(1)

    # Calculate actual checksum
    actual_checksum = calculate_mapping_checksum(args.mapping_file)

    # Extract checksum from README.md
    readme_checksum = extract_checksum_from_readme(args.readme_file)

    if readme_checksum is None:
        # Check if file was read correctly
        try:
            content = args.readme_file.read_text(encoding="utf-8")
            if "mapping toml checksum" not in content:
                print(
                    "ERROR: Could not find checksum comment in README.md. "
                    "Expected: <!-- mapping toml checksum: sha256:... -->",
                    file=sys.stderr,
                )
            else:
                print(
                    "ERROR: Checksum comment found but contains PLACEHOLDER. "
                    "Please run: python scripts/update-readme.py",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"ERROR: Failed to read README.md: {e}",
                file=sys.stderr,
            )
        sys.exit(1)

    # Compare checksums
    if actual_checksum != readme_checksum:
        print(
            f"ERROR: Checksum mismatch!\n"
            f"  mapping.toml checksum: {actual_checksum}\n"
            f"  README.md checksum:    {readme_checksum}\n"
            f"\n"
            f"Please run: python scripts/update-readme.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"✓ Checksums match: {actual_checksum}")
    sys.exit(0)


if __name__ == "__main__":
    main()
