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
"""Script to validate that container digests in mapping.toml match actual registry digests.

This script reads mapping.toml, extracts container images and their digest comments,
fetches the actual digests from the registry, and compares them to ensure they match.
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


def extract_container_digests_from_mapping(
    mapping_file: pathlib.Path,
) -> dict[str, str]:
    """Extract container images and their digest comments from mapping.toml.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        Dictionary mapping container image to digest (sha256:...)
    """
    if not mapping_file.exists():
        return {}

    with open(mapping_file, "r", encoding="utf-8") as f:
        content = f.read()

    container_digests = {}
    lines = content.split("\n")

    # Find all container entries and their digest comments
    for i, line in enumerate(lines):
        # Match container = "..." lines (harness-level sections only)
        match = re.match(r'^\s*container\s*=\s*"([^"]+)"', line)
        if match:
            container = match.group(1)
            # Check if this is a harness-level section
            section_line_idx = i
            while section_line_idx >= 0 and not lines[
                section_line_idx
            ].strip().startswith("["):
                section_line_idx -= 1
            if section_line_idx >= 0:
                section_line = lines[section_line_idx].strip()
                # Harness-level sections don't have dots
                if "." not in section_line.strip("[]"):
                    # Look for digest comment in next few lines
                    for offset in range(1, MAX_DIGEST_COMMENT_OFFSET_LINES + 1):
                        if i + offset >= len(lines):
                            break
                        next_line = lines[i + offset].strip()
                        # Pattern: # container-digest:sha256:...
                        digest_match = re.match(
                            r"#\s*container-digest:\s*(sha256:[a-f0-9]+)",
                            next_line,
                            re.IGNORECASE,
                        )
                        if digest_match:
                            container_digests[container] = digest_match.group(1).lower()
                            break
                        # Stop if we hit another section or key
                        if next_line.startswith("[") or (
                            next_line
                            and not next_line.startswith("#")
                            and "=" in next_line
                        ):
                            break

    return container_digests


def validate_container_digests(
    mapping_file: pathlib.Path,
) -> tuple[bool, list[str], list[str]]:
    """Validate that container digests in mapping.toml match actual registry digests.

    Args:
        mapping_file: Path to mapping.toml file

    Returns:
        Tuple of (all_valid: bool, all_errors: list[str], mismatches: list[str])
        where all_errors includes both errors and mismatches,
        and mismatches contains only digest mismatch errors
    """
    # Import here to avoid import errors if package not installed
    try:
        import sys

        # Add scripts directory to path so we can import from load_framework_definitions
        repo_root = find_repo_root(mapping_file.parent)
        scripts_dir = repo_root / "packages" / "nemo-evaluator-launcher" / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from load_framework_definitions import (
            create_authenticator,
            get_container_digest,
        )
        from nemo_evaluator_launcher.common.helpers import parse_container_image
    except ImportError as e:
        return (
            False,
            [
                f"Failed to import required modules: {e}. "
                "Make sure nemo-evaluator-launcher package is installed."
            ],
            [],
        )

    # Extract container digests from mapping.toml
    container_digests = extract_container_digests_from_mapping(mapping_file)

    if not container_digests:
        return (
            False,
            [
                "No container digests found in mapping.toml. "
                "Make sure mapping.toml contains container entries with digest comments."
            ],
            [],
        )

    errors = []
    mismatches = []

    # Validate each container digest
    for container, expected_digest in container_digests.items():
        try:
            # Parse container image
            registry_type, registry_url, repository, tag = parse_container_image(
                container
            )

            # Create authenticator
            authenticator = create_authenticator(
                registry_type, registry_url, repository
            )

            # Authenticate (but don't fail if it returns False - may work for public containers)
            # The get_container_digest call will retry without authentication if needed
            try:
                authenticator.authenticate(repository=repository)
                # Don't fail here - authentication may fail but public containers can still be accessed
            except Exception as auth_error:
                # Continue anyway - may work for public containers
                # Authentication errors are logged by the authenticator itself
                pass

            # Get actual digest from registry
            actual_digest = get_container_digest(authenticator, repository, tag)

            if not actual_digest:
                errors.append(
                    f"Failed to fetch digest from registry for container '{container}'"
                )
                continue

            # Normalize digests for comparison (lowercase)
            actual_digest = actual_digest.lower()
            expected_digest = expected_digest.lower()

            # Compare digests
            if actual_digest != expected_digest:
                mismatches.append(
                    f"Container '{container}': "
                    f"expected digest '{expected_digest}' "
                    f"but registry has '{actual_digest}'"
                )

        except Exception as e:
            errors.append(f"Error validating container '{container}': {str(e)}")

    all_valid = len(errors) == 0 and len(mismatches) == 0
    all_errors = errors + mismatches

    return all_valid, all_errors, mismatches


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Validate that container digests in mapping.toml match actual registry digests"
    )
    parser.add_argument(
        "--mapping-file",
        type=pathlib.Path,
        default=None,
        help="Path to mapping.toml file (default: auto-detect)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=True,
        help="Exit with non-zero code if validation fails (default: True)",
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

    # Validate container digests
    all_valid, all_errors, mismatches = validate_container_digests(args.mapping_file)

    if not all_valid:
        print("ERROR: Container digest validation failed:", file=sys.stderr)
        print("", file=sys.stderr)
        if mismatches:
            print("Digest mismatches:", file=sys.stderr)
            for error in mismatches:
                print(f"  ❌ {error}", file=sys.stderr)
            print("", file=sys.stderr)
        # Separate errors from mismatches
        non_mismatch_errors = [e for e in all_errors if e not in mismatches]
        if non_mismatch_errors:
            print("Errors:", file=sys.stderr)
            for error in non_mismatch_errors:
                print(f"  ⚠️  {error}", file=sys.stderr)
            print("", file=sys.stderr)
        print(
            "To fix digest mismatches, run: "
            "python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py",
            file=sys.stderr,
        )
        if args.fail_on_error:
            sys.exit(1)
        else:
            sys.exit(0)

    # Count containers validated
    container_digests = extract_container_digests_from_mapping(args.mapping_file)
    print(
        f"✓ Validated {len(container_digests)} container digests - all match registry digests"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
