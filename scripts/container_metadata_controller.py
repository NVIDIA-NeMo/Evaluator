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
"""Unified script to manage container metadata: verify and update digests, checksums, and files.

This script combines the functionality of:
- load_framework_definitions.py: Load framework definitions and update digests
- update_readme.py: Update README.md table
- validate_container_digests.py: Validate digests match registry
- check_readme_checksum.py: Verify README checksum
- check_all_tasks_irs_checksum.py: Verify all_tasks_irs.yaml checksum

Modes:
  --verify: Check that digests and checksums are correct
  --update: Update digests, load framework definitions, and update README
"""

import argparse
import hashlib
import os
import pathlib
import re
import sys
from typing import Any, Optional

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.container_metadata import (
    HarnessIntermediateRepresentation,
    RegistryAuthenticator,
    TaskIntermediateRepresentation,
    create_authenticator,
    extract_framework_yml,
    load_tasks_from_tasks_file,
    parse_framework_to_irs,
)
from nemo_evaluator_launcher.common.helpers import parse_container_image
from nemo_evaluator_launcher.common.logging_utils import logger

# Maximum number of lines to check after container declaration for digest comment
MAX_DIGEST_COMMENT_OFFSET_LINES = 4


def find_repo_root(start_path: pathlib.Path) -> pathlib.Path:
    """Find the repository root by looking for .git directory."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return start_path.parent.parent


def calculate_mapping_checksum(mapping_file: pathlib.Path) -> str:
    """Calculate SHA256 checksum of mapping.toml file."""
    with open(mapping_file, "rb") as f:
        file_content = f.read()
    checksum = hashlib.sha256(file_content).hexdigest()
    return f"sha256:{checksum}"


# create_authenticator removed - use from helpers instead


def get_container_digest(
    authenticator: RegistryAuthenticator, repository: str, reference: str
) -> Optional[str]:
    """Get the manifest digest for a container image."""
    try:
        _, digest = authenticator.get_manifest_and_digest(repository, reference)
        return digest
    except Exception as e:
        logger.warning(
            "Failed to get container digest",
            repository=repository,
            reference=reference,
            error=str(e),
        )
        return None


def extract_container_digests_from_mapping(
    mapping_file: pathlib.Path,
) -> dict[str, str]:
    """Extract container images and their digest comments from mapping.toml."""
    if not mapping_file.exists():
        return {}

    with open(mapping_file, "r", encoding="utf-8") as f:
        content = f.read()

    container_digests = {}
    lines = content.split("\n")

    for i, line in enumerate(lines):
        match = re.match(r'^\s*container\s*=\s*"([^"]+)"', line)
        if match:
            container = match.group(1)
            section_line_idx = i
            while section_line_idx >= 0 and not lines[
                section_line_idx
            ].strip().startswith("["):
                section_line_idx -= 1
            if section_line_idx >= 0:
                section_line = lines[section_line_idx].strip()
                if "." not in section_line.strip("[]"):
                    for offset in range(1, MAX_DIGEST_COMMENT_OFFSET_LINES + 1):
                        if i + offset >= len(lines):
                            break
                        next_line = lines[i + offset].strip()
                        digest_match = re.match(
                            r"#\s*container-digest:\s*(sha256:[a-f0-9]+)",
                            next_line,
                            re.IGNORECASE,
                        )
                        if digest_match:
                            container_digests[container] = digest_match.group(1).lower()
                            break
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
    """Validate that container digests in mapping.toml match actual registry digests."""
    container_digests = extract_container_digests_from_mapping(mapping_file)

    if not container_digests:
        return (
            False,
            ["No container digests found in mapping.toml"],
            [],
        )

    errors = []
    mismatches = []

    for container, expected_digest in container_digests.items():
        try:
            registry_type, registry_url, repository, tag = parse_container_image(
                container
            )
            authenticator = create_authenticator(
                registry_type, registry_url, repository
            )
            authenticator.authenticate(repository=repository)
            actual_digest = get_container_digest(authenticator, repository, tag)

            if not actual_digest:
                errors.append(
                    f"Failed to fetch digest from registry for container '{container}'"
                )
                continue

            if actual_digest.lower() != expected_digest.lower():
                mismatches.append(
                    f"Container '{container}': expected '{expected_digest}' "
                    f"but registry has '{actual_digest}'"
                )
        except Exception as e:
            errors.append(f"Error validating container '{container}': {str(e)}")

    all_valid = len(errors) == 0 and len(mismatches) == 0
    return all_valid, errors + mismatches, mismatches


def verify_checksums(
    mapping_file: pathlib.Path,
    all_tasks_irs_file: pathlib.Path,
    readme_file: Optional[pathlib.Path],
) -> tuple[bool, list[str]]:
    """Verify checksums in all_tasks_irs.yaml and README.md match mapping.toml."""
    errors = []
    actual_checksum = calculate_mapping_checksum(mapping_file)

    # Check all_tasks_irs.yaml checksum
    if all_tasks_irs_file.exists():
        try:
            with open(all_tasks_irs_file, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            stored_checksum = yaml_data.get("metadata", {}).get("mapping_toml_checksum")
            if stored_checksum and stored_checksum != actual_checksum:
                errors.append(
                    f"all_tasks_irs.yaml checksum mismatch: "
                    f"stored '{stored_checksum}' != actual '{actual_checksum}'"
                )
        except Exception as e:
            errors.append(f"Failed to read all_tasks_irs.yaml: {e}")
    else:
        logger.debug("all_tasks_irs.yaml not found, skipping checksum check")

    # Check README.md checksum (skip if readme_file is None)
    if readme_file is not None and readme_file.exists():
        try:
            content = readme_file.read_text(encoding="utf-8")
            pattern = r"<!--\s*mapping\s+toml\s+checksum:\s*(sha256:[a-f0-9]+)\s*-->"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                readme_checksum = match.group(1)
                if readme_checksum != actual_checksum:
                    errors.append(
                        f"README.md checksum mismatch: "
                        f"stored '{readme_checksum}' != actual '{actual_checksum}'"
                    )
            else:
                logger.debug("Could not find checksum comment in README.md")
        except Exception as e:
            errors.append(f"Failed to read README.md: {e}")
    elif readme_file is None:
        logger.debug("README.md check skipped (--no-readme specified)")
    else:
        logger.debug("README.md not found, skipping checksum check")

    return len(errors) == 0, errors


def verify_mode(
    mapping_file: pathlib.Path,
    all_tasks_irs_file: pathlib.Path,
    readme_file: Optional[pathlib.Path],
) -> int:
    """Verify digests and checksums."""
    all_valid = True

    # Verify container digests
    logger.info("Validating container digests...")
    digests_valid, digest_errors, mismatches = validate_container_digests(mapping_file)
    if not digests_valid:
        all_valid = False
        if mismatches:
            logger.error("Digest mismatches:")
            for error in mismatches:
                logger.error(f"  {error}")
        for error in digest_errors:
            if error not in mismatches:
                logger.error(f"  {error}")

    # Verify checksums
    logger.info("Validating checksums...")
    checksums_valid, checksum_errors = verify_checksums(
        mapping_file, all_tasks_irs_file, readme_file
    )
    if not checksums_valid:
        all_valid = False
        for error in checksum_errors:
            logger.error(f"  {error}")

    if all_valid:
        logger.info("✓ All validations passed")
        return 0
    else:
        logger.error("✗ Validation failed")
        return 1


def _find_section_header(lines: list[str], line_index: int) -> int | None:
    """Find the section header for a given line index."""
    section_line_idx = line_index
    while section_line_idx >= 0 and not lines[section_line_idx].strip().startswith("["):
        section_line_idx -= 1
    return section_line_idx if section_line_idx >= 0 else None


def _is_harness_level_section(section_line: str) -> bool:
    """Check if a section line represents a harness-level section."""
    return "." not in section_line.strip("[]")


def update_digest_comment_in_mapping_toml(
    mapping_file: pathlib.Path, container: str, digest: str
) -> None:
    """Update or add digest comment for container in mapping.toml."""
    with open(mapping_file, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    updated = False

    for i, line in enumerate(lines):
        match = re.match(r'^(\s*)container\s*=\s*"([^"]+)"', line)
        if match and match.group(2) == container:
            indent = match.group(1)
            section_line_idx = _find_section_header(lines, i)
            if section_line_idx is not None:
                section_line = lines[section_line_idx].strip()
                if _is_harness_level_section(section_line):
                    digest_comment_pattern = (
                        r"#\s*container-digest:\s*(sha256:[a-f0-9]+)"
                    )
                    existing_digest = None
                    comment_line_idx = None

                    for offset in range(1, MAX_DIGEST_COMMENT_OFFSET_LINES + 1):
                        if i + offset >= len(lines):
                            break
                        next_line = lines[i + offset]
                        match_comment = re.search(
                            digest_comment_pattern, next_line, re.IGNORECASE
                        )
                        if match_comment:
                            existing_digest = match_comment.group(1)
                            comment_line_idx = i + offset
                            break
                        if next_line.strip().startswith("[") or (
                            next_line.strip()
                            and not next_line.strip().startswith("#")
                            and "=" in next_line
                        ):
                            break

                    digest_comment = f"{indent}# container-digest:{digest}"

                    if existing_digest:
                        if existing_digest.lower() == digest.lower():
                            continue
                        lines[comment_line_idx] = digest_comment
                    else:
                        lines.insert(i + 1, digest_comment)
                    updated = True
                    break

    if updated:
        with open(mapping_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def load_mapping_toml(mapping_file: pathlib.Path) -> tuple[dict[str, str], str]:
    """Load mapping.toml and extract unique containers per harness."""
    checksum = calculate_mapping_checksum(mapping_file)

    with open(mapping_file, "rb") as f:
        mapping_toml = tomllib.load(f)

    harness_containers: dict[str, str] = {}
    for section_name, section_data in mapping_toml.items():
        if "." not in section_name and isinstance(section_data, dict):
            container = section_data.get("container")
            if container:
                harness_containers[section_name] = container

    return harness_containers, checksum


def serialize_tasks_irs(tasks: list[Any], metadata: dict[str, Any]) -> str:
    """Serialize task IRs and metadata into single YAML format."""
    output_dict = {
        "metadata": metadata,
        "tasks": [task.to_dict() for task in tasks],
    }
    return yaml.dump(output_dict, default_flow_style=False, sort_keys=False)


def extract_container_name_and_version(container: str) -> tuple[str, str]:
    """Extract container name and version from container image identifier."""
    if not container:
        return "", ""

    version = ""
    if ":" in container:
        container, version = container.rsplit(":", 1)

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
    """Generate README.md table markdown inside HTML comments."""
    lines = []
    lines.append("<!-- BEGIN AUTOGENERATION -->")
    lines.append(f"<!-- mapping toml checksum: {checksum} -->")

    table_lines = []
    table_lines.append(
        "| Container | Description | NGC Catalog | Latest Tag | Supported benchmarks |"
    )
    table_lines.append(
        "|-----------|-------------|-------------|------------| ------------|"
    )

    sorted_harnesses = sorted(harnesses.items(), key=lambda x: x[0].lower())

    for harness_name, (harness_ir, tasks) in sorted_harnesses:
        container = harness_ir.container
        if not container and tasks:
            container = tasks[0].container

        container_name, version = extract_container_name_and_version(container)

        if container_name:
            ngc_url = f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/{container_name}"
            if version:
                ngc_url += f"?version={version}"
            ngc_link = f"[Link]({ngc_url})"
        else:
            ngc_link = "N/A"

        description = harness_ir.description or ""
        if not description and tasks:
            description = tasks[0].description or ""

        task_names = [task.name for task in tasks]
        tasks_display = ", ".join(task_names)
        latest_tag = version if version else "25.11"
        description_display = description.replace("|", "\\|").replace("\n", " ")
        container_display = f"**{harness_name}**"

        table_lines.append(
            f"| {container_display} | {description_display} | {ngc_link} | `{latest_tag}` | {tasks_display} |"
        )

    lines.append("<!--")
    for table_line in table_lines:
        lines.append(table_line)
    lines.append("-->")
    lines.append("<!-- END AUTOGENERATION -->")

    return "\n".join(lines)


def update_readme_table(readme_path: pathlib.Path, table_content: str) -> None:
    """Update the README.md file with new table content."""
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

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

    before = content[: begin_match.start()]
    after = content[end_match.end() :]
    new_content = before + table_content + after

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def update_mode(
    mapping_file: pathlib.Path,
    all_tasks_irs_file: pathlib.Path,
    readme_file: Optional[pathlib.Path],
    max_layer_size: int,
    use_cache: bool,
) -> int:
    """Update digests, load framework definitions, and update README."""
    # Load containers from mapping.toml
    harness_containers, mapping_checksum = load_mapping_toml(mapping_file)

    if not harness_containers:
        logger.warning("No containers found in mapping.toml")
        return 1

    # Extract framework.yml from each container and convert to IRs
    all_task_irs: list[TaskIntermediateRepresentation] = []
    successful = 0
    failed_containers: list[tuple[str, str, Optional[str]]] = []
    name_warnings: list[str] = []  # Collect harness name validation warnings

    for harness_name, container in harness_containers.items():
        logger.info(
            "Processing container",
            harness=harness_name,
            container=container,
            progress=f"{successful + len(failed_containers) + 1}/{len(harness_containers)}",
        )

        framework_content, container_digest = extract_framework_yml(
            container=container,
            max_layer_size=max_layer_size,
            use_cache=use_cache,
        )

        if container_digest:
            update_digest_comment_in_mapping_toml(
                mapping_file, container, container_digest
            )

        if not framework_content:
            failed_containers.append((harness_name, container, container_digest))
            logger.warning(
                "Skipping container (framework.yml not found)",
                harness=harness_name,
                container=container,
            )
            continue

        try:
            # Extract original framework name from framework.yml for validation
            # (validation and normalization happen in this controller script, not in parse_framework_to_irs)
            import yaml

            framework_data = yaml.safe_load(framework_content)
            framework_info = (
                framework_data.get("framework", {}) if framework_data else {}
            )
            original_framework_name = framework_info.get("name", "")

            # Normalize original name for comparison (lowercase, same validation as parse_framework_to_irs)
            if original_framework_name:
                if not isinstance(original_framework_name, str):
                    name_warnings.append(
                        f"Container '{container}' (harness '{harness_name}'): "
                        f"framework.name must be a string, got {type(original_framework_name).__name__}"
                    )
                    normalized_framework_name = ""
                elif not original_framework_name.replace("-", "").isalnum():
                    invalid_chars = set(
                        c
                        for c in original_framework_name
                        if not (c.isalnum() or c == "-")
                    )
                    name_warnings.append(
                        f"Container '{container}' (harness '{harness_name}'): "
                        f"framework.name '{original_framework_name}' contains invalid characters: {sorted(invalid_chars)}. "
                        f"Only letters, digits, and hyphens are allowed."
                    )
                    normalized_framework_name = original_framework_name.lower()
                else:
                    normalized_framework_name = original_framework_name.lower()
            else:
                normalized_framework_name = ""

            harness_ir, task_irs = parse_framework_to_irs(
                framework_content=framework_content,
                container_id=container,
                container_digest=container_digest,
            )

            # Validate that harness name from TOML matches normalized harness name from framework.yml
            # Normalize both for comparison (lowercase)
            normalized_harness_name = harness_name.lower()
            normalized_harness_ir_name = harness_ir.name.lower()
            if normalized_framework_name and normalized_framework_name != normalized_harness_name:
                name_warnings.append(
                    f"Container '{container}' (harness '{harness_name}'): "
                    f"Harness name mismatch: TOML has '{harness_name}' (normalized: '{normalized_harness_name}'), "
                    f"but framework.yml has '{original_framework_name}' (normalized: '{normalized_framework_name}'). "
                    f"These should match after normalization (lowercase)."
                )
            # Also verify that parse_framework_to_irs returned the expected name
            if normalized_framework_name and normalized_harness_ir_name != normalized_framework_name:
                name_warnings.append(
                    f"Container '{container}' (harness '{harness_name}'): "
                    f"Harness IR name mismatch: framework.yml has '{original_framework_name}' (normalized: '{normalized_framework_name}'), "
                    f"but parse_framework_to_irs returned '{harness_ir.name}' (normalized: '{normalized_harness_ir_name}'). "
                    f"These should match."
                )

            all_task_irs.extend(task_irs)
            successful += 1
        except Exception as e:
            failed_containers.append((harness_name, container, container_digest))
            logger.error(
                "Failed to parse framework.yml to IRs",
                harness=harness_name,
                container=container,
                error=str(e),
            )

    # Recalculate checksum after digest updates
    mapping_checksum = calculate_mapping_checksum(mapping_file)

    # Write all_tasks_irs.yaml
    if all_task_irs:
        metadata = {
            "mapping_toml_checksum": mapping_checksum,
            "num_tasks": len(all_task_irs),
        }
        yaml_content = serialize_tasks_irs(all_task_irs, metadata)
        all_tasks_irs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(all_tasks_irs_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(
            "Wrote all_tasks_irs.yaml",
            num_tasks=len(all_task_irs),
            checksum=mapping_checksum,
        )

    # Update README.md
    tasks, _ = load_tasks_from_tasks_file(all_tasks_irs_file)
    harnesses: dict[
        str,
        tuple[HarnessIntermediateRepresentation, list[TaskIntermediateRepresentation]],
    ] = {}
    for task in tasks:
        if not task.harness:
            continue

        if task.harness not in harnesses:
            container = str(task.container) if task.container else ""
            container_digest = (
                str(task.container_digest) if task.container_digest else None
            )
            harness_ir = HarnessIntermediateRepresentation(
                name=task.harness,
                description="",
                full_name=None,
                url=None,
                container=container,
                container_digest=container_digest,
            )
            harnesses[task.harness] = (harness_ir, [])

        harnesses[task.harness][1].append(task)

    # Update README.md (skip if readme_file is None)
    if readme_file is not None:
        table_content = generate_readme_table(harnesses, mapping_checksum)
        update_readme_table(readme_file, table_content)
        logger.info("Updated README.md", checksum=mapping_checksum)
    else:
        logger.info("Skipping README.md update (--no-readme specified)")

    # Display harness name validation warnings at the end
    if name_warnings:
        logger.warning("Harness name validation warnings:")
        for warning in name_warnings:
            logger.warning(f"  {warning}")

    logger.info(
        "Update complete",
        total=len(harness_containers),
        successful=successful,
        failed=len(failed_containers),
    )

    return 0 if successful > 0 else 1


def get_default_paths(
    repo_root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Get default paths for mapping.toml, all_tasks_irs.yaml, and README.md.

    Args:
        repo_root: Repository root directory

    Returns:
        Tuple of (mapping_toml_path, all_tasks_irs_path, readme_path)
    """
    mapping_toml = (
        repo_root
        / "packages"
        / "nemo-evaluator-launcher"
        / "src"
        / "nemo_evaluator_launcher"
        / "resources"
        / "mapping.toml"
    )
    all_tasks_irs = (
        repo_root
        / "packages"
        / "nemo-evaluator-launcher"
        / "src"
        / "nemo_evaluator_launcher"
        / "resources"
        / "all_tasks_irs.yaml"
    )
    readme_file = repo_root / "README.md"
    return mapping_toml, all_tasks_irs, readme_file


def main():
    """Main entry point for the script."""
    # Find repository root first to set defaults
    repo_root = find_repo_root(pathlib.Path(__file__))
    default_mapping_toml, default_all_tasks_irs, default_readme = get_default_paths(
        repo_root
    )

    parser = argparse.ArgumentParser(
        description="Unified container metadata controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default file locations:
  --mapping-toml: {default_mapping_toml}
  --all-tasks-irs: {default_all_tasks_irs}
  --readme-file: {default_readme}

Examples:
  # Verify digests and checksums (using defaults)
  {sys.argv[0]} verify

  # Update metadata (using defaults)
  {sys.argv[0]} update

  # Verify with custom mapping file
  {sys.argv[0]} verify --mapping-toml custom.toml
        """,
    )
    parser.add_argument(
        "--mapping-toml",
        type=pathlib.Path,
        default=default_mapping_toml,
        help=f"Path to mapping.toml file (default: {default_mapping_toml})",
    )
    parser.add_argument(
        "--all-tasks-irs",
        type=pathlib.Path,
        default=default_all_tasks_irs,
        help=f"Path to all_tasks_irs.yaml file (default: {default_all_tasks_irs})",
    )
    readme_group = parser.add_mutually_exclusive_group()
    readme_group.add_argument(
        "--readme-file",
        type=pathlib.Path,
        default=default_readme,
        help=f"Path to README.md file (default: {default_readme})",
    )
    readme_group.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip README.md checks/updates",
    )
    parser.add_argument(
        "--max-layer-size",
        type=int,
        default=100 * 1024,
        help="Maximum layer size in bytes for framework.yml extraction (default: 100KB)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of framework.yml files",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("verify", help="Verify digests and checksums")
    update_parser = subparsers.add_parser("update", help="Update digests and files")
    update_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be updated without writing files",
    )

    args = parser.parse_args()

    # Handle --no-readme: set readme_file to None if --no-readme is specified
    if args.no_readme:
        args.readme_file = None

    # Set log level
    if not os.getenv("NEMO_EVALUATOR_LOG_LEVEL") and not os.getenv("LOG_LEVEL"):
        os.environ["NEMO_EVALUATOR_LOG_LEVEL"] = "INFO"
        import importlib

        from nemo_evaluator_launcher.common import logging_utils

        importlib.reload(logging_utils)
        logging_utils._configure_structlog()

    # Validate files exist
    if not args.mapping_toml.exists():
        logger.error("mapping.toml not found", path=str(args.mapping_toml))
        sys.exit(1)

    if args.mode == "verify":
        sys.exit(verify_mode(args.mapping_toml, args.all_tasks_irs, args.readme_file))
    elif args.mode == "update":
        if args.dry_run:
            logger.info("DRY RUN mode - no files will be modified")
        sys.exit(
            update_mode(
                args.mapping_toml,
                args.all_tasks_irs,
                args.readme_file,
                args.max_layer_size,
                not args.no_cache,
            )
        )


if __name__ == "__main__":
    main()
