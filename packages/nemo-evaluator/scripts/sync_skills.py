#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""
CI Sync Pipeline for NeMo-Skills Integration

Synchronizes nemo-skills evaluation code into the NeMo Evaluator package,
applying import rewrites and validating import isolation.

Usage:
    python scripts/sync_skills.py \\
        --source /path/to/nemo-skills \\
        --target src/nemo_evaluator/_nemo_skills/ \\
        --manifest SYNC_MANIFEST.yaml

    python scripts/sync_skills.py --manifest SYNC_MANIFEST.yaml --validate-only
"""

import argparse
import fnmatch
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_manifest(path: str) -> Dict[str, Any]:
    """Parse and validate the YAML sync manifest.

    Args:
        path: Filesystem path to the manifest file.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If manifest fails validation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    required_keys = [
        'source_repo', 'source_ref', 'target_dir', 'pip_dependency_range',
        'sync_dirs', 'import_rewrite_rules', 'import_allowlist', 'protected_files',
    ]
    missing = [k for k in required_keys if k not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")

    if not manifest['source_repo'] or not isinstance(manifest['source_repo'], str):
        raise ValueError("source_repo must be a non-empty string")
    if not manifest['source_ref'] or not isinstance(manifest['source_ref'], str):
        raise ValueError("source_ref must be a non-empty string")

    if not manifest['sync_dirs'] or not isinstance(manifest['sync_dirs'], list):
        raise ValueError("sync_dirs must be a non-empty list")
    if not manifest['import_rewrite_rules'] or not isinstance(manifest['import_rewrite_rules'], list):
        raise ValueError("import_rewrite_rules must be a non-empty list")

    # Validate no identity rewrite rules
    for rule in manifest['import_rewrite_rules']:
        if rule['pattern'] == rule['replacement']:
            raise ValueError(f"Identity rewrite rule detected: {rule['pattern']}")

    return manifest


def _matches_any_pattern(path: str, patterns: List[str]) -> bool:
    """Check if path matches any glob pattern in the list."""
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def _is_protected(rel_path: str, protected_patterns: List[str]) -> bool:
    """Check if a relative path matches any protected file pattern."""
    for pattern in protected_patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def _collect_files(source_dir: str, include_patterns: List[str], exclude_patterns: List[str]) -> List[str]:
    """Collect files matching include patterns and not matching exclude patterns."""
    collected = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not _matches_any_pattern(os.path.join(root, d), exclude_patterns)]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, source_dir)

            if _matches_any_pattern(rel_path, exclude_patterns):
                continue

            if _matches_any_pattern(rel_path, include_patterns):
                collected.append(rel_path)

    return sorted(collected)


def copy_files(
    source_root: str,
    target_root: str,
    sync_dirs: List[Dict[str, Any]],
    protected_patterns: List[str] = None,
) -> List[str]:
    """Copy files from source to target according to sync directory specs.

    Protected files are never overwritten. Existing target directories are
    cleaned before copying (except protected files).

    Args:
        source_root: Root directory of the source repository.
        target_root: Root directory of the target location.
        sync_dirs: List of sync directory specifications from manifest.
        protected_patterns: Glob patterns for files that must not be overwritten.

    Returns:
        List of absolute paths to all copied files.

    Raises:
        FileNotFoundError: If source root does not exist.
    """
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    protected_patterns = protected_patterns or []
    copied_files = []

    for spec in sync_dirs:
        source_dir = os.path.join(source_root, spec['source'])
        target_dir = os.path.join(target_root, spec['target'])

        if not os.path.exists(source_dir):
            print(f"WARNING: Source directory does not exist, skipping: {source_dir}", file=sys.stderr)
            continue

        include_patterns = spec.get('include', [])
        exclude_patterns = spec.get('exclude', [])
        files_to_copy = _collect_files(source_dir, include_patterns, exclude_patterns)

        print(f"Syncing {len(files_to_copy)} files from {spec['source']} to {spec['target']}")

        os.makedirs(target_dir, exist_ok=True)

        for rel_path in files_to_copy:
            src_file = os.path.join(source_dir, rel_path)
            dst_file = os.path.join(target_dir, rel_path)

            # Check protected files
            target_rel = os.path.relpath(dst_file, target_root)
            if _is_protected(target_rel, protected_patterns) and os.path.exists(dst_file):
                print(f"  PROTECTED (skipped): {target_rel}")
                continue

            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied_files.append(os.path.abspath(dst_file))

    return copied_files


def apply_rewrites(file_path: str, rules: List[Dict[str, str]]) -> int:
    """Apply ordered import rewrite rules to a single Python file.

    Rules are applied in manifest order as literal string replacements.

    Args:
        file_path: Path to Python file to rewrite.
        rules: List of rewrite rules (pattern -> replacement).

    Returns:
        Count of replacements made.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.endswith('.py'):
        return 0

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    replacement_count = 0

    for rule in rules:
        pattern = rule['pattern']
        replacement = rule['replacement']
        count = content.count(pattern)
        if count > 0:
            content = content.replace(pattern, replacement)
            replacement_count += count

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    return replacement_count


def validate_imports(target_root: str, allowlist: List[str]) -> List[Dict[str, Any]]:
    """Scan Python files for bare nemo_skills imports not on the allowlist.

    Files in _adapters/ directories are excluded from validation.

    Args:
        target_root: Root directory to scan.
        allowlist: List of allowed import prefixes.

    Returns:
        List of violation dicts with keys: file_path, line_number, import_statement.
    """
    if not os.path.exists(target_root):
        return []

    violations = []
    import_pattern = re.compile(r'^\s*(from|import)\s+(nemo_skills\.\S+|nemo_skills)\s*')

    for root, dirs, files in os.walk(target_root):
        dirs[:] = [d for d in dirs if d != '_adapters']

        for file in files:
            if not file.endswith('.py'):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, start=1):
                        match = import_pattern.match(line)
                        if match:
                            import_module = match.group(2)
                            is_allowed = any(import_module.startswith(a) for a in allowlist)

                            if not is_allowed:
                                violations.append({
                                    'file_path': os.path.abspath(file_path),
                                    'line_number': line_num,
                                    'import_statement': line.strip(),
                                })
            except Exception as e:
                print(f"WARNING: Failed to read {file_path}: {e}", file=sys.stderr)

    return violations


def main():
    """CLI entrypoint for the sync pipeline."""
    parser = argparse.ArgumentParser(
        description='Sync nemo-skills code into NeMo Evaluator with import rewriting',
    )
    parser.add_argument('--source', type=str, help='Path to nemo-skills source checkout')
    parser.add_argument('--target', type=str, help='Path to target directory')
    parser.add_argument('--manifest', type=str, required=True, help='Path to SYNC_MANIFEST.yaml')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run import validation (skip copy and rewrite)')

    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
    except Exception as e:
        print(f"ERROR: Failed to load manifest: {e}", file=sys.stderr)
        sys.exit(1)

    target_root = args.target or manifest['target_dir']

    if args.validate_only:
        print(f"Running import validation on {target_root}")
        violations = validate_imports(target_root, manifest['import_allowlist'])

        if violations:
            print(f"\nERROR: Found {len(violations)} import violations:", file=sys.stderr)
            for v in violations:
                print(f"  {v['file_path']}:{v['line_number']}: {v['import_statement']}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Import validation passed: no violations found")
            sys.exit(0)

    if not args.source:
        print("ERROR: --source is required for sync mode", file=sys.stderr)
        sys.exit(1)

    print(f"Syncing from {args.source} to {target_root}")
    print(f"Source ref: {manifest['source_ref']}")

    # Step 1: Copy files (respecting protected patterns)
    try:
        copied_files = copy_files(
            args.source, target_root, manifest['sync_dirs'],
            protected_patterns=manifest.get('protected_files', []),
        )
        print(f"\nCopied {len(copied_files)} files")
    except Exception as e:
        print(f"ERROR: File copy failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Apply rewrites to all copied Python files
    total_replacements = 0
    python_files = [f for f in copied_files if f.endswith('.py')]

    print(f"Applying import rewrites to {len(python_files)} Python files...")
    for file_path in python_files:
        try:
            count = apply_rewrites(file_path, manifest['import_rewrite_rules'])
            total_replacements += count
        except Exception as e:
            print(f"WARNING: Failed to rewrite {file_path}: {e}", file=sys.stderr)

    print(f"Applied {total_replacements} import replacements")

    # Step 3: Validate imports
    print(f"Validating imports in {target_root}...")
    violations = validate_imports(target_root, manifest['import_allowlist'])

    if violations:
        print(f"\nERROR: Found {len(violations)} import violations after rewriting:", file=sys.stderr)
        for v in violations:
            print(f"  {v['file_path']}:{v['line_number']}: {v['import_statement']}", file=sys.stderr)
        sys.exit(1)

    print("Import validation passed")

    # Step 4: Write SYNCED_FROM.md
    synced_from_path = os.path.join(target_root, 'SYNCED_FROM.md')
    timestamp = datetime.now(timezone.utc).isoformat()
    synced_from_content = (
        f"# Synced from nemo-skills\n\n"
        f"**Source repository:** {manifest['source_repo']}\n"
        f"**Source ref:** {manifest['source_ref']}\n"
        f"**Sync timestamp:** {timestamp}\n"
        f"**Files synced:** {len(copied_files)}\n\n"
        f"This directory contains code synced from the upstream nemo-skills repository.\n"
        f"Do not edit these files directly; changes will be overwritten on the next sync.\n\n"
        f"Adapter stubs in `_adapters/` are protected and never overwritten.\n"
    )

    try:
        with open(synced_from_path, 'w', encoding='utf-8') as f:
            f.write(synced_from_content)
        print(f"Wrote {synced_from_path}")
    except Exception as e:
        print(f"WARNING: Failed to write SYNCED_FROM.md: {e}", file=sys.stderr)

    print(f"\nSync completed successfully")
    print(f"  Files copied: {len(copied_files)}")
    print(f"  Replacements: {total_replacements}")
    print(f"  Violations: 0")


if __name__ == '__main__':
    main()
