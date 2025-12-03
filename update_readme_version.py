#!/usr/bin/env python3
"""
Script to update version numbers in README.md and other documentation files.

This script updates version numbers from an old release version to a new one
across markdown files in the repository. It handles various contexts where 
version numbers appear, including tables, links, and text.

Usage:
    python update_readme_version.py --old-version 25.09.1 --new-version 25.10
    python update_readme_version.py --old-version 25.09.1 --new-version 25.10.1 --dry-run
    python update_readme_version.py --old-version 25.09.1 --new-version 25.10 --files README.md
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set


# Pattern to match semantic version tags (e.g., 25.09.1, 1.2.3, 25.10)
VERSION_PATTERN = re.compile(r'\d+\.\d+(?:\.\d+)?')


def find_readme_files(start_path: Optional[Path] = None) -> List[Path]:
    """Find all README.md and documentation files in the project structure."""
    if start_path is None:
        start_path = Path.cwd()
    
    readme_files = []
    
    # Search for README.md files
    readme_files.extend(start_path.rglob("README.md"))
    
    # Also search for other markdown files that might contain versions
    # Exclude specific directories that are generated or not relevant
    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
    
    for md_file in start_path.rglob("*.md"):
        # Skip if in excluded directory
        if any(excluded in md_file.parts for excluded in exclude_dirs):
            continue
        # Add if it's not already in the list
        if md_file not in readme_files:
            # Only add if it actually contains a version number
            try:
                content = md_file.read_text()
                if VERSION_PATTERN.search(content):
                    readme_files.append(md_file)
            except Exception:
                # Skip files that can't be read
                pass
    
    return sorted(readme_files)


def normalize_version(version: str) -> str:
    """
    Normalize version string by removing leading/trailing whitespace.
    Allows versions like 25.10 or 25.10.1
    """
    return version.strip()


def escape_version_for_regex(version: str) -> str:
    """Escape version string for use in regex pattern."""
    return re.escape(version)


def update_versions_in_content(
    content: str,
    old_version: str,
    new_version: str
) -> tuple[str, int]:
    """
    Update all occurrences of old_version to new_version in content.
    
    Args:
        content: The original file content
        old_version: The old version string to replace
        new_version: The new version string
        
    Returns:
        Tuple of (updated_content, number_of_replacements)
    """
    # Escape the old version for regex
    old_version_escaped = escape_version_for_regex(old_version)
    
    # Count occurrences before replacement
    pattern = re.compile(old_version_escaped)
    count = len(pattern.findall(content))
    
    # Replace all occurrences
    updated_content = pattern.sub(new_version, content)
    
    return updated_content, count


def update_files(
    files: List[Path],
    old_version: str,
    new_version: str,
    dry_run: bool = False
) -> List[Dict]:
    """
    Update version numbers in a list of files.
    
    Args:
        files: List of file paths to update
        old_version: The old version string to replace
        new_version: The new version string
        dry_run: If True, don't actually modify files, just report what would change
        
    Returns:
        List of dicts containing information about changes made
    """
    changes = []
    
    for file_path in files:
        try:
            content = file_path.read_text()
            updated_content, count = update_versions_in_content(
                content, old_version, new_version
            )
            
            if count > 0:
                if not dry_run:
                    file_path.write_text(updated_content)
                
                changes.append({
                    'file': str(file_path),
                    'replacements': count,
                    'old_version': old_version,
                    'new_version': new_version
                })
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}", file=sys.stderr)
    
    return changes


def main():
    parser = argparse.ArgumentParser(
        description='Update version numbers in README.md and other documentation files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update version from 25.09.1 to 25.10 in all markdown files
  python update_readme_version.py --old-version 25.09.1 --new-version 25.10
  
  # Update version with patch number
  python update_readme_version.py --old-version 25.09.1 --new-version 25.10.1
  
  # Dry run to preview changes
  python update_readme_version.py --old-version 25.09.1 --new-version 25.10 --dry-run
  
  # Update only specific files
  python update_readme_version.py --old-version 25.09.1 --new-version 25.10 --files README.md docs/index.md
        """
    )
    
    parser.add_argument(
        '--old-version',
        required=True,
        help='Old version string to replace (e.g., 25.09.1)'
    )
    
    parser.add_argument(
        '--new-version',
        required=True,
        help='New version string to use (e.g., 25.10 or 25.10.1)'
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        type=Path,
        help='Specific files to update (if not specified, find all markdown files with versions)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    
    parser.add_argument(
        '--search-dir',
        type=Path,
        default=Path.cwd(),
        help='Directory to search for files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Normalize versions
    old_version = normalize_version(args.old_version)
    new_version = normalize_version(args.new_version)
    
    # Validate version format
    if not VERSION_PATTERN.fullmatch(old_version):
        print(f"Error: Invalid old version format: {old_version}")
        print("Expected format: X.Y or X.Y.Z (e.g., 25.09.1, 25.10)")
        sys.exit(1)
    
    if not VERSION_PATTERN.fullmatch(new_version):
        print(f"Error: Invalid new version format: {new_version}")
        print("Expected format: X.Y or X.Y.Z (e.g., 25.09.1, 25.10)")
        sys.exit(1)
    
    # Get files to update
    if args.files:
        files_to_update = [f.resolve() for f in args.files]
        # Validate that files exist
        for file_path in files_to_update:
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
    else:
        print(f"Searching for markdown files in {args.search_dir}...")
        files_to_update = find_readme_files(args.search_dir)
        if not files_to_update:
            print("No markdown files with version numbers found.")
            sys.exit(0)
    
    print(f"\nFiles to scan: {len(files_to_update)}")
    for f in files_to_update:
        print(f"  - {f}")
    
    # Update the files
    changes = update_files(
        files_to_update,
        old_version,
        new_version,
        dry_run=args.dry_run
    )
    
    # Report changes
    if not changes:
        print(f"\nNo occurrences of version '{old_version}' found.")
        sys.exit(0)
    
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Changes made:")
    print("=" * 80)
    
    total_replacements = 0
    for change in changes:
        print(f"\n{change['file']}")
        print(f"  Replacements: {change['replacements']}")
        print(f"  {change['old_version']} → {change['new_version']}")
        total_replacements += change['replacements']
    
    print("\n" + "=" * 80)
    print(f"Total files modified: {len(changes)}")
    print(f"Total replacements: {total_replacements}")
    
    if not args.dry_run:
        print(f"\n✅ Successfully updated version from {old_version} to {new_version}")
    else:
        print("\n⚠️  DRY RUN - No changes written to files")
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
