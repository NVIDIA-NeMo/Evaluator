#!/usr/bin/env python3
"""
Script to update container tags in mapping.toml for new release versions.

This script updates container image tags in the mapping.toml file from an old
release version to a new one. It supports both full updates (all frameworks)
and selective updates (specific frameworks only).

Usage:
    python update_mapping_toml.py --new-tag 25.10.1
    python update_mapping_toml.py --old-tag 25.09.1 --new-tag 25.10.1
    python update_mapping_toml.py --new-tag 25.10.1 --frameworks lm-evaluation-harness mtbench
    python update_mapping_toml.py --new-tag 25.10.1 --dry-run
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


# Pattern to match semantic version tags (e.g., 25.09.1, 1.2.3)
VERSION_TAG_PATTERN = re.compile(r'\d+\.\d+\.\d+')

# Pattern to match container lines in TOML
CONTAINER_LINE_PATTERN = re.compile(
    r'^(container\s*=\s*"[^:]+:)([^"]+)("\s*)$'
)


def find_mapping_toml(start_path: Optional[Path] = None) -> Path:
    """Find the mapping.toml file in the project structure."""
    if start_path is None:
        start_path = Path.cwd()
    
    # Expected path relative to project root
    expected_path = start_path / "packages" / "nemo-evaluator-launcher" / "src" / \
                   "nemo_evaluator_launcher" / "resources" / "mapping.toml"
    
    if expected_path.exists():
        return expected_path
    
    # Try to find it by walking up the directory tree
    current = start_path.resolve()
    for _ in range(5):  # Don't go too far up
        candidate = current / "nemo_evaluator_launcher_internal" / "src" / \
                   "nemo_evaluator_launcher_internal" / "resources" / "mapping.toml"
        if candidate.exists():
            return candidate
        current = current.parent
    
    raise FileNotFoundError(
        f"Could not find mapping.toml starting from {start_path}. "
        f"Please run this script from the project root or specify the path."
    )


def parse_toml_structure(content: str) -> Dict[str, Dict]:
    """
    Parse the TOML file to identify frameworks and their container lines.
    
    Returns a dict mapping framework names to their line info.
    """
    frameworks = {}
    current_framework = None
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Check for framework section header (e.g., [lm-evaluation-harness])
        if line.strip().startswith('[') and not '.' in line:
            match = re.match(r'^\[([^\]]+)\]', line.strip())
            if match:
                framework_name = match.group(1)
                current_framework = framework_name
                frameworks[framework_name] = {
                    'header_line': i,
                    'container_line': None,
                    'container_tag': None
                }
        
        # Check for container line
        elif current_framework and line.strip().startswith('container'):
            container_match = CONTAINER_LINE_PATTERN.match(line)
            if container_match:
                tag = container_match.group(2)
                frameworks[current_framework]['container_line'] = i
                frameworks[current_framework]['container_tag'] = tag
    
    return frameworks


def is_version_tag(tag: str) -> bool:
    """Check if a tag is a semantic version tag (e.g., 25.09.1)."""
    return bool(VERSION_TAG_PATTERN.match(tag))


def extract_version_from_tag(tag: str) -> Optional[str]:
    """Extract version number from a container tag."""
    match = VERSION_TAG_PATTERN.search(tag)
    return match.group(0) if match else None


def update_container_tags(
    content: str,
    old_tag: Optional[str],
    new_tag: str,
    frameworks_to_update: Optional[Set[str]] = None,
    update_all: bool = False,
    dry_run: bool = False
) -> tuple[str, List[Dict]]:
    """
    Update container tags in the TOML content.
    
    Args:
        content: The original TOML content
        old_tag: The old version tag to replace (if None, auto-detect or update all)
        new_tag: The new version tag
        frameworks_to_update: Set of framework names to update (if None, update all)
        update_all: If True, replace ALL tags regardless of format with new_tag
        dry_run: If True, don't actually modify, just report what would change
        
    Returns:
        Tuple of (updated_content, list of changes)
    """
    lines = content.split('\n')
    frameworks = parse_toml_structure(content)
    changes = []
    
    # Auto-detect old tag if not provided and not updating all
    if old_tag is None and not update_all:
        version_tags = set()
        for fw_info in frameworks.values():
            tag = fw_info.get('container_tag')
            if tag:
                version = extract_version_from_tag(tag)
                if version:
                    version_tags.add(version)
        
        if len(version_tags) == 1:
            old_tag = version_tags.pop()
            print(f"Auto-detected old version tag: {old_tag}")
        elif len(version_tags) > 1:
            print(f"Warning: Multiple version tags found: {sorted(version_tags)}")
            print("Please specify --old-tag explicitly or use --update-all")
            old_tag = max(version_tags)  # Use the highest version as default
            print(f"Using {old_tag} as the old tag")
    
    # Update container lines
    for framework_name, fw_info in frameworks.items():
        # Skip if we're only updating specific frameworks
        if frameworks_to_update and framework_name not in frameworks_to_update:
            continue
        
        container_line = fw_info.get('container_line')
        container_tag = fw_info.get('container_tag')
        
        if container_line is None or container_tag is None:
            continue
        
        old_full_tag = container_tag
        new_full_tag = None
        
        if update_all:
            # Replace the entire tag with the new tag
            new_full_tag = new_tag
        else:
            # Check if this tag should be updated (only semantic versions)
            current_version = extract_version_from_tag(container_tag)
            if current_version and (old_tag is None or current_version == old_tag):
                new_full_tag = container_tag.replace(current_version, new_tag)
        
        if new_full_tag and new_full_tag != old_full_tag:
            # Update the line
            old_line = lines[container_line]
            new_line = old_line.replace(old_full_tag, new_full_tag)
            
            if not dry_run:
                lines[container_line] = new_line
            
            changes.append({
                'framework': framework_name,
                'line': container_line + 1,  # 1-indexed for display
                'old': old_full_tag,
                'new': new_full_tag
            })
    
    updated_content = '\n'.join(lines)
    return updated_content, changes


def main():
    parser = argparse.ArgumentParser(
        description='Update container tags in mapping.toml for new release versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all semantic version tags to 25.10.1 (auto-detect old version)
  python update_mapping_toml.py --new-tag 25.10.1
  
  # Update ALL container tags (including timestamp-based) to 25.10.1
  python update_mapping_toml.py --new-tag 25.10.1 --update-all
  
  # Update specific old version to new version
  python update_mapping_toml.py --old-tag 25.09.1 --new-tag 25.10.1
  
  # Update only specific frameworks
  python update_mapping_toml.py --new-tag 25.10.1 --frameworks lm-evaluation-harness mtbench
  
  # Dry run to preview changes
  python update_mapping_toml.py --new-tag 25.10.1 --update-all --dry-run
  
  # Specify custom path to mapping.toml
  python update_mapping_toml.py --new-tag 25.10.1 --file path/to/mapping.toml
        """
    )
    
    parser.add_argument(
        '--new-tag',
        required=True,
        help='New version tag to apply (e.g., 25.10.1)'
    )
    
    parser.add_argument(
        '--old-tag',
        help='Old version tag to replace (if not specified, auto-detect)'
    )
    
    parser.add_argument(
        '--frameworks',
        nargs='+',
        help='Specific frameworks to update (if not specified, update all)'
    )
    
    parser.add_argument(
        '--update-all',
        action='store_true',
        help='Update ALL container tags (including timestamp-based ones) to the new tag'
    )
    
    parser.add_argument(
        '--file',
        type=Path,
        help='Path to mapping.toml file (if not specified, auto-detect)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the file'
    )
    
    args = parser.parse_args()
    
    # Validate new tag format (only for non-update-all mode)
    if not args.update_all and not VERSION_TAG_PATTERN.match(args.new_tag):
        print(f"Error: Invalid version tag format: {args.new_tag}")
        print("Expected format: X.Y.Z (e.g., 25.09.1)")
        print("Or use --update-all to replace all tags with any format")
        sys.exit(1)
    
    # Find the mapping.toml file
    try:
        if args.file:
            mapping_file = args.file
            if not mapping_file.exists():
                raise FileNotFoundError(f"File not found: {mapping_file}")
        else:
            mapping_file = find_mapping_toml()
        print(f"Using mapping file: {mapping_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Read the current content
    content = mapping_file.read_text()
    
    # Convert frameworks list to set
    frameworks_to_update = set(args.frameworks) if args.frameworks else None
    
    # Update the tags
    updated_content, changes = update_container_tags(
        content,
        args.old_tag,
        args.new_tag,
        frameworks_to_update,
        update_all=args.update_all,
        dry_run=args.dry_run
    )
    
    # Report changes
    if not changes:
        print("\nNo changes made. No matching version tags found.")
        if frameworks_to_update:
            print(f"Frameworks specified: {', '.join(frameworks_to_update)}")
        sys.exit(0)
    
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Changes to be made:")
    print("=" * 80)
    
    for change in changes:
        print(f"\n[{change['framework']}] (line {change['line']})")
        print(f"  OLD: {change['old']}")
        print(f"  NEW: {change['new']}")
    
    print("\n" + "=" * 80)
    print(f"Total changes: {len(changes)}")
    
    # Write the updated content
    if not args.dry_run:
        mapping_file.write_text(updated_content)
        print(f"\n✅ Successfully updated {mapping_file}")
    else:
        print("\n⚠️  DRY RUN - No changes written to file")
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
