#!/usr/bin/env python3
"""
Master script to update all release-related files for a new version.

This script orchestrates the execution of all individual update scripts:
1. update_mapping_toml.py - Updates container tags in mapping.toml
2. update_readme_version.py - Updates version numbers in markdown files
3. update_tasks_table.py - Updates tasks table from Excel file

Usage:
    python update_release.py --old-version 25.09.1 --new-version 25.10 --excel release_summary.xlsx --dry-run
    python update_release.py --old-version 25.09.1 --new-version 25.10 --excel release_summary.xlsx
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_script(script_name: str, args: List[str], dry_run: bool = False) -> bool:
    """
    Run a Python script with the given arguments.
    
    Args:
        script_name: Name of the script to run
        args: List of arguments to pass to the script
        dry_run: If True, add --dry-run flag
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, script_name] + args
    if dry_run:
        cmd.append('--dry-run')
    
    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print('=' * 80)
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        if result.returncode != 0:
            print(f"❌ Error: {script_name} failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Master script to update all release-related files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview all changes for a new release
  python update_release.py --old-version 25.09.1 --new-version 25.10 --excel release_summary.xlsx --dry-run
  
  # Apply all changes for a new release
  python update_release.py --old-version 25.09.1 --new-version 25.10 --excel release_summary.xlsx
  
  # Skip tasks table update (if no Excel file available)
  python update_release.py --old-version 25.09.1 --new-version 25.10 --skip-tasks-table
  
  # Update only mapping.toml
  python update_release.py --new-version 25.10 --only-mapping
        """
    )
    
    parser.add_argument(
        '--old-version',
        help='Old version string to replace (e.g., 25.09.1). Required unless --only-mapping is used.'
    )
    
    parser.add_argument(
        '--new-version',
        required=True,
        help='New version string to use (e.g., 25.10 or 25.10.1)'
    )
    
    parser.add_argument(
        '--excel',
        type=Path,
        help='Path to the release summary Excel file. Required unless --skip-tasks-table is used.'
    )
    
    parser.add_argument(
        '--sheet-name',
        default='Tasks',
        help='Name of the sheet containing the tasks table (default: Tasks)'
    )
    
    parser.add_argument(
        '--skip-tasks-table',
        action='store_true',
        help='Skip updating the tasks table'
    )
    
    parser.add_argument(
        '--only-mapping',
        action='store_true',
        help='Only update mapping.toml (skip README and tasks table)'
    )
    
    parser.add_argument(
        '--only-readme',
        action='store_true',
        help='Only update README and markdown files (skip mapping and tasks table)'
    )
    
    parser.add_argument(
        '--only-tasks',
        action='store_true',
        help='Only update tasks table (skip mapping and README)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.only_mapping and not args.old_version:
        print("Error: --old-version is required unless --only-mapping is used")
        sys.exit(1)
    
    if not args.skip_tasks_table and not args.excel and not args.only_mapping and not args.only_readme:
        print("Error: --excel is required unless --skip-tasks-table or --only-mapping or --only-readme is used")
        sys.exit(1)
    
    # Determine which scripts to run
    run_mapping = not args.only_readme and not args.only_tasks
    run_readme = not args.only_mapping and not args.only_tasks
    run_tasks = not args.skip_tasks_table and not args.only_mapping and not args.only_readme
    
    success = True
    scripts_run = []
    
    print("\n" + "=" * 80)
    print(f"{'DRY RUN - ' if args.dry_run else ''}Release Update")
    print("=" * 80)
    if args.old_version:
        print(f"Old Version: {args.old_version}")
    print(f"New Version: {args.new_version}")
    if args.excel:
        print(f"Excel File: {args.excel}")
    print("=" * 80)
    
    # 1. Update mapping.toml
    if run_mapping:
        mapping_args = ['--new-tag', args.new_version, '--update-all']
        if run_script('update_mapping_toml.py', mapping_args, args.dry_run):
            scripts_run.append('mapping.toml')
        else:
            success = False
    
    # 2. Update README and markdown files
    if run_readme:
        if not args.old_version:
            print("\nSkipping README update: --old-version not provided")
        else:
            readme_args = ['--old-version', args.old_version, '--new-version', args.new_version]
            if run_script('update_readme_version.py', readme_args, args.dry_run):
                scripts_run.append('README files')
            else:
                success = False
    
    # 3. Update tasks table
    if run_tasks:
        if not args.excel:
            print("\nSkipping tasks table update: --excel not provided")
        elif not args.excel.exists():
            print(f"\nError: Excel file not found: {args.excel}")
            success = False
        else:
            tasks_args = ['--excel', str(args.excel), '--sheet-name', args.sheet_name]
            if run_script('update_tasks_table.py', tasks_args, args.dry_run):
                scripts_run.append('tasks table')
            else:
                success = False
    
    # Final summary
    print("\n" + "=" * 80)
    if success:
        if args.dry_run:
            print("✅ DRY RUN completed successfully")
            print("\nAll preview checks passed. Run without --dry-run to apply changes.")
        else:
            print("✅ All updates completed successfully")
            print(f"\nUpdated: {', '.join(scripts_run)}")
            print("\nNext steps:")
            print("  1. Review the changes: git diff")
            print("  2. Test the changes")
            print("  3. Commit: git add -A && git commit -m 'Update version to " + args.new_version + "'")
    else:
        print("❌ Some updates failed")
        print("\nPlease review the errors above and fix any issues.")
        sys.exit(1)
    print("=" * 80)


if __name__ == '__main__':
    main()

