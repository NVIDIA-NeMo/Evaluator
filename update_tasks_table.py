#!/usr/bin/env python3
"""
Script to update tasks-table.md from release summary Excel file.

This script reads a release summary Excel file and updates the tasks table
in docs/_resources/tasks-table.md with the latest information about containers,
descriptions, and key benchmarks.

Usage:
    python update_tasks_table.py --excel release_summary_20251031_091746.xlsx
    python update_tasks_table.py --excel release_summary_20251031_091746.xlsx --dry-run
    python update_tasks_table.py --excel release_summary_20251031_091746.xlsx --output-file docs/_resources/tasks-table.md
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required to read Excel files.")
    print("Install it with: pip install pandas openpyxl")
    sys.exit(1)


def find_tasks_table_file(start_path: Optional[Path] = None) -> Path:
    """Find the tasks-table.md file in the project structure."""
    if start_path is None:
        start_path = Path.cwd()
    
    # Expected path relative to project root
    expected_path = start_path / "docs" / "_resources" / "tasks-table.md"
    
    if expected_path.exists():
        return expected_path
    
    # Try to find it by walking up the directory tree
    current = start_path.resolve()
    for _ in range(5):  # Don't go too far up
        candidate = current / "docs" / "_resources" / "tasks-table.md"
        if candidate.exists():
            return candidate
        current = current.parent
    
    raise FileNotFoundError(
        f"Could not find tasks-table.md starting from {start_path}. "
        f"Please run this script from the project root or specify the output path."
    )


def read_tasks_from_excel(excel_path: Path, sheet_name: str = "Tasks") -> pd.DataFrame:
    """
    Read the Tasks table from the Excel file.
    
    Args:
        excel_path: Path to the Excel file
        sheet_name: Name of the sheet containing the tasks table
        
    Returns:
        DataFrame containing the tasks information
    """
    try:
        # Try to read the specified sheet
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        # If that fails, try to list available sheets and provide helpful error
        try:
            excel_file = pd.ExcelFile(excel_path)
            sheets = excel_file.sheet_names
            print(f"Error reading sheet '{sheet_name}': {e}")
            print(f"\nAvailable sheets in the Excel file:")
            for sheet in sheets:
                print(f"  - {sheet}")
            sys.exit(1)
        except Exception as e2:
            print(f"Error reading Excel file: {e2}")
            sys.exit(1)


def format_benchmark_list(benchmarks: str) -> str:
    """Format a list of benchmarks as a comma-separated string."""
    if pd.isna(benchmarks) or not benchmarks:
        return ""
    
    # If it's already a string, clean it up
    if isinstance(benchmarks, str):
        # Split by comma, newline, or semicolon, strip whitespace, and rejoin
        items = [item.strip() for item in benchmarks.replace('\n', ',').replace(';', ',').split(',')]
        items = [item for item in items if item]  # Remove empty strings
        return ", ".join(items)
    
    return str(benchmarks)


def parse_existing_info(existing_file: Optional[Path]) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse existing tasks-table.md to extract descriptions and NGC links.
    
    Returns:
        Tuple of (descriptions dict, ngc_links dict) mapping container names to their info
    """
    descriptions = {}
    ngc_links = {}
    
    if not existing_file or not existing_file.exists():
        return descriptions, ngc_links
    
    try:
        content = existing_file.read_text()
        lines = content.split('\n')
        
        current_container = None
        line_after_container = 0
        for i, line in enumerate(lines):
            # Look for container lines (e.g., "* - **agentic_eval**")
            if line.strip().startswith('* - **'):
                # Extract container name
                match = re.search(r'\* - \*\*([^*]+)\*\*', line)
                if match:
                    current_container = match.group(1)
                    line_after_container = 1
            # Look for description line (1st line after container)
            elif current_container and line_after_container == 1:
                line_after_container = 2
                description = line.strip()[2:].strip() if line.strip().startswith('- ') else ''
                if description and description != '{{ docker_compose_latest }}':
                    descriptions[current_container] = description
            # Look for NGC link line (2nd line after container)
            elif current_container and line_after_container == 2:
                line_after_container = 0
                # Extract NGC link
                ngc_match = re.search(r'\[NGC\]\(([^)]+)\)', line)
                if ngc_match:
                    ngc_links[current_container] = ngc_match.group(1)
                current_container = None
        
    except Exception as e:
        print(f"Warning: Could not parse existing info: {e}")
    
    return descriptions, ngc_links


def normalize_container_name(name: str) -> str:
    """Normalize container name for matching (handle variations like underscore vs dash)."""
    return name.lower().replace('_', '-').replace(' ', '-')


def create_tasks_table_content(
    df: pd.DataFrame, 
    version_placeholder: str = "{{ docker_compose_latest }}",
    existing_file: Optional[Path] = None
) -> str:
    """
    Create the content for tasks-table.md from the DataFrame.
    
    Args:
        df: DataFrame containing tasks information
        version_placeholder: The version placeholder to use in the output
        existing_file: Path to existing tasks-table.md to preserve descriptions and links
        
    Returns:
        String containing the formatted MyST table
    """
    # Parse existing descriptions and NGC links
    existing_descriptions, existing_ngc_links = parse_existing_info(existing_file)
    
    lines = [
        "",
        "```{list-table}",
        ":header-rows: 1",
        ":widths: 20 25 15 15 25",
        "",
        "* - Container",
        "  - Description",
        "  - NGC Catalog",
        "  - Latest Tag",
        "  - Key Benchmarks"
    ]
    
    # Expected column names (try variations)
    component_cols = ['Component Name', 'component name', 'Component', 'component']
    container_cols = ['Container', 'container', 'Framework', 'framework']
    description_cols = ['Description', 'description']
    ngc_cols = ['NGC Catalog', 'NGC', 'ngc', 'NGC Link', 'Link']
    benchmarks_cols = ['Tasks (comma-separated)', 'Key Benchmarks', 'Benchmarks', 'benchmarks', 'Tasks', 'tasks']
    public_cols = ['Public/Internal', 'public/internal', 'Visibility', 'visibility']
    
    # Find actual column names
    component_col = next((col for col in component_cols if col in df.columns), None)
    container_col = next((col for col in container_cols if col in df.columns), None)
    description_col = next((col for col in description_cols if col in df.columns), None)
    ngc_col = next((col for col in ngc_cols if col in df.columns), None)
    benchmarks_col = next((col for col in benchmarks_cols if col in df.columns), None)
    public_col = next((col for col in public_cols if col in df.columns), None)
    
    # Prefer component name over full container path
    name_col = component_col if component_col else container_col
    
    if not name_col:
        print("Error: Could not find Component Name or Container column in Excel file")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Process each row
    for _, row in df.iterrows():
        # Check if this is a public container (if column exists)
        if public_col and not pd.isna(row[public_col]):
            visibility = str(row[public_col]).strip().lower()
            if visibility != 'public':
                continue  # Skip non-public containers
        
        container_name = row[name_col]
        
        # Skip if container is empty or NaN
        if pd.isna(container_name) or not str(container_name).strip():
            continue
        
        container_name = str(container_name).strip()
        
        # Get other fields with fallbacks
        description = ""
        if description_col and not pd.isna(row[description_col]):
            description = str(row[description_col]).strip()
        
        # If no description in Excel, try to preserve from existing file
        if not description:
            # Try exact match first
            if container_name in existing_descriptions:
                description = existing_descriptions[container_name]
            else:
                # Try normalized match (handles underscore vs dash differences)
                normalized = normalize_container_name(container_name)
                for existing_name, existing_desc in existing_descriptions.items():
                    if normalize_container_name(existing_name) == normalized:
                        description = existing_desc
                        break
        
        # Handle NGC link - try to preserve from existing file first
        ngc_link = ""
        
        # Try to get NGC link from existing file
        if container_name in existing_ngc_links:
            ngc_link = f"[NGC]({existing_ngc_links[container_name]})"
        else:
            # Try normalized match
            normalized = normalize_container_name(container_name)
            for existing_name, existing_link in existing_ngc_links.items():
                if normalize_container_name(existing_name) == normalized:
                    ngc_link = f"[NGC]({existing_link})"
                    break
        
        # If no existing link found, check Excel or generate one
        if not ngc_link:
            if ngc_col and not pd.isna(row[ngc_col]):
                ngc_value = str(row[ngc_col])
                if ngc_value.startswith('http'):
                    ngc_link = f"[NGC]({ngc_value})"
                else:
                    # Construct the link
                    ngc_link = f"[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/{container_name})"
            else:
                ngc_link = f"[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/{container_name})"
        
        # Get benchmarks
        benchmarks = ""
        if benchmarks_col and not pd.isna(row[benchmarks_col]):
            benchmarks = format_benchmark_list(row[benchmarks_col])
        
        # Add the row to the table
        lines.append(f"* - **{container_name}**")
        lines.append(f"  - {description}")
        lines.append(f"  - {ngc_link}")
        lines.append(f"  - {version_placeholder}")
        lines.append(f"  - {benchmarks}")
    
    lines.append("```")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Update tasks-table.md from release summary Excel file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update tasks table from Excel file
  python update_tasks_table.py --excel release_summary_20251031_091746.xlsx
  
  # Dry run to preview changes
  python update_tasks_table.py --excel release_summary_20251031_091746.xlsx --dry-run
  
  # Specify custom output file
  python update_tasks_table.py --excel release_summary.xlsx --output-file docs/_resources/tasks-table.md
  
  # Specify custom sheet name
  python update_tasks_table.py --excel release_summary.xlsx --sheet-name "Task List"
        """
    )
    
    parser.add_argument(
        '--excel',
        type=Path,
        required=True,
        help='Path to the release summary Excel file'
    )
    
    parser.add_argument(
        '--sheet-name',
        default='Tasks',
        help='Name of the sheet containing the tasks table (default: Tasks)'
    )
    
    parser.add_argument(
        '--output-file',
        type=Path,
        help='Path to output tasks-table.md file (if not specified, auto-detect)'
    )
    
    parser.add_argument(
        '--version-placeholder',
        default='{{ docker_compose_latest }}',
        help='Version placeholder to use in the output (default: {{ docker_compose_latest }})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the file'
    )
    
    args = parser.parse_args()
    
    # Validate Excel file exists
    if not args.excel.exists():
        print(f"Error: Excel file not found: {args.excel}")
        sys.exit(1)
    
    print(f"Reading Excel file: {args.excel}")
    
    # Read the Excel file
    df = read_tasks_from_excel(args.excel, args.sheet_name)
    print(f"Read {len(df)} rows from sheet '{args.sheet_name}'")
    
    # Show column names for debugging
    print(f"Columns found: {', '.join(df.columns)}")
    
    # Determine output file first (needed for preserving descriptions)
    if args.output_file:
        output_file = args.output_file
    else:
        try:
            output_file = find_tasks_table_file()
            print(f"Found tasks table file: {output_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Create the new content (pass existing file to preserve descriptions)
    new_content = create_tasks_table_content(df, args.version_placeholder, output_file)
    
    # Show preview
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - Preview of new content:")
        print("=" * 80)
        print(new_content)
        print("=" * 80)
        print("\n⚠️  DRY RUN - No changes written to file")
        print("Run without --dry-run to apply changes")
    else:
        # Write the new content
        output_file.write_text(new_content)
        print(f"\n✅ Successfully updated {output_file}")
        print(f"   Total containers: {len(df[df[df.columns[0]].notna()])}")


if __name__ == '__main__':
    main()

