#!/usr/bin/env python3
"""Pre-commit hook to enforce centralized logging policy.

This script checks that all Python files use the centralized logging configuration
from `src/nv_eval/common/logging_utils` instead of importing structlog or logging directly.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def check_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Check a single file for improper logging imports.

    Args:
        file_path: Path to the file to check.

    Returns:
        List of tuples containing (line_number, line_content, violation_type).
    """
    violations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return violations

    for line_num, line in enumerate(lines, 1):
        line = line.rstrip("\n")

        # Check for direct structlog imports
        if re.search(r"^import\s+structlog", line):
            violations.append((line_num, line, "Direct structlog import"))
        elif re.search(r"^from\s+structlog\s+import", line):
            violations.append((line_num, line, "Direct structlog import"))

        # Check for direct logging imports
        elif re.search(r"^import\s+logging", line):
            violations.append((line_num, line, "Direct logging import"))
        elif re.search(r"^from\s+logging\s+import", line):
            violations.append((line_num, line, "Direct logging import"))

        # Check for direct structlog.get_logger() calls
        elif re.search(r"structlog\.get_logger\(", line):
            violations.append((line_num, line, "Direct structlog.get_logger() call"))

        # Check for direct logging.getLogger() calls
        elif re.search(r"logging\.getLogger\(", line):
            violations.append((line_num, line, "Direct logging.getLogger() call"))

    return violations


def main() -> int:
    """Main function to check all Python files.

    Returns:
        0 if no violations found, 1 if violations exist.
    """
    # Get files from command line arguments (pre-commit passes them)
    files = sys.argv[1:] if len(sys.argv) > 1 else []

    if not files:
        print("No files provided to check")
        return 0

    all_violations = []

    for file_path_str in files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            continue

        # Skip the logging_utils.py file since it's the central logging module
        print(file_path.name)
        if file_path.name in ["logging_utils.py", "check_logging_imports.py"]:
            print("...skipping")
            continue

        if file_path.name.startswith("test_"):
            print("...skipping")
            continue

        violations = check_file(file_path)
        if violations:
            all_violations.extend(
                [
                    (file_path, line_num, line, violation_type)
                    for line_num, line, violation_type in violations
                ]
            )

    if all_violations:
        print("Logging policy violations found:")
        print()
        print("All logging must go through src/nv_eval/common/logging_utils")
        print("Use: from nemo_evaluator_launcher.common.logging_utils import logger")
        print()

        for file_path, line_num, line, violation_type in all_violations:
            print(f"  {file_path}:{line_num}: {violation_type}")
            print(f"    {line}")
            print()

        print("Please fix these violations and try again.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
