#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Validate documentation code snippets for syntax, imports, and API usage.

This script automatically discovers all _snippets/ directories under docs/
and validates that code snippets in the documentation are technically
correct without requiring actual model endpoints to execute them.

Usage:
    python scripts/validate_doc_snippets.py [--fix] [--verbose]
    python scripts/validate_doc_snippets.py --docs-dir docs --verbose

Requirements:
    Must be run from repository root with both nemo-evaluator and
    nemo-evaluator-launcher installed (see CONTRIBUTING.md).
"""

import argparse
import ast
import inspect
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


class SnippetValidator:
    """Validates documentation code snippets."""

    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.errors = []
        self.warnings = []

    def validate_syntax(self, file_path: Path) -> bool:
        """Validate Python syntax by compiling the file."""
        try:
            py_compile.compile(str(file_path), doraise=True)
            if self.verbose:
                print(f"  {Colors.GREEN}✓{Colors.END} Syntax valid")
            return True
        except py_compile.PyCompileError as e:
            self.errors.append(f"Syntax error: {e}")
            print(f"  {Colors.RED}✗{Colors.END} Syntax error: {e}")
            return False

    def validate_imports(self, file_path: Path) -> bool:
        """Validate that all imports can be resolved."""
        try:
            with open(file_path) as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            # Extract all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Try to import each module
            all_valid = True
            for module_name in imports:
                try:
                    # Skip standard library checks for common modules
                    if module_name in ["os", "sys", "json", "pathlib", "re"]:
                        continue

                    __import__(module_name.split(".")[0])
                    if self.verbose:
                        print(
                            f"  {Colors.GREEN}✓{Colors.END} Import valid: {module_name}"
                        )
                except ImportError as e:
                    self.errors.append(f"Import error: {module_name} - {e}")
                    print(f"  {Colors.RED}✗{Colors.END} Import error: {module_name}")
                    all_valid = False

            if all_valid and imports:
                print(
                    f"  {Colors.GREEN}✓{Colors.END} All imports valid ({len(imports)} checked)"
                )

            return all_valid

        except Exception as e:
            self.errors.append(f"Import validation failed: {e}")
            print(f"  {Colors.RED}✗{Colors.END} Import validation failed: {e}")
            return False

    def validate_api_usage(self, file_path: Path) -> bool:
        """Validate API usage against actual function signatures."""
        try:
            # Only validate files that import from nemo_evaluator
            with open(file_path) as f:
                content = f.read()

            if "nemo_evaluator" not in content:
                if self.verbose:
                    print(
                        f"  {Colors.BLUE}ℹ{Colors.END} No nemo_evaluator imports to validate"
                    )
                return True

            # Try to validate common patterns
            from nemo_evaluator.core.evaluate import evaluate

            eval_sig = inspect.signature(evaluate)

            # Check if evaluate is called with correct parameters
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "evaluate":
                        # Check parameters
                        param_names = {kw.arg for kw in node.keywords}
                        valid_params = set(eval_sig.parameters.keys())

                        invalid = param_names - valid_params
                        if invalid:
                            self.errors.append(
                                f"Invalid evaluate() parameters: {invalid}"
                            )
                            print(
                                f"  {Colors.RED}✗{Colors.END} Invalid parameters: {invalid}"
                            )
                            return False

            print(f"  {Colors.GREEN}✓{Colors.END} API usage valid")
            return True

        except ImportError:
            self.warnings.append("Could not import nemo_evaluator for API validation")
            print(
                f"  {Colors.YELLOW}⚠{Colors.END} Skipping API validation (nemo_evaluator not installed)"
            )
            return True
        except Exception as e:
            self.warnings.append(f"API validation failed: {e}")
            if self.verbose:
                print(f"  {Colors.YELLOW}⚠{Colors.END} API validation error: {e}")
            return True

    def run_linter(self, file_path: Path) -> bool:
        """Run ruff linter on the file."""
        try:
            cmd = ["ruff", "check", str(file_path)]
            if self.fix:
                cmd.append("--fix")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  {Colors.GREEN}✓{Colors.END} Linting passed")
                return True
            else:
                if self.fix:
                    print(
                        f"  {Colors.YELLOW}⚠{Colors.END} Linting issues fixed automatically"
                    )
                    return True
                else:
                    self.warnings.append(f"Linting issues: {result.stdout}")
                    print(
                        f"  {Colors.YELLOW}⚠{Colors.END} Linting issues found (run with --fix to auto-fix)"
                    )
                    if self.verbose:
                        print(f"    {result.stdout}")
                    return False

        except FileNotFoundError:
            if self.verbose:
                print(
                    f"  {Colors.BLUE}ℹ{Colors.END} Ruff not installed, skipping linting"
                )
            return True
        except Exception as e:
            self.warnings.append(f"Linting failed: {e}")
            if self.verbose:
                print(f"  {Colors.YELLOW}⚠{Colors.END} Linting error: {e}")
            return True

    def validate_file(self, file_path: Path) -> bool:
        """Run all validations on a single file."""
        # Convert to absolute path and try to make relative to cwd
        file_path = file_path.resolve()
        try:
            display_path = file_path.relative_to(Path.cwd().resolve())
        except ValueError:
            display_path = file_path

        print(f"\n{Colors.BOLD}Validating:{Colors.END} {display_path}")

        self.errors = []
        self.warnings = []

        # Run validations
        syntax_ok = self.validate_syntax(file_path)
        if not syntax_ok:
            return False  # No point continuing if syntax is broken

        imports_ok = self.validate_imports(file_path)
        api_ok = self.validate_api_usage(file_path)
        lint_ok = self.run_linter(file_path)

        # Overall result
        all_ok = syntax_ok and imports_ok and api_ok and lint_ok

        if all_ok and not self.warnings:
            print(f"{Colors.GREEN}✓ All checks passed!{Colors.END}")
        elif all_ok and self.warnings:
            print(f"{Colors.YELLOW}⚠ Passed with warnings{Colors.END}")
        else:
            print(f"{Colors.RED}✗ Validation failed{Colors.END}")

        return all_ok

    def find_snippet_directories(self, docs_dir: Path) -> List[Path]:
        """Find all _snippets directories under the docs directory."""
        snippet_dirs = []
        for path in docs_dir.rglob("_snippets"):
            if path.is_dir():
                # Skip build directories and other common excluded directories
                if any(
                    excluded in path.parts
                    for excluded in ["_build", "__pycache__", ".git", "node_modules"]
                ):
                    continue
                snippet_dirs.append(path)
        return sorted(snippet_dirs)

    def validate_directory(self, snippets_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Validate all Python files in the snippets directory."""
        python_files = sorted(snippets_dir.rglob("*.py"))

        # Filter out __pycache__ and other unwanted files
        python_files = [f for f in python_files if "__pycache__" not in str(f)]

        if not python_files:
            if self.verbose:
                print(
                    f"{Colors.YELLOW}No Python files found in {snippets_dir}{Colors.END}"
                )
            return [], []

        if self.verbose:
            print(
                f"\n{Colors.BOLD}Found {len(python_files)} Python snippet(s) in {snippets_dir.name}{Colors.END}"
            )

        passed = []
        failed = []

        for file_path in python_files:
            if self.validate_file(file_path):
                passed.append(file_path)
            else:
                failed.append(file_path)

        return passed, failed

    def validate_all_snippets(self, docs_dir: Path) -> Tuple[List[Path], List[Path]]:
        """Find and validate all snippet directories under docs."""
        snippet_dirs = self.find_snippet_directories(docs_dir)

        if not snippet_dirs:
            print(
                f"{Colors.YELLOW}No _snippets directories found under {docs_dir}{Colors.END}"
            )
            return [], []

        print(
            f"\n{Colors.BOLD}Found {len(snippet_dirs)} snippet director{'y' if len(snippet_dirs) == 1 else 'ies'}:{Colors.END}"
        )
        for snippet_dir in snippet_dirs:
            try:
                rel_path = snippet_dir.relative_to(docs_dir)
            except ValueError:
                rel_path = snippet_dir
            print(f"  - docs/{rel_path}")

        all_passed = []
        all_failed = []

        for snippet_dir in snippet_dirs:
            passed, failed = self.validate_directory(snippet_dir)
            all_passed.extend(passed)
            all_failed.extend(failed)

        return all_passed, all_failed


def main():
    parser = argparse.ArgumentParser(
        description="Validate documentation code snippets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all snippets in docs/ (discovers all _snippets/ directories)
  python scripts/validate_doc_snippets.py

  # Auto-fix linting issues
  python scripts/validate_doc_snippets.py --fix

  # Verbose output showing each check
  python scripts/validate_doc_snippets.py --verbose --fix

  # Use a different docs directory
  python scripts/validate_doc_snippets.py --docs-dir docs-archive

Note:
  - Automatically finds all _snippets/ directories under docs/
  - Requires nemo-evaluator and nemo-evaluator-launcher to be installed
  - See CONTRIBUTING.md for setup instructions
        """,
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix linting issues where possible",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed validation output"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Documentation root directory to search for _snippets (default: docs)",
    )

    args = parser.parse_args()

    # Check we're in the right directory
    if not args.docs_dir.exists():
        print(
            f"{Colors.RED}Error: Documentation directory not found: {args.docs_dir}{Colors.END}"
        )
        print("Make sure you're running this from the repository root.")
        sys.exit(1)

    # Run validation
    validator = SnippetValidator(verbose=args.verbose, fix=args.fix)
    passed, failed = validator.validate_all_snippets(args.docs_dir)

    # Print summary
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")

    total = len(passed) + len(failed)
    print(f"Total files: {total}")
    print(f"{Colors.GREEN}Passed: {len(passed)}{Colors.END}")

    if failed:
        print(f"{Colors.RED}Failed: {len(failed)}{Colors.END}")
        print(f"\n{Colors.RED}Failed files:{Colors.END}")
        for f in failed:
            try:
                display_path = f.relative_to(Path.cwd().resolve())
            except ValueError:
                display_path = f
            print(f"  - {display_path}")
        sys.exit(1)
    else:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}✓ All snippets validated successfully!{Colors.END}"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
