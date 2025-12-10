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
"""Autogen script runner for Sphinx documentation builds."""

import os
import pathlib
import subprocess
import sys
import traceback
from typing import Optional

from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)


def run_autogen_script(
    docs_dir: pathlib.Path,
    repo_root: Optional[pathlib.Path] = None,
) -> None:
    """Run the autogen script to regenerate task documentation.

    Args:
        docs_dir: Path to the docs directory (where conf.py is located)
        repo_root: Path to repository root. If None, derived from docs_dir.

    Raises:
        RuntimeError: If autogen script fails or expected output files are missing
    """
    if repo_root is None:
        repo_root = docs_dir.parent

    # Path to autogen script
    autogen_script = (
        repo_root
        / "packages"
        / "nemo-evaluator-launcher"
        / "scripts"
        / "autogen_task_yamls.py"
    )

    # Expected output files that should be generated
    expected_harnesses_dir = docs_dir / "task_catalog" / "harnesses"
    expected_benchmarks_table_file = docs_dir / "task_catalog" / "benchmarks-table.md"

    # Only run if script exists
    if not autogen_script.exists():
        logger.info(
            "ℹ Autogen script not found, skipping task documentation regeneration"
        )
        return

    try:
        logger.info("=" * 80)
        logger.info("Running autogen script to regenerate task documentation...")
        logger.info("=" * 80)

        # Build PYTHONPATH with both launcher and evaluator packages
        launcher_src = repo_root / "packages" / "nemo-evaluator-launcher" / "src"
        evaluator_src = repo_root / "packages" / "nemo-evaluator" / "src"

        env = os.environ.copy()
        pythonpath_parts = []
        if launcher_src.exists():
            pythonpath_parts.append(str(launcher_src))
        if evaluator_src.exists():
            pythonpath_parts.append(str(evaluator_src))

        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)

        if pythonpath_parts:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        # Run autogen script - FAIL BUILD if it fails
        # Stream stdout in real-time for better UX, but capture stderr for error reporting
        result = subprocess.run(
            [sys.executable, str(autogen_script)],
            cwd=str(repo_root),
            env=env,
            stdout=None,  # Let stdout stream normally so users see progress
            stderr=subprocess.PIPE,  # Capture stderr for error reporting
            text=True,
            check=False,  # We'll handle the error ourselves for better messaging
        )

        if result.returncode == 0:
            logger.info("✓ Autogen script completed successfully")

            # Verify expected output files exist
            missing_files = []
            if not expected_harnesses_dir.exists():
                missing_files.append(str(expected_harnesses_dir))
            if not expected_benchmarks_table_file.exists():
                missing_files.append(str(expected_benchmarks_table_file))

            if missing_files:
                error_msg = (
                    "ERROR: Autogen script completed but expected output files are missing:\n"
                    + "\n".join(f"  - {f}" for f in missing_files)
                    + "\n\n"
                    + "This indicates the autogen script did not generate the expected documentation files.\n"
                    + "Please check the autogen script output above for errors."
                )
                logger.error("=" * 80)
                logger.error(error_msg)
                logger.error("=" * 80)
                raise RuntimeError(error_msg)
        else:
            # Build detailed error message
            error_msg_parts = [
                "=" * 80,
                f"ERROR: Autogen script failed with exit code {result.returncode}",
                "=" * 80,
                "",
                "The documentation build cannot proceed because task documentation generation failed.",
                "",
                "STDERR:",
                "-" * 80,
            ]
            if result.stderr:
                error_msg_parts.extend(result.stderr.strip().split("\n"))
            else:
                error_msg_parts.append(
                    "(no stderr output - check stdout above for details)"
                )

            error_msg_parts.extend(
                [
                    "",
                    "=" * 80,
                    "To fix this issue:",
                    "1. Ensure all_tasks_irs.yaml is up to date by running:",
                    "   python scripts/container_metadata_controller.py update",
                    "2. Check that all required dependencies are installed",
                    "3. Verify the autogen script can run independently",
                    "=" * 80,
                ]
            )

            error_msg = "\n".join(error_msg_parts)

            # Log error prominently
            logger.error(error_msg)
            raise RuntimeError(
                f"Autogen script failed with exit code {result.returncode}. "
                "See error output above for details."
            )
    except subprocess.CalledProcessError as e:
        # This shouldn't happen with check=False, but handle it just in case
        error_msg = (
            f"ERROR: Autogen script execution failed: {e}\n"
            f"Command: {e.cmd}\n"
            f"Return code: {e.returncode}\n"
        )
        if e.stdout:
            error_msg += f"STDOUT:\n{e.stdout}\n"
        if e.stderr:
            error_msg += f"STDERR:\n{e.stderr}\n"
        logger.error("=" * 80)
        logger.error(error_msg)
        logger.error("=" * 80)
        raise RuntimeError(
            "Autogen script execution failed. See error output above."
        ) from e
    except Exception as e:
        # Log error prominently and fail the build
        error_msg = (
            f"ERROR: Unexpected error while running autogen script: {e}\n"
            f"\nTraceback:\n{traceback.format_exc()}"
        )
        logger.error("=" * 80)
        logger.error(error_msg)
        logger.error("=" * 80)
        raise RuntimeError(
            "Autogen script execution failed. See error output above."
        ) from e
