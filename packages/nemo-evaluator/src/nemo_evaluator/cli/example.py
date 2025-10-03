# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

"""
CLI script to set up NVIDIA Eval Factory framework files in a destination folder.
"""

import argparse
import os
from importlib.resources import files
from pathlib import Path

import jinja2

from nemo_evaluator.logging.utils import logger


def add_example_files(package_name: str, destination_folder: str | None = None) -> None:
    """
    Set up NVIDIA framework files in the specified destination folder.

    Args:
        package_name: Package name for the framework
        destination_folder: Path to the destination folder where to create the framework.
            If not provided, current working directory will be used.
    """

    if not package_name.isidentifier():
        raise ValueError(
            f"Package name {package_name} is not a valid Python identifier."
        )

    if destination_folder is None:
        destination_folder = os.getcwd()

    dest_path = Path(destination_folder).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    nvidia_path: Path = dest_path / "core_evals" / package_name
    nvidia_path.mkdir(parents=True, exist_ok=True)

    # Use importlib.resources to access package resources
    resources_dir = files("nemo_evaluator.resources")

    template_files = {
        "framework_tpl.yml": "framework.yml",
        "output_tpl.py": "output.py",
        "init_tpl.py": "__init__.py",
    }

    for template_name, target_name in template_files.items():
        template_path = resources_dir / template_name
        target_path = nvidia_path / target_name

        if target_path.exists():
            with open(target_path, "r") as f:
                content = f.read().strip()
            if content != "":
                logger.warning(
                    f"Target exists and is not empty, not copying: {target_path}"
                )
                continue
        with template_path.open() as in_f:
            template_rendered = (
                jinja2.Environment()
                .from_string(in_f.read())
                .render(framework_name=package_name)
            )

            with target_path.open("w") as out_f:
                out_f.write(template_rendered)
            logger.info(f"Created: {target_path}")

    logger.info(
        "The Eval Factory compatibility package was initialized! Please ensure the following:\n"
        + f"1) Implement the core_evals/{package_name}/framework.yml according to contributing guide\n"
        + "2) Add core_evals module in your build config to be included in your wheel\n"
        + f"3) Implement output parsing logic in core_evals/{package_name}/output.py\n"
        + "4) Add `nemo_evaluator` to your package dependencies\n"
        + "Please also consult the documentation. Good luck and happy packaging!\n"
    )


def main():
    """Main entry point for the CLI script."""
    parser = argparse.ArgumentParser(
        description="Set up NVIDIA framework files in a destination folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nemo_evaluator_example my_package /path/to/destination
        """,
    )

    parser.add_argument(
        "package_name",
        help="Python package-like name for the NVIDIA Eval Factory framework",
    )

    parser.add_argument(
        "destination",
        help="Destination folder where to create the NVIDIA Eval Factory framework files."
        " If not provided, current working directory will be used.",
    )

    args = parser.parse_args()

    add_example_files(args.package_name, args.destination)


if __name__ == "__main__":
    main()
