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

"""Package installation pre-evaluation hook."""

import subprocess
import sys
from typing import List

from pydantic import BaseModel, Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import AdapterGlobalContext, PreEvalHook
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


@register_for_adapter(
    name="package_install_prehook",
    description="Installs Python packages using pip before evaluation starts",
)
class PackageInstallPreHook(PreEvalHook):
    """Pre-evaluation hook that installs required Python packages using pip."""

    class Params(BaseModel):
        """Configuration parameters for package installation."""

        packages: List[str] = Field(
            default_factory=list,
            description="List of package specifications to install (e.g., ['numpy>=1.20', 'torch==2.0.0'])",
        )

        requirements_file: str | None = Field(
            default=None,
            description="Path to requirements.txt file to install packages from",
        )

        upgrade: bool = Field(
            default=False,
            description="Whether to upgrade packages if already installed",
        )

        extra_args: List[str] = Field(
            default_factory=list,
            description="Additional arguments to pass to pip",
        )

        check_installed: bool = Field(
            default=True,
            description="Check if packages are already installed before attempting installation",
        )

        fail_on_error: bool = Field(
            default=True,
            description="Whether to fail evaluation if package installation fails",
        )

    def __init__(self, params: Params):
        """
        Initialize the package installation pre-hook.

        Args:
            params: Configuration parameters
        """
        self.params = params

        if not self.params.packages and not self.params.requirements_file:
            raise ValueError(
                "Either 'packages' list or 'requirements_file' must be provided"
            )

        logger.info(
            "Initialized PackageInstallPreHook",
            num_packages=len(params.packages),
            requirements_file=params.requirements_file,
        )

    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Install required packages before evaluation."""
        logger.info("Starting package installation")

        try:
            # Install from requirements file if provided
            if self.params.requirements_file:
                self._install_from_requirements()

            # Install individual packages if provided
            if self.params.packages:
                self._install_packages()

            logger.info("Package installation completed successfully")

        except Exception as e:
            logger.error("Package installation failed", error=str(e))
            if self.params.fail_on_error:
                raise
            else:
                logger.warning(
                    "Continuing evaluation despite package installation failure"
                )

    def _is_package_installed(self, package_spec: str) -> bool:
        """Check if a package is already installed."""
        # Extract package name from specification (e.g., 'numpy>=1.20' -> 'numpy')
        package_name = package_spec.split(">=")[0].split("==")[0].split("<")[0].strip()

        try:
            __import__(package_name)
            logger.debug("Package already installed", package=package_name)
            return True
        except ImportError:
            return False

    def _install_packages(self) -> None:
        """Install individual packages using pip."""
        packages_to_install = []

        # Filter out already installed packages if check is enabled
        if self.params.check_installed and not self.params.upgrade:
            for package in self.params.packages:
                if not self._is_package_installed(package):
                    packages_to_install.append(package)
                else:
                    logger.info("Package already installed, skipping", package=package)
        else:
            packages_to_install = self.params.packages

        if not packages_to_install:
            logger.info("All packages already installed, nothing to do")
            return

        logger.info("Installing packages with pip", packages=packages_to_install)

        # Build pip install command
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if self.params.upgrade:
            cmd.append("--upgrade")
        
        cmd.extend(self.params.extra_args)
        cmd.extend(packages_to_install)
        
        self._run_command(cmd)

    def _install_from_requirements(self) -> None:
        """Install packages from requirements file using pip."""
        requirements_file = self.params.requirements_file

        logger.info(
            "Installing from requirements file with pip",
            file=requirements_file,
        )

        # Build pip install command
        cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
        
        if self.params.upgrade:
            cmd.append("--upgrade")
        
        cmd.extend(self.params.extra_args)

        self._run_command(cmd)

    def _run_command(self, cmd: List[str]) -> None:
        """Run a command and handle errors."""
        logger.debug("Running command", command=" ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout:
                logger.debug("Command output", stdout=result.stdout)

        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"

            logger.error("Command execution failed", error=error_msg)
            raise RuntimeError(error_msg) from e
