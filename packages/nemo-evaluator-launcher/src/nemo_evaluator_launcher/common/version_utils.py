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

"""Version information for nemo-evaluator-launcher."""

import sys
from pathlib import Path

# Conditional import for TOML parsing
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.logging_utils import logger

# Get the path to pyproject.toml relative to this file
_pyproject_path = Path(__file__).parents[3] / "pyproject.toml"


def _get_name() -> str:
    """Read version from pyproject.toml.

    Raises:
        FileNotFoundError, KeyError, tomllib.TOMLDecodeError: if pyproject can't be found
           this fails fast if the structure has changed
    """
    with open(_pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["name"]


def _get_version() -> str:
    """Read version from pyproject.toml.

    Raises:
        FileNotFoundError, KeyError, tomllib.TOMLDecodeError: if pyproject can't be found
           this fails fast if the structure has changed
    """
    with open(_pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]


__main_pkg_name__ = _get_name()
__version__ = _get_version()

logger.info("Version info", pkg=__main_pkg_name__, ver=__version__)
