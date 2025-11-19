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
"""Version command for nemo-evaluator-launcher."""

import importlib
from dataclasses import dataclass

from nemo_evaluator_launcher import __package_name__, __version__
from nemo_evaluator_launcher.common.logging_utils import logger


def get_versions() -> dict:
    internal_module_name = "nemo_evaluator_launcher_internal"
    res = {__package_name__: __version__}
    # Check for internal package
    try:
        internal_module = importlib.import_module(internal_module_name)
        # Try to get version from internal package
        internal_version = getattr(internal_module, "__version__", None)
        if internal_version:
            res[internal_module_name] = internal_version
        else:
            res[internal_module_name] = "available (version unknown)"
    except ImportError:
        # Internal package not available - this is expected in many cases
        pass
    except Exception as e:
        logger.error(f"nemo_evaluator_launcher_internal: error loading ({e})")
        raise

    return res


@dataclass
class Cmd:
    """Show version information for nemo-evaluator-launcher and internal packages."""

    def execute(self) -> None:
        """Execute the version command."""
        res = get_versions()
        for package, version in res.items():
            print(f"{package}: {version}")
