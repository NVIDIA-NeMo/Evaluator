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
"""NeMo Evaluator Launcher - Public API.

This package provides the public API for the NeMo Evaluator Launcher.
It automatically initializes logging and conditionally loads internal components.
"""

import importlib
import warnings

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.package_info import (
    __contact_emails__,
    __contact_names__,
    __download_url__,
    __homepage__,
    __package_name__,
    __repository_url__,
    __version__,
)

# Suppress pydantic warnings from third-party libraries (e.g., wandb) that are not
# compatible with Pydantic 2.x field metadata on Python 3.13+
warnings.filterwarnings(
    "ignore",
    message=r"The 'repr' attribute.*Field\(\).*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The 'frozen' attribute.*Field\(\).*",
    category=Warning,
)


logger.info("Version info", pkg=__package_name__, ver=__version__)

try:
    importlib.import_module("nemo_evaluator_launcher_internal")
    logger.debug(
        "Successfully loaded internal package",
        package="nemo_evaluator_launcher_internal",
    )
except ImportError as e:
    logger.debug(
        "Internal package not available",
        package="nemo_evaluator_launcher_internal",
        error=str(e),
    )
except Exception as e:
    logger.warning(
        "Failed to load internal package",
        package="nemo_evaluator_launcher_internal",
        error=str(e),
    )

__all__ = [
    "__version__",
    "__package_name__",
    "__contact_names__",
    "__contact_emails__",
    "__homepage__",
    "__repository_url__",
    "__download_url__",
]
