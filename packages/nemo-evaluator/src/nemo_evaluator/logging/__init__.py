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

"""Logging module for nemo-evaluator.

This module provides centralized logging configuration, utilities, and request context
management for the nemo-evaluator package.
"""

from .config import BaseLoggingParams
from .context import (
    bind_request_id,
    get_bound_logger,
    get_current_request_id,
    request_context,
)
from .utils import configure_logging, get_logger

__all__ = [
    "BaseLoggingParams",
    "get_logger",
    "configure_logging",
    "bind_request_id",
    "request_context",
    "get_current_request_id",
    "get_bound_logger",
]
