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

"""Adapter utilities for synced nemo-skills code."""

import dataclasses
import glob
from pathlib import Path
from typing import Any, List, Optional


def get_logger_name(file_path: str) -> str:
    """Returns a logger name derived from the given file path.

    Args:
        file_path: Path to the file

    Returns:
        Logger name derived from file path
    """
    # Simple implementation: use the module path relative to nemo_evaluator
    path = Path(file_path)
    return path.stem


def nested_dataclass(cls: Optional[type] = None, /, **kwargs: Any) -> type:
    """Thin wrapper around dataclasses.dataclass.

    Accepts all standard dataclass keyword arguments and returns a
    dataclass-decorated class.

    Args:
        cls: Class to decorate (when used without arguments)
        **kwargs: Standard dataclass keyword arguments

    Returns:
        Dataclass-decorated class
    """
    def wrap(c):
        return dataclasses.dataclass(c, **kwargs)

    if cls is None:
        return wrap
    return wrap(cls)


def unroll_files(file_patterns: List[str]) -> List[str]:
    """Expands glob patterns and returns a flat list of matching file paths.

    Args:
        file_patterns: List of glob patterns to expand

    Returns:
        Flat list of matching file paths

    Raises:
        FileNotFoundError: If any pattern matches zero files
    """
    result = []
    for pattern in file_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No files match pattern: {pattern}")
        result.extend(matches)
    return result


def get_help_message() -> str:
    """Returns the empty string. Stub only.

    Returns:
        Empty string
    """
    return ""
