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

"""File utilities adapter for synced nemo-skills code."""

import json
from typing import Any, List

from nemo_evaluator._nemo_skills._adapters.utils import unroll_files as _unroll_files


# Re-export unroll_files
unroll_files = _unroll_files


def jdump(data: Any, file_path: str, mode: str = "w") -> None:
    """Writes data as JSON to file_path with indent=2 and ensure_ascii=False.

    Args:
        data: Data to serialize as JSON
        file_path: Path to write JSON file
        mode: File open mode (default "w")
    """
    with open(file_path, mode, encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def jload(file_path: str) -> Any:
    """Reads and parses JSON from file_path using UTF-8 encoding.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
