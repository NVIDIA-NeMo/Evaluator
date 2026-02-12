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

"""Dataset utilities adapter for synced nemo-skills code."""

import importlib
import importlib.util
from types import ModuleType
from typing import Any, Optional


def get_dataset_module(benchmark: str, data_dir: Optional[str] = None, **kwargs: Any) -> ModuleType:
    """Dynamically imports and returns nemo_evaluator._nemo_skills.dataset.{benchmark}.

    Args:
        benchmark: Benchmark name
        data_dir: Data directory (unused in this stub)
        **kwargs: Additional keyword arguments (unused in this stub)

    Returns:
        Imported module

    Raises:
        ModuleNotFoundError: If the module does not exist
    """
    module_name = f"nemo_evaluator._nemo_skills.dataset.{benchmark}"
    return importlib.import_module(module_name)


def import_from_path(module_path_str: str) -> ModuleType:
    """Imports a Python module from an absolute filesystem path.

    Args:
        module_path_str: Absolute path to Python module file

    Returns:
        Imported module

    Raises:
        FileNotFoundError: If the path does not exist
        ImportError: If the module cannot be loaded
    """
    spec = importlib.util.spec_from_file_location("_dynamic_module", module_path_str)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path_str}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
