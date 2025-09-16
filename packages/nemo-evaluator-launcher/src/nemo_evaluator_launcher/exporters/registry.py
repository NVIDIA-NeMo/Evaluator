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
from typing import Callable, Dict

from nemo_evaluator_launcher.exporters.base import BaseExporter

_EXPORTER_REGISTRY: Dict[str, BaseExporter] = {}


def register_exporter(name: str) -> Callable:
    def wrapper(cls):
        _EXPORTER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_exporter(name: str) -> BaseExporter:
    if name not in _EXPORTER_REGISTRY:
        raise ValueError(
            f"Unknown exporter: {name}. Available: {list(_EXPORTER_REGISTRY.keys())}"
        )
    return _EXPORTER_REGISTRY[name]


def available_exporters() -> list[str]:
    return list(_EXPORTER_REGISTRY.keys())
