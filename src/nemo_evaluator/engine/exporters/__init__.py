# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Export plugins: push evaluation results to experiment trackers."""

from __future__ import annotations

from typing import Any

from nemo_evaluator.engine.exporters.base import ExportPlugin

_REGISTRY: dict[str, type[ExportPlugin]] = {}


def register_exporter(name: str, cls: type[ExportPlugin]) -> None:
    _REGISTRY[name] = cls


def get_exporter(name: str, **kwargs: Any) -> ExportPlugin:
    if name not in _REGISTRY:
        _lazy_load()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown exporter {name!r}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def _lazy_load() -> None:
    from nemo_evaluator.engine.exporters.wandb_export import WandBExporter

    register_exporter("wandb", WandBExporter)

    from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

    register_exporter("mlflow", MLflowExporter)

    from nemo_evaluator.engine.exporters.inspect_export import InspectExporter

    register_exporter("inspect", InspectExporter)
