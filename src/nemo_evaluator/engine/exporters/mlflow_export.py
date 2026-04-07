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
"""MLflow exporter: log evaluation results to MLflow tracking."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MLflowExporter:
    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "nemo-evaluator", **kwargs: Any) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def export(self, bundles: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow is required: pip install mlflow")

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)

        mlflow.set_experiment(self._experiment_name)

        with mlflow.start_run():
            if config:
                mlflow.log_params(_flatten_dict(config, max_depth=2))

            for bundle in bundles:
                bm = bundle.get("benchmark", {})
                name = bm.get("name", "unknown")
                scores = bm.get("scores", {})

                for metric, val in scores.items():
                    if isinstance(val, dict) and "value" in val:
                        key = f"{name}/{metric}".replace("@", "_at_")
                        mlflow.log_metric(key, val["value"])
                    elif isinstance(val, (int, float)):
                        key = f"{name}/{metric}".replace("@", "_at_")
                        mlflow.log_metric(key, val)

            with tempfile.TemporaryDirectory() as tmpdir:
                for i, bundle in enumerate(bundles):
                    p = Path(tmpdir) / f"bundle_{i}.json"
                    p.write_text(json.dumps(bundle, default=str, indent=2))
                    mlflow.log_artifact(str(p))

        logger.info("Results exported to MLflow experiment=%s", self._experiment_name)


def _flatten_dict(d: dict, prefix: str = "", max_depth: int = 3) -> dict[str, str]:
    """Flatten nested dict for MLflow params (which must be flat strings)."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and max_depth > 0:
            items.update(_flatten_dict(v, key, max_depth - 1))
        else:
            items[key] = str(v)[:250]
    return items
