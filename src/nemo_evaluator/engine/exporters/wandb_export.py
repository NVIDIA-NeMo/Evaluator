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
"""WandB exporter: log evaluation results to Weights & Biases."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WandBExporter:
    def __init__(
        self, project: str = "nemo-evaluator", entity: str | None = None, tags: list[str] | None = None, **kwargs: Any
    ) -> None:
        self._project = project
        self._entity = entity
        self._tags = tags or []

    def export(self, bundles: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is required: pip install wandb")

        run = wandb.init(
            project=self._project,
            entity=self._entity,
            tags=self._tags,
            config=config or {},
        )

        rows = []
        for bundle in bundles:
            bm = bundle.get("benchmark", {})
            name = bm.get("name", "unknown")
            scores = bm.get("scores", {})

            flat_scores = {}
            for metric, val in scores.items():
                if isinstance(val, dict) and "value" in val:
                    flat_scores[f"{name}/{metric}"] = val["value"]
                elif isinstance(val, (int, float)):
                    flat_scores[f"{name}/{metric}"] = val

            wandb.log(flat_scores)

            rows.append(
                {
                    "benchmark": name,
                    "samples": bm.get("samples", 0),
                    **{k.split("/")[-1]: v for k, v in flat_scores.items()},
                }
            )

        if rows:
            all_cols: list[str] = []
            seen: set[str] = set()
            for r in rows:
                for k in r:
                    if k not in seen:
                        all_cols.append(k)
                        seen.add(k)
            table = wandb.Table(
                columns=all_cols,
                data=[[r.get(c) for c in all_cols] for r in rows],
            )
            wandb.log({"results": table})

        run.finish()
        logger.info("Results exported to WandB project=%s", self._project)
