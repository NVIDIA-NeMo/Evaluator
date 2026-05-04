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
"""Tests for engine/exporters — contract tests with mocked backends."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestExporterRegistry:
    def test_lazy_load_populates(self):
        from nemo_evaluator.engine.exporters import _lazy_load, _REGISTRY

        _lazy_load()
        assert "wandb" in _REGISTRY
        assert "mlflow" in _REGISTRY
        assert "inspect" in _REGISTRY

    def test_get_unknown_raises(self):
        from nemo_evaluator.engine.exporters import get_exporter

        with pytest.raises(KeyError, match="Unknown exporter"):
            get_exporter("nonexistent_exporter_xyz")


class TestWandBExporter:
    def test_init(self):
        from nemo_evaluator.engine.exporters.wandb_export import WandBExporter

        exp = WandBExporter(project="test-proj", entity="test-entity", tags=["ci"])
        assert exp._project == "test-proj"
        assert exp._entity == "test-entity"
        assert exp._tags == ["ci"]

    def test_export_calls_wandb(self):
        from nemo_evaluator.engine.exporters.wandb_export import WandBExporter

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        exp = WandBExporter(project="test")
        bundles = [
            {
                "benchmark": {
                    "name": "mmlu",
                    "samples": 10,
                    "scores": {"pass@1": {"value": 0.8}},
                },
                "_results": [],
            }
        ]
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            exp.export(bundles)
        mock_wandb.init.assert_called_once()


class TestMLflowExporter:
    def test_init(self):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        exp = MLflowExporter(experiment_name="test-exp")
        assert exp._experiment_name == "test-exp"

    def test_export_logs_metrics(self):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        mock_mlflow = MagicMock()
        exp = MLflowExporter(experiment_name="test-exp")
        bundles = [
            {
                "benchmark": {
                    "name": "gsm8k",
                    "samples": 5,
                    "scores": {"pass@1": {"value": 0.6}},
                },
                "_results": [],
            }
        ]
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            exp.export(bundles)
        assert mock_mlflow.set_experiment.called or mock_mlflow.start_run.called


class TestInspectExporter:
    def test_init(self):
        pytest.importorskip("inspect_ai")
        from nemo_evaluator.engine.exporters.inspect_export import InspectExporter

        exp = InspectExporter(format="json")
        assert exp._format == "json"
