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

"""Destination-specific exporter tests: GSheets, WandB, MLflow, Leaderboard."""

from unittest.mock import Mock, patch

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.exporters.mlflow import MLflowExporter
from nemo_evaluator_launcher.exporters.wandb import WandBExporter


class TestGSheetsExporter:
    def test_placeholder(self):
        # GSheets tests moved or covered elsewhere; keep file scaffold minimal.
        assert True


class TestWandBExporter:
    def test_wandb_not_available(self):
        with patch("nemo_evaluator_launcher.exporters.wandb.WANDB_AVAILABLE", False):
            exporter = WandBExporter()
            result = exporter.export_job(Mock())
            assert not result.success
            assert "not installed" in result.message

    def test_wandb_export_no_metrics(self):
        job_data = JobData(
            invocation_id="test1234",
            job_id="test1234.0",
            timestamp=1234567890.0,
            executor="local",
            data={"output_dir": "/tmp/test"},
        )
        with patch("nemo_evaluator_launcher.exporters.wandb.WANDB_AVAILABLE", True):
            with patch(
                "nemo_evaluator_launcher.exporters.wandb.extract_accuracy_metrics",
                return_value={},
            ):
                exporter = WandBExporter({"entity": "test", "project": "test"})
                result = exporter.export_job(job_data)
                assert not result.success
                assert "No metrics found" in result.message


class TestMLflowExporter:
    def test_mlflow_not_available(self):
        with patch("nemo_evaluator_launcher.exporters.mlflow.MLFLOW_AVAILABLE", False):
            exporter = MLflowExporter()
            result = exporter.export_job(Mock())
            assert not result.success
            assert "not installed" in result.message
