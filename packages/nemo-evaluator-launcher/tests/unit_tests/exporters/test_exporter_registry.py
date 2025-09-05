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

"""Tests for exporter registry and core interface consistency."""

import pytest
from nemo_evaluator_launcher.exporters import create_exporter
from nemo_evaluator_launcher.exporters.gsheets import GSheetsExporter
from nemo_evaluator_launcher.exporters.local import LocalExporter
from nemo_evaluator_launcher.exporters.mlflow import MLflowExporter
from nemo_evaluator_launcher.exporters.wandb import WandBExporter


class TestExporterRegistry:
    def test_create_exporter_local(self):
        exporter = create_exporter("local")
        assert isinstance(exporter, LocalExporter)

    def test_create_exporter_gsheets(self):
        exporter = create_exporter("gsheets")
        assert isinstance(exporter, GSheetsExporter)

    def test_create_exporter_wandb(self):
        exporter = create_exporter("wandb")
        assert isinstance(exporter, WandBExporter)

    def test_create_exporter_mlflow(self):
        exporter = create_exporter("mlflow")
        assert isinstance(exporter, MLflowExporter)

    def test_create_exporter_invalid(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            create_exporter("invalid_exporter")


class TestExporterConsistency:
    @pytest.mark.parametrize("exporter_name", ["local", "gsheets", "wandb", "mlflow"])
    def test_exporter_has_required_methods(self, exporter_name):
        exporter = create_exporter(exporter_name)
        assert hasattr(exporter, "export_job")
        assert hasattr(exporter, "supports_executor")
        assert callable(exporter.export_job)
        assert callable(exporter.supports_executor)

    @pytest.mark.parametrize("exporter_name", ["local", "gsheets", "wandb", "mlflow"])
    def test_exporter_supports_executor_returns_bool(self, exporter_name):
        exporter = create_exporter(exporter_name)
        for executor_type in ["local", "slurm", "gitlab", "unknown"]:
            result = exporter.supports_executor(executor_type)
            assert isinstance(result, bool)

    def test_local_exporter_supports_all_executors(self):
        exporter = LocalExporter()
        assert exporter.supports_executor("local") is True
        assert exporter.supports_executor("slurm") is True
        assert exporter.supports_executor("gitlab") is True

    def test_other_exporters_support_local(self):
        exporters = [
            GSheetsExporter(),
            WandBExporter(),
            MLflowExporter(),
        ]
        for exporter in exporters:
            assert exporter.supports_executor("local") is True
