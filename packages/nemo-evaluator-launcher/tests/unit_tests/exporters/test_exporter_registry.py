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
"""Tests for exporter registry and core interface consistency."""

import pytest

from nemo_evaluator_launcher.exporters import available_exporters, create_exporter
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
        exporter = create_exporter("wandb", {"entity": "e", "project": "p"})
        assert isinstance(exporter, WandBExporter)

    def test_create_exporter_mlflow(self):
        exporter = create_exporter("mlflow", {"tracking_uri": "http://mlflow"})
        assert isinstance(exporter, MLflowExporter)

    def test_create_exporter_invalid(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            create_exporter("invalid_exporter")

    def test_available_exporters_keys(self):
        keys = set(available_exporters())
        assert {"local", "mlflow", "wandb", "gsheets"}.issubset(keys)
