# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the config module."""

import pytest


class TestLoadConfig:
    """Tests for load_config."""

    def test_returns_defaults_when_no_file(self, tmp_path):
        from nemo_evaluator.config import NemoEvaluatorConfig, TelemetryLevel, load_config

        cfg = load_config(path=tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, NemoEvaluatorConfig)
        assert cfg.telemetry.level == TelemetryLevel.DEFAULT

    def test_reads_valid_file(self, tmp_path):
        from nemo_evaluator.config import TelemetryLevel, load_config, save_config, NemoEvaluatorConfig, TelemetryConfig

        config_file = tmp_path / "config.yaml"
        save_config(
            NemoEvaluatorConfig(telemetry=TelemetryConfig(level=TelemetryLevel.OFF)),
            path=config_file,
        )
        loaded = load_config(path=config_file)
        assert loaded.telemetry.level == TelemetryLevel.OFF

    def test_raises_on_invalid_data(self, tmp_path):
        from nemo_evaluator.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("telemetry:\n  level: 99\n")
        with pytest.raises(Exception):
            load_config(path=config_file)

    def test_raises_on_malformed_yaml(self, tmp_path):
        from nemo_evaluator.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(": :\n  bad yaml [[[")
        with pytest.raises(Exception):
            load_config(path=config_file)


class TestSaveConfig:
    """Tests for save_config."""

    def test_round_trip(self, tmp_path):
        from nemo_evaluator.config import (
            NemoEvaluatorConfig,
            TelemetryConfig,
            TelemetryLevel,
            load_config,
            save_config,
        )

        config_file = tmp_path / "config.yaml"
        cfg = NemoEvaluatorConfig(
            telemetry=TelemetryConfig(level=TelemetryLevel.MINIMAL)
        )
        save_config(cfg, path=config_file)
        loaded = load_config(path=config_file)
        assert loaded.telemetry.level == TelemetryLevel.MINIMAL

    def test_creates_parent_dirs(self, tmp_path):
        from nemo_evaluator.config import NemoEvaluatorConfig, save_config

        config_file = tmp_path / "nested" / "dir" / "config.yaml"
        save_config(NemoEvaluatorConfig(), path=config_file)
        assert config_file.exists()

    def test_produces_valid_yaml(self, tmp_path):
        import yaml

        from nemo_evaluator.config import NemoEvaluatorConfig, TelemetryConfig, TelemetryLevel, save_config

        config_file = tmp_path / "config.yaml"
        save_config(
            NemoEvaluatorConfig(telemetry=TelemetryConfig(level=TelemetryLevel.OFF)),
            path=config_file,
        )
        data = yaml.safe_load(config_file.read_text())
        assert data == {"telemetry": {"level": 0}}
