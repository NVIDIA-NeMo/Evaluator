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
"""Tests for orchestration/orchestrator.py — wiring and factory tests, all mocked."""

from __future__ import annotations

from unittest.mock import MagicMock


from nemo_evaluator.config.sandboxes import NoSandbox as NoSandboxConfig


class TestMakeSandboxManager:
    def test_no_sandbox_returns_none(self):
        from nemo_evaluator.orchestration.orchestrator import _make_sandbox_manager

        sb = NoSandboxConfig()
        result = _make_sandbox_manager(sb)
        assert result is None


class TestSafeName:
    def test_sanitizes_special_chars(self):
        from nemo_evaluator.orchestration.orchestrator import _safe_name

        result = _safe_name("hello/world:v1")
        assert "/" not in result
        assert ":" not in result

    def test_replaces_whitespace(self):
        from nemo_evaluator.orchestration.orchestrator import _safe_name

        result = _safe_name("  hello world  ")
        assert " " not in result


class TestInterceptorSpecs:
    def test_returns_list_from_configs(self):
        from nemo_evaluator.orchestration.orchestrator import _interceptor_specs

        ic = MagicMock()
        ic.name = "turn_counter"
        ic.config = {"max_turns": 200}
        result = _interceptor_specs([ic])
        assert len(result) == 1
        assert result[0]["name"] == "turn_counter"
        assert result[0]["config"]["max_turns"] == 200

    def test_empty_list(self):
        from nemo_evaluator.orchestration.orchestrator import _interceptor_specs

        assert _interceptor_specs([]) == []


class TestGenerateReports:
    """Smoke test: _generate_reports can import from reports.eval and produce output."""

    def test_generate_reports_smoke(self, tmp_path):
        import json

        from nemo_evaluator.orchestration.orchestrator import _generate_reports

        bundle = {
            "run_id": "smoke-test",
            "benchmark": {
                "name": "test_bench",
                "samples": 2,
                "repeats": 1,
                "scores": {"mean_reward": {"value": 0.75}},
            },
            "config": {"model": "test-model"},
        }
        (tmp_path / "eval-smoke.json").write_text(json.dumps(bundle), encoding="utf-8")

        config = MagicMock()
        config.output.report = ["markdown"]
        config.output.export = []

        _generate_reports(config, tmp_path)

        md = tmp_path / "report.md"
        assert md.exists(), "Markdown report not generated"
        content = md.read_text()
        assert "test-model" in content
        assert "test_bench" in content


class TestResolveGeneration:
    def test_returns_generation_config(self):
        from nemo_evaluator.config import GenerationConfig, EvalConfig
        from nemo_evaluator.orchestration.orchestrator import _resolve_generation

        config = MagicMock(spec=EvalConfig)
        svc = MagicMock()
        svc.generation = GenerationConfig(temperature=0.7, max_tokens=100)
        config.get_service.return_value = svc

        solver_cfg = MagicMock()
        solver_cfg.service = "my-svc"
        solver_cfg.generation = None

        result = _resolve_generation(config, solver_cfg)
        assert result.temperature == 0.7
        assert result.max_tokens == 100
