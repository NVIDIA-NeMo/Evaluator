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
