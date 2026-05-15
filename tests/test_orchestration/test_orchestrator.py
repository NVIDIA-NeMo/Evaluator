"""Tests for orchestration/orchestrator.py — wiring and factory tests, all mocked."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


class TestRunLocalArtifactAccess:
    def test_registers_background_eval_log_as_artifact(self, tmp_path):
        from nemo_evaluator.orchestration.orchestrator import run_local

        config = SimpleNamespace(
            cluster=SimpleNamespace(container_env={}),
            output=SimpleNamespace(
                dir=str(tmp_path),
                report=[],
                export=[],
                export_config={},
            ),
            services={},
            benchmarks=[],
        )

        with patch("nemo_evaluator.orchestration.orchestrator.apply_artifact_access") as apply_access:
            run_local(config)

        artifact_files = apply_access.call_args.kwargs["artifact_files"]
        assert tmp_path / "nel_eval.log" in artifact_files


class TestRunSingleBenchmarkCleanup:
    def _bench(self):
        return SimpleNamespace(
            name="bench",
            solver=SimpleNamespace(),
            sandbox=NoSandboxConfig(),
            max_concurrent=1,
            repeats=1,
            max_problems=1,
            scoring=SimpleNamespace(metrics=[]),
            skip_failed=False,
            max_system_retries=0,
            instruction_template=None,
            shuffle_seed=None,
        )

    def _config(self, tmp_path):
        return SimpleNamespace(
            output=SimpleNamespace(progress_interval=60, dir=str(tmp_path)),
            services={},
        )

    def _patch_eval_path(self, monkeypatch, tmp_path, *, eval_side_effect=None):
        from nemo_evaluator.orchestration import orchestrator

        class _Env:
            name = "bench@1.0"

            async def run_batch(self, **_kwargs):
                return None

        class _Proxy:
            async def async_stop(self):
                raise RuntimeError("proxy cleanup exploded")

        class _Judge:
            async def close(self):
                raise RuntimeError("judge cleanup exploded")

        class _TrafficStore:
            def close(self):
                raise RuntimeError("traffic cleanup exploded")

        bundle = {
            "run_id": "eval-ok",
            "benchmark": {"name": "bench@1.0", "scores": {"pass@1": {"value": 1.0}}},
        }

        async def _run_evaluation(*_args, **_kwargs):
            if eval_side_effect is not None:
                raise eval_side_effect
            return bundle

        monkeypatch.setattr(
            orchestrator,
            "_resolve_service_connection",
            lambda *_args: ("http://model", "m", None, None),
        )
        monkeypatch.setattr(
            orchestrator,
            "_start_proxy",
            lambda *_args: ("http://proxy", _Proxy(), _TrafficStore()),
        )
        monkeypatch.setattr(orchestrator, "_build_environment", lambda *_args: _Env())
        monkeypatch.setattr(orchestrator, "_create_client_and_solver", lambda *_args: (None, object()))
        monkeypatch.setattr(orchestrator, "_find_judge_client", lambda *_args: _Judge())
        monkeypatch.setattr(orchestrator, "_make_sandbox_manager", lambda *_args: None)
        monkeypatch.setattr("nemo_evaluator.engine.eval_loop.run_evaluation", _run_evaluation)
        monkeypatch.setattr(
            "nemo_evaluator.engine.artifacts.write_all",
            lambda _bundle, _task_dir: {"bundle": tmp_path / "bench@1.0" / "eval-ok.json"},
        )
        return bundle

    def test_cleanup_errors_do_not_replace_successful_bundle(self, tmp_path, monkeypatch, caplog):
        from nemo_evaluator.orchestration.orchestrator import _run_single_benchmark

        caplog.set_level(logging.ERROR)
        expected = self._patch_eval_path(monkeypatch, tmp_path)

        result = asyncio.run(
            _run_single_benchmark(
                self._bench(),
                self._config(tmp_path),
                {},
                tmp_path,
            )
        )

        assert result["run_id"] == expected["run_id"]
        assert "Failed to close judge client" in caplog.text
        assert "Failed to stop adapter proxy" in caplog.text
        assert "Failed to close model traffic store" in caplog.text

    def test_cleanup_errors_do_not_hide_eval_failure(self, tmp_path, monkeypatch, caplog):
        from nemo_evaluator.orchestration.orchestrator import _run_single_benchmark

        caplog.set_level(logging.ERROR)
        self._patch_eval_path(monkeypatch, tmp_path, eval_side_effect=RuntimeError("eval exploded"))

        with pytest.raises(RuntimeError, match="eval exploded"):
            asyncio.run(
                _run_single_benchmark(
                    self._bench(),
                    self._config(tmp_path),
                    {},
                    tmp_path,
                )
            )

        assert "Failed to close judge client" in caplog.text
        assert "Failed to stop adapter proxy" in caplog.text
        assert "Failed to close model traffic store" in caplog.text


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
