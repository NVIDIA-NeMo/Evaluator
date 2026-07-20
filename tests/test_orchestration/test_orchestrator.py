# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for orchestration/orchestrator.py — wiring and factory tests, all mocked."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any
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


class TestStartProxyFinishReason:
    def _run(self, svc: object, config: object | None = None) -> dict[str, Any]:
        import nemo_evaluator.adapters.proxy as proxy_mod
        from nemo_evaluator.orchestration.orchestrator import _start_proxy

        captured: dict[str, Any] = {}

        class _Handle:
            url = "http://127.0.0.1:0"

        def _fake_start(**kwargs: Any) -> _Handle:
            captured.update(kwargs)
            return _Handle()

        with patch.object(proxy_mod, "start_adapter_proxy", _fake_start):
            _url, _handle, store = _start_proxy("http://upstream:8000", "m", None, svc, "svc", config)
        if store is not None:
            store.close()
        return captured

    def _names(self, svc: object, config: object | None = None) -> list[str]:
        return [s["name"] for s in (self._run(svc, config).get("interceptor_specs") or [])]

    def test_injected_by_default_and_last(self):
        svc = MagicMock()
        svc.proxy = None
        svc.generation = None
        names = self._names(svc)
        assert names[-1] == "finish_reason"

    def test_injected_after_payload_modifier(self):
        svc = MagicMock()
        svc.proxy = None
        gen = MagicMock()
        gen.model_dump.return_value = {"temperature": 0.0}
        svc.generation = gen
        names = self._names(svc)
        assert names == ["payload_modifier", "finish_reason"]

    def test_docker_sandbox_proxy_listens_on_all_interfaces(self):
        svc = MagicMock()
        svc.proxy = None
        svc.generation = None
        config = SimpleNamespace(benchmarks=[SimpleNamespace(sandbox=SimpleNamespace(type="docker"))])

        captured = self._run(svc, config)

        assert captured["listen_host"] == "0.0.0.0"

    def test_non_docker_proxy_stays_loopback_only(self):
        svc = MagicMock()
        svc.proxy = None
        svc.generation = None
        config = SimpleNamespace(benchmarks=[SimpleNamespace(sandbox=SimpleNamespace(type="none"))])

        captured = self._run(svc, config)

        assert captured["listen_host"] == "127.0.0.1"

    def test_not_duplicated_when_configured(self):
        from nemo_evaluator.config.services import InterceptorConfig, ProxyConfig

        svc = MagicMock()
        svc.generation = None
        svc.proxy = ProxyConfig(interceptors=[InterceptorConfig(name="finish_reason")])
        names = self._names(svc)
        assert names.count("finish_reason") == 1

    def test_preconfigured_finish_reason_moved_to_end(self):
        from nemo_evaluator.config.services import InterceptorConfig, ProxyConfig

        svc = MagicMock()
        gen = MagicMock()
        gen.model_dump.return_value = {"max_tokens": 16}
        svc.generation = gen
        svc.proxy = ProxyConfig(
            interceptors=[
                InterceptorConfig(name="finish_reason"),
                InterceptorConfig(name="turn_counter"),
            ]
        )
        names = self._names(svc)
        assert names.count("finish_reason") == 1
        assert names[-1] == "finish_reason"
        assert names.index("finish_reason") > names.index("payload_modifier")

    def test_opt_out_via_proxy_flag(self):
        from nemo_evaluator.config.services import ProxyConfig

        svc = MagicMock()
        svc.generation = None
        svc.proxy = ProxyConfig(finish_reason_fixup=False)
        names = self._names(svc)
        assert "finish_reason" not in names

    def test_opt_out_still_keeps_other_interceptors(self):
        from nemo_evaluator.config.services import InterceptorConfig, ProxyConfig

        svc = MagicMock()
        svc.generation = None
        svc.proxy = ProxyConfig(
            interceptors=[InterceptorConfig(name="turn_counter")],
            finish_reason_fixup=False,
        )
        names = self._names(svc)
        assert names == ["turn_counter"]


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
        from nemo_evaluator.config import EvalConfig, GenerationConfig
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


class TestInstallDefaultExecutor:
    def test_replaces_default_executor_with_given_max_workers(self):
        from nemo_evaluator.orchestration.orchestrator import _install_default_executor

        async def _runner():
            executor = _install_default_executor(50)
            loop = asyncio.get_running_loop()
            assert loop._default_executor is executor  # type: ignore[attr-defined]
            assert executor._max_workers == 50

        asyncio.run(_runner())


class TestBuildBatchConfigEnvVars:
    """``_build_batch_config`` merges ``cluster.container_env`` (global) under
    ``solver.env_vars`` (per-benchmark) so users can declare shared env vars
    once at cluster scope and override per-benchmark when needed."""

    def _container_solver(self, env_vars=None):
        from nemo_evaluator.config.solvers import ContainerSolverConfig

        return ContainerSolverConfig(service="endpoint", env_vars=env_vars or {})

    def test_cluster_container_env_propagates_for_container_solver(self):
        from nemo_evaluator.config.clusters import LocalCluster
        from nemo_evaluator.orchestration.orchestrator import _build_batch_config

        cluster = LocalCluster(container_env={"HF_TOKEN": "tok", "HF_HOME": "/cache"})
        solver_cfg = self._container_solver()
        svc = MagicMock(protocol="chat_completions")

        cfg = _build_batch_config("u", "m", None, solver_cfg, svc, cluster=cluster)
        assert cfg["env_vars"] == {"HF_TOKEN": "tok", "HF_HOME": "/cache"}

    def test_solver_env_vars_override_cluster(self):
        from nemo_evaluator.config.clusters import LocalCluster
        from nemo_evaluator.orchestration.orchestrator import _build_batch_config

        cluster = LocalCluster(container_env={"HF_TOKEN": "global", "HF_HOME": "/cache"})
        solver_cfg = self._container_solver(env_vars={"HF_TOKEN": "per-bench"})
        svc = MagicMock(protocol="chat_completions")

        cfg = _build_batch_config("u", "m", None, solver_cfg, svc, cluster=cluster)
        assert cfg["env_vars"] == {"HF_TOKEN": "per-bench", "HF_HOME": "/cache"}
