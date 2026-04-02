"""Tests for orchestration/orchestrator.py — wiring and factory tests, all mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestMakeSolver:
    @patch("nemo_evaluator.orchestration.orchestrator.ChatSolver")
    def test_simple_chat_solver(self, MockChat):
        from nemo_evaluator.orchestration.orchestrator import _make_solver

        bench = MagicMock()
        bench.solver = MagicMock()
        bench.solver.__class__.__name__ = "SimpleSolver"
        type(bench.solver).__name__ = "SimpleSolver"
        bench.solver.protocol = "chat"
        bench.solver.image_detail = None
        bench.solver.system_prompt = None

        config = MagicMock()
        client = MagicMock()

        solver = _make_solver(bench, config, client, "http://localhost:8000/v1", "m", "k")
        assert solver is not None

    @patch("nemo_evaluator.orchestration.orchestrator.CompletionSolver")
    def test_simple_completion_solver(self, MockCompletion):
        from nemo_evaluator.orchestration.orchestrator import _make_solver

        bench = MagicMock()
        bench.solver = MagicMock()
        type(bench.solver).__name__ = "SimpleSolver"
        bench.solver.protocol = "completions"
        bench.solver.image_detail = None

        config = MagicMock()
        client = MagicMock()

        solver = _make_solver(bench, config, client, "http://localhost:8000/v1", "m", "k")
        assert solver is not None


class TestMakeSandboxManager:
    def test_no_sandbox_returns_none(self):
        from nemo_evaluator.orchestration.orchestrator import _make_sandbox_manager

        sb = MagicMock()
        type(sb).__name__ = "NoSandbox"
        result = _make_sandbox_manager(sb)
        assert result is None


class TestRunLocal:
    """Smoke test that run_local wiring doesn't crash with fully mocked deps."""

    @patch("nemo_evaluator.orchestration.orchestrator._generate_reports")
    @patch("nemo_evaluator.orchestration.orchestrator._run_single_benchmark")
    @patch("nemo_evaluator.orchestration.orchestrator.CheckpointManager")
    def test_empty_benchmarks(self, MockCkpt, mock_run_bench, mock_reports):
        from nemo_evaluator.orchestration.orchestrator import run_local

        config = MagicMock()
        config.benchmarks = []
        config.services = {}
        config.output.dir = "/tmp/test-run"
        config.output.run_name = None
        config.cluster = MagicMock()
        config.cluster.container_env = {}
        config.exporters = []

        ckpt = MagicMock()
        ckpt.is_completed.return_value = False
        MockCkpt.return_value = ckpt

        bundles = run_local(config)
        assert bundles == []
        mock_run_bench.assert_not_called()
