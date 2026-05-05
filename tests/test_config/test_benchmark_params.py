# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``BenchmarkConfig.params`` and ``get_environment`` forwarding."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_evaluator.config.benchmarks import BenchmarkConfig
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import (
    _REGISTRY,
    _filter_init_kwargs,
    get_environment,
    register,
)

# ---------------------------------------------------------------------------
# BenchmarkConfig schema
# ---------------------------------------------------------------------------


_MINIMAL_SOLVER = {"type": "simple", "service": "svc"}


def test_benchmarkconfig_params_default_empty():
    cfg = BenchmarkConfig.model_validate({"name": "dummy", "solver": _MINIMAL_SOLVER})
    assert cfg.params == {}


def test_benchmarkconfig_shuffle_seed_default_is_42():
    """Default-on shuffling with seed 42 gives reproducible shard balance
    without requiring users to opt in."""
    cfg = BenchmarkConfig.model_validate({"name": "dummy", "solver": _MINIMAL_SOLVER})
    assert cfg.shuffle_seed == 42


def test_benchmarkconfig_shuffle_seed_none_disables_shuffle():
    cfg = BenchmarkConfig.model_validate({"name": "dummy", "solver": _MINIMAL_SOLVER, "shuffle_seed": None})
    assert cfg.shuffle_seed is None


def test_benchmarkconfig_shuffle_seed_custom_int():
    cfg = BenchmarkConfig.model_validate({"name": "dummy", "solver": _MINIMAL_SOLVER, "shuffle_seed": 7})
    assert cfg.shuffle_seed == 7


def test_benchmarkconfig_params_roundtrip():
    cfg = BenchmarkConfig.model_validate(
        {
            "name": "dummy",
            "solver": _MINIMAL_SOLVER,
            "params": {"dataset_path": "/tmp/x", "some_flag": True},
        }
    )
    assert cfg.params == {"dataset_path": "/tmp/x", "some_flag": True}
    assert cfg.model_dump()["params"] == {"dataset_path": "/tmp/x", "some_flag": True}


# ---------------------------------------------------------------------------
# get_environment kwarg forwarding
# ---------------------------------------------------------------------------


class _ParamsCapture(EvalEnvironment):
    """Stub env that records what constructor kwargs it received."""

    def __init__(self, dataset_path: str, flag: bool = False, num_examples: int | None = None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.flag = flag
        self.num_examples = num_examples

    async def dataset_size(self) -> int:
        return 0

    async def seed(self, idx: int) -> SeedResult:  # pragma: no cover - never called
        return SeedResult(prompt="", expected_answer=None)

    async def verify(self, response: str, expected: object) -> VerifyResult:  # pragma: no cover
        return VerifyResult(reward=0.0)


@pytest.fixture
def _params_capture_registered():
    name = "_params_capture_test"
    register(name)(_ParamsCapture)
    try:
        yield name
    finally:
        _REGISTRY.pop(name, None)


def test_get_environment_forwards_user_params(_params_capture_registered):
    env = get_environment(
        _params_capture_registered,
        params={"dataset_path": "/tmp/x", "flag": True},
    )
    assert isinstance(env, _ParamsCapture)
    assert env.dataset_path == "/tmp/x"
    assert env.flag is True


def test_get_environment_drops_irrelevant_auto_kwargs(_params_capture_registered):
    """``num_fewshot`` is injected by the orchestrator but our env doesn't take it."""
    env = get_environment(
        _params_capture_registered,
        num_examples=5,
        num_fewshot=2,
        params={"dataset_path": "/tmp/x"},
    )
    assert env.num_examples == 5


def test_get_environment_unknown_param_raises(_params_capture_registered):
    with pytest.raises(TypeError) as excinfo:
        get_environment(
            _params_capture_registered,
            params={"dataset_path": "/tmp/x", "unknown_arg": 123},
        )
    assert "unknown_arg" in str(excinfo.value)


def test_get_environment_missing_required_param_raises(_params_capture_registered):
    with pytest.raises(TypeError):
        get_environment(_params_capture_registered, params={})


# ---------------------------------------------------------------------------
# _filter_init_kwargs unit
# ---------------------------------------------------------------------------


def test_filter_init_kwargs_keeps_only_declared():
    kept = _filter_init_kwargs(_ParamsCapture, {"dataset_path": "x", "bogus": 1, "flag": False})
    assert kept == {"dataset_path": "x", "flag": False}


def test_filter_init_kwargs_passthrough_for_varkw():
    class _Varkw:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kept = _filter_init_kwargs(_Varkw, {"anything": 1, "else": 2})
    assert kept == {"anything": 1, "else": 2}


# ---------------------------------------------------------------------------
# nmp_harbor registration smoke test
# ---------------------------------------------------------------------------


def _make_fake_nmp_repo(root: Path, *, task_names=("t0",)) -> Path:
    (root / "Dockerfile.harbor").write_text("FROM python:3.12-slim\n")
    tasks_root = root / "tests" / "agentic-use"
    tasks_root.mkdir(parents=True)
    for name in task_names:
        (tasks_root / name).mkdir()
        (tasks_root / name / "instruction.md").write_text(f"do {name}")
    return root


def test_nmp_harbor_registered_and_instantiates(tmp_path: Path, monkeypatch):
    _make_fake_nmp_repo(tmp_path)
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    from nemo_evaluator.benchmarks.nmp_harbor import NmpHarborEnvironment

    env = get_environment("nmp_harbor")
    assert isinstance(env, NmpHarborEnvironment)
    assert env._base_image == "nmp-harbor:latest"
    assert env._base_dockerfile == (tmp_path / "Dockerfile.harbor").resolve()
    assert env._base_context == tmp_path.resolve()


def test_nmp_harbor_missing_repo_raises(monkeypatch):
    monkeypatch.delenv("NMP_REPO", raising=False)
    with pytest.raises(ValueError, match="NMP_REPO"):
        get_environment("nmp_harbor")


def test_nmp_harbor_explicit_nmp_repo_param(tmp_path: Path, monkeypatch):
    _make_fake_nmp_repo(tmp_path)
    monkeypatch.delenv("NMP_REPO", raising=False)

    env = get_environment("nmp_harbor", params={"nmp_repo": str(tmp_path)})
    assert env._base_context == tmp_path.resolve()


def test_nmp_harbor_task_names_filters(tmp_path: Path, monkeypatch):
    _make_fake_nmp_repo(tmp_path, task_names=("alpha", "beta", "gamma"))
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor", params={"task_names": ["beta"]})
    assert [t.name for t in env._tasks] == ["beta"]


def test_nmp_harbor_task_names_unknown_raises(tmp_path: Path, monkeypatch):
    _make_fake_nmp_repo(tmp_path, task_names=("alpha",))
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    with pytest.raises(ValueError, match="task_names not found"):
        get_environment("nmp_harbor", params={"task_names": ["does-not-exist"]})


def test_nmp_harbor_task_names_accepts_scalar_string(tmp_path: Path, monkeypatch):
    """``-O params.task_names=foo`` passes a scalar string; coerce to list."""
    _make_fake_nmp_repo(tmp_path, task_names=("alpha", "beta"))
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor", params={"task_names": "beta"})
    assert [t.name for t in env._tasks] == ["beta"]


def test_nmp_harbor_image_build_requests_prepends_base(tmp_path: Path, monkeypatch):
    import asyncio

    _make_fake_nmp_repo(tmp_path)
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor")
    reqs = asyncio.run(env.image_build_requests())
    assert reqs, "expected at least the base-image request"
    assert reqs[0].specs[0].image == "nmp-harbor:latest"
    assert reqs[0].docker_build_fn is not None


def _make_fake_full_task(root: Path, name: str = "t0") -> Path:
    """Create a minimal task layout ``HarborEnvironment.seed()`` can consume.

    We provide an ``environment/Dockerfile`` with a single ``FROM`` line so
    ``_resolve_image`` returns a non-None image and both ``sandbox_spec`` and
    ``verify_sandbox_spec`` are populated.
    """
    task = root / "tests" / "agentic-use" / name
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text(f"do {name}\n")
    (task / "environment" / "Dockerfile").write_text("FROM nmp-harbor:latest\n")
    return root


def test_nmp_harbor_seed_forces_stateful(tmp_path: Path, monkeypatch):
    """NMP state lives in the API process; seed() must force StatefulSandbox."""
    import asyncio

    _make_fake_full_task(tmp_path)
    (tmp_path / "Dockerfile.harbor").write_text("FROM python:3.12-slim\n")
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor")
    result = asyncio.run(env.seed(0))
    assert result.sandbox_spec is not None
    assert result.sandbox_spec.entrypoint is None
    # pick_lifecycle() selects StatefulSandbox iff verify_sandbox_spec is None.
    assert result.verify_sandbox_spec is None
    assert result.capture_cmd is None
    assert result.apply_cmd is None


# ---------------------------------------------------------------------------
# nmp_harbor.verify() readiness gate
# ---------------------------------------------------------------------------


class _FakeExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _RecordingSandbox:
    """Minimal stand-in for ``Sandbox`` that records exec calls."""

    def __init__(self, *, wait_rc: int = 0, wait_stderr: str = "") -> None:
        self._wait_rc = wait_rc
        self._wait_stderr = wait_stderr
        self.calls: list[tuple[str, int | None]] = []

    async def exec(self, cmd: str, timeout_sec: int | None = None):
        self.calls.append((cmd, timeout_sec))
        if "curl" in cmd and "localhost:8080/health" in cmd:
            return _FakeExecResult(return_code=self._wait_rc, stderr=self._wait_stderr)
        return _FakeExecResult(return_code=0)


def test_nmp_harbor_verify_runs_wait_before_super(tmp_path: Path, monkeypatch):
    """``verify()`` must exec the readiness probe before delegating."""
    import asyncio

    _make_fake_full_task(tmp_path)
    (tmp_path / "Dockerfile.harbor").write_text("FROM python:3.12-slim\n")
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor")
    sandbox = _RecordingSandbox(wait_rc=0)

    # Short-circuit super().verify() so we don't actually run tests.
    from nemo_evaluator.environments.harbor import HarborEnvironment

    async def _fake_super_verify(self, response, expected, sandbox=None, **md):
        # Assert the wait-probe already ran exactly once.
        assert sandbox is not None
        assert any("localhost:8080/health" in c for c, _ in sandbox.calls), (
            "expected wait_api_cmd to be exec'd before super().verify()"
        )
        return VerifyResult(reward=0.5, scoring_details={"method": "stub"})

    monkeypatch.setattr(HarborEnvironment, "verify", _fake_super_verify)

    result = asyncio.run(env.verify("ok", "", sandbox=sandbox, task_dir=str(tmp_path)))
    assert result.reward == 0.5
    # The probe should have been called exactly once.
    probes = [c for c, _ in sandbox.calls if "localhost:8080/health" in c]
    assert len(probes) == 1


def test_nmp_harbor_verify_returns_zero_on_probe_failure(tmp_path: Path, monkeypatch):
    """If the API never comes up, ``verify()`` short-circuits to reward=0."""
    import asyncio

    _make_fake_full_task(tmp_path)
    (tmp_path / "Dockerfile.harbor").write_text("FROM python:3.12-slim\n")
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor")
    sandbox = _RecordingSandbox(wait_rc=1, wait_stderr="nmp_harbor: NMP API failed")

    from nemo_evaluator.environments.harbor import HarborEnvironment

    async def _never_called(self, *a, **kw):
        raise AssertionError("super().verify() must not run when API probe fails")

    monkeypatch.setattr(HarborEnvironment, "verify", _never_called)

    result = asyncio.run(env.verify("ok", "", sandbox=sandbox, task_dir=str(tmp_path)))
    assert result.reward == 0.0
    assert result.scoring_details.get("error") == "nmp_api_not_ready"
    assert "NMP API failed" in (result.scoring_details.get("wait_stderr") or "")


def test_nmp_harbor_wait_api_cmd_override(tmp_path: Path, monkeypatch):
    """Users can replace the readiness probe via ``params.wait_api_cmd``."""

    _make_fake_full_task(tmp_path)
    (tmp_path / "Dockerfile.harbor").write_text("FROM python:3.12-slim\n")
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor", params={"wait_api_cmd": "echo ready"})
    assert env._wait_api_cmd == "echo ready"


def test_nmp_harbor_empty_wait_api_cmd_skips_probe(tmp_path: Path, monkeypatch):
    """``wait_api_cmd=""`` disables the probe (and must not exec it)."""
    import asyncio

    _make_fake_full_task(tmp_path)
    (tmp_path / "Dockerfile.harbor").write_text("FROM python:3.12-slim\n")
    monkeypatch.setenv("NMP_REPO", str(tmp_path))

    env = get_environment("nmp_harbor", params={"wait_api_cmd": ""})
    sandbox = _RecordingSandbox()

    from nemo_evaluator.environments.harbor import HarborEnvironment

    async def _fake_super_verify(self, response, expected, sandbox=None, **md):
        return VerifyResult(reward=1.0, scoring_details={"method": "stub"})

    monkeypatch.setattr(HarborEnvironment, "verify", _fake_super_verify)

    asyncio.run(env.verify("ok", "", sandbox=sandbox, task_dir=str(tmp_path)))
    assert not any("localhost:8080/health" in c for c, _ in sandbox.calls)
