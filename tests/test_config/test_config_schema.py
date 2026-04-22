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
"""Tests for eval config schema validation and env var expansion."""

import pytest
import warnings

from nemo_evaluator.config import (
    EvalConfig,
    ExternalApiService,
    VllmService,
    BenchmarkConfig,
    SimpleSolver,
    HarborSolverConfig,
    DockerSandbox,
    ScoringConfig,
    JudgeMetric,
    SlurmCluster,
    LocalCluster,
    GenerationConfig,
    parse_eval_config,
)
from nemo_evaluator.config.eval_config import _expand_env


class TestEnvExpansion:
    def test_simple_var(self, monkeypatch):
        monkeypatch.setenv("TEST_URL", "http://example.com")
        assert _expand_env("${TEST_URL}") == "http://example.com"

    def test_default_value(self):
        result = _expand_env("${NONEXISTENT_VAR:-fallback}")
        assert result == "fallback"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("MY_MODEL", "gpt-4")
        result = _expand_env({"model": "${MY_MODEL}", "url": "${MISSING:-default}"})
        assert result == {"model": "gpt-4", "url": "default"}

    def test_list_expansion(self, monkeypatch):
        monkeypatch.setenv("BENCH", "gsm8k")
        result = _expand_env(["${BENCH}", "triviaqa"])
        assert result == ["gsm8k", "triviaqa"]

    def test_non_string_passthrough(self):
        assert _expand_env(42) == 42
        assert _expand_env(3.14) == 3.14
        assert _expand_env(None) is None

    def test_strict_raises_on_unset(self):
        with pytest.raises(ValueError, match="not set"):
            _expand_env("${TOTALLY_UNDEFINED_VAR}")

    def test_empty_default(self):
        result = _expand_env("${NONEXISTENT:-}")
        assert result == ""

    def test_escaped_env_var_preserved(self):
        assert _expand_env("$${RUNTIME_VAR}/v1") == "${RUNTIME_VAR}/v1"

    def test_escaped_mixed_with_expanded(self, monkeypatch):
        monkeypatch.setenv("HOST_VAR", "http://localhost")
        result = _expand_env("${HOST_VAR}/$${RUNTIME_VAR}")
        assert result == "http://localhost/${RUNTIME_VAR}"

    def test_escaped_with_default_syntax(self):
        assert _expand_env("$${VAR:-default}") == "${VAR:-default}"

    def test_double_dollar_no_brace_passthrough(self):
        assert _expand_env("cost: $$100") == "cost: $$100"


class TestParseEvalConfig:
    def test_minimal_api_config(self):
        raw = {
            "services": {
                "solver": {
                    "type": "api",
                    "url": "http://localhost:8000/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "gpt-4",
                },
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "solver"},
                },
            ],
        }
        cfg = parse_eval_config(raw)
        assert len(cfg.benchmarks) == 1
        assert cfg.benchmarks[0].name == "gsm8k"
        assert isinstance(cfg.benchmarks[0].solver, SimpleSolver)
        assert cfg.benchmarks[0].solver.service == "solver"

    def test_vllm_service(self):
        raw = {
            "services": {
                "model": {
                    "type": "vllm",
                    "model": "Qwen/Qwen3.5-9B",
                    "protocol": "chat_completions",
                    "port": 8000,
                },
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "model"},
                },
            ],
        }
        cfg = parse_eval_config(raw)
        svc = cfg.services["model"]
        assert isinstance(svc, VllmService)
        assert svc.type == "vllm"
        assert svc.model == "Qwen/Qwen3.5-9B"

    def test_empty_benchmarks_raises(self):
        raw = {
            "services": {
                "s": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            "benchmarks": [],
        }
        with pytest.raises(Exception):
            parse_eval_config(raw)

    def test_missing_services_raises(self):
        raw = {"benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "x"}}]}
        with pytest.raises(Exception):
            parse_eval_config(raw)

    def test_cluster_defaults_to_local(self):
        raw = {
            "services": {
                "s": {
                    "type": "api",
                    "url": "http://localhost:8000/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "test",
                },
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "s"},
                },
            ],
        }
        cfg = parse_eval_config(raw)
        assert isinstance(cfg.cluster, LocalCluster)
        assert cfg.cluster.type == "local"

    def test_slurm_cluster_with_node_pools(self):
        raw = {
            "services": {
                "s": {
                    "type": "api",
                    "url": "http://localhost:8000/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "test",
                },
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "s"},
                },
            ],
            "cluster": {
                "type": "slurm",
                "walltime": "02:00:00",
                "node_pools": {
                    "gpu": {"partition": "gpu", "nodes": 1, "gres": "gpu:4"},
                },
            },
        }
        cfg = parse_eval_config(raw)
        assert isinstance(cfg.cluster, SlurmCluster)
        assert "gpu" in cfg.cluster.node_pools
        assert cfg.cluster.node_pools["gpu"].gres == "gpu:4"

    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("MODEL_URL", "http://my-server:8000/v1/chat/completions")
        monkeypatch.setenv("MODEL_ID", "llama-3")
        raw = {
            "services": {
                "s": {
                    "type": "api",
                    "url": "${MODEL_URL}",
                    "protocol": "chat_completions",
                    "model": "${MODEL_ID}",
                },
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "s"},
                },
            ],
        }
        cfg = parse_eval_config(raw)
        svc = cfg.services["s"]
        assert isinstance(svc, ExternalApiService)
        assert svc.url == "http://my-server:8000/v1/chat/completions"
        assert svc.model == "llama-3"

    def test_managed_services(self):
        cfg = EvalConfig(
            services={
                "vllm_model": VllmService(
                    type="vllm",
                    model="Qwen/Qwen3.5-9B",
                    protocol="chat_completions",
                ),
                "external": ExternalApiService(
                    type="api",
                    url="http://x/v1/chat/completions",
                    protocol="chat_completions",
                ),
            },
            benchmarks=[
                BenchmarkConfig(
                    name="gsm8k",
                    solver=SimpleSolver(type="simple", service="vllm_model"),
                ),
            ],
        )
        managed = cfg.managed_services()
        assert "vllm_model" in managed
        assert "external" not in managed

    def test_get_service(self):
        cfg = EvalConfig(
            services={
                "s": ExternalApiService(
                    type="api",
                    url="http://remote:8000/v1/chat/completions",
                    protocol="chat_completions",
                    model="test",
                ),
            },
            benchmarks=[
                BenchmarkConfig(
                    name="gsm8k",
                    solver=SimpleSolver(type="simple", service="s"),
                ),
            ],
        )
        assert cfg.get_model_url("s") == "http://remote:8000/v1/chat/completions"
        assert cfg.get_model_id("s") == "test"

    def test_reserved_service_name_raises(self):
        with pytest.raises(Exception, match="Reserved"):
            EvalConfig(
                services={
                    "default": ExternalApiService(
                        type="api",
                        url="http://x/v1/chat/completions",
                        protocol="chat_completions",
                    ),
                },
                benchmarks=[
                    BenchmarkConfig(
                        name="gsm8k",
                        solver=SimpleSolver(type="simple", service="default"),
                    ),
                ],
            )


class TestServiceValidation:
    def test_missing_service_reference(self):
        with pytest.raises(Exception, match="not in services"):
            EvalConfig(
                services={
                    "s": ExternalApiService(
                        type="api",
                        url="http://x/v1/chat/completions",
                        protocol="chat_completions",
                    ),
                },
                benchmarks=[
                    BenchmarkConfig(
                        name="gsm8k",
                        solver=SimpleSolver(type="simple", service="nonexistent"),
                    ),
                ],
            )

    def test_self_judge_raises(self):
        with pytest.raises(Exception, match="same as the solver"):
            EvalConfig(
                services={
                    "s": ExternalApiService(
                        type="api",
                        url="http://x/v1/chat/completions",
                        protocol="chat_completions",
                    ),
                },
                benchmarks=[
                    BenchmarkConfig(
                        name="simpleqa",
                        solver=SimpleSolver(type="simple", service="s"),
                        scoring=ScoringConfig(
                            metrics=[
                                JudgeMetric(
                                    type="judge",
                                    name="test",
                                    service="s",
                                    max_score=5.0,
                                ),
                            ],
                        ),
                    ),
                ],
            )

    def test_self_judge_allowed_with_flag(self):
        cfg = EvalConfig(
            services={
                "s": ExternalApiService(
                    type="api",
                    url="http://x/v1/chat/completions",
                    protocol="chat_completions",
                ),
            },
            benchmarks=[
                BenchmarkConfig(
                    name="simpleqa",
                    solver=SimpleSolver(type="simple", service="s"),
                    scoring=ScoringConfig(
                        metrics=[
                            JudgeMetric(
                                type="judge",
                                name="test",
                                service="s",
                                max_score=5.0,
                                allow_self_judge=True,
                            ),
                        ],
                    ),
                ),
            ],
        )
        assert len(cfg.benchmarks) == 1


class TestSandboxValidation:
    def test_harbor_requires_sandbox(self):
        with pytest.raises(Exception, match="requires a sandbox"):
            EvalConfig(
                services={
                    "s": ExternalApiService(
                        type="api",
                        url="http://x/v1/chat/completions",
                        protocol="chat_completions",
                    ),
                },
                benchmarks=[
                    BenchmarkConfig(
                        name="swebench",
                        solver=HarborSolverConfig(
                            type="harbor",
                            service="s",
                            agent="openhands",
                        ),
                    ),
                ],
            )

    def test_harbor_with_docker_sandbox_ok(self):
        cfg = EvalConfig(
            services={
                "s": ExternalApiService(
                    type="api",
                    url="http://x/v1/chat/completions",
                    protocol="chat_completions",
                ),
            },
            benchmarks=[
                BenchmarkConfig(
                    name="swebench",
                    solver=HarborSolverConfig(
                        type="harbor",
                        service="s",
                        agent="openhands",
                    ),
                    sandbox=DockerSandbox(type="docker"),
                ),
            ],
        )
        assert isinstance(cfg.benchmarks[0].sandbox, DockerSandbox)


class TestNamedSandboxes:
    def test_sandbox_reference(self):
        raw = {
            "services": {
                "s": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            "sandboxes": {
                "code": {"type": "docker", "image": "python:3.12"},
            },
            "benchmarks": [
                {
                    "name": "humaneval",
                    "solver": {"type": "simple", "service": "s"},
                    "sandbox": "code",
                },
            ],
        }
        cfg = parse_eval_config(raw)
        assert isinstance(cfg.benchmarks[0].sandbox, DockerSandbox)
        assert cfg.benchmarks[0].sandbox.image == "python:3.12"

    def test_bad_sandbox_reference_raises(self):
        raw = {
            "services": {
                "s": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            "sandboxes": {},
            "benchmarks": [
                {
                    "name": "humaneval",
                    "solver": {"type": "simple", "service": "s"},
                    "sandbox": "nonexistent",
                },
            ],
        }
        with pytest.raises(Exception, match="nonexistent"):
            parse_eval_config(raw)


class TestGenerationConfig:
    def test_merge_onto(self):
        base = GenerationConfig(temperature=0.0, max_tokens=2048)
        override = GenerationConfig(temperature=0.7)
        merged = override.merge_onto(base)
        assert merged.temperature == 0.7
        assert merged.max_tokens == 2048

    def test_merge_onto_all_fields(self):
        base = GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_tokens=2048,
            seed=42,
            stop=["END"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        override = GenerationConfig(temperature=0.7, seed=99, stop=["STOP", "DONE"])
        merged = override.merge_onto(base)
        assert merged.temperature == 0.7
        assert merged.top_p == 0.9
        assert merged.max_tokens == 2048
        assert merged.seed == 99
        assert merged.stop == ["STOP", "DONE"]
        assert merged.frequency_penalty == 0.5
        assert merged.presence_penalty == 0.3

    def test_merge_onto_none_inherits(self):
        base = GenerationConfig(frequency_penalty=0.5, presence_penalty=-0.5)
        override = GenerationConfig()
        merged = override.merge_onto(base)
        assert merged.frequency_penalty == 0.5
        assert merged.presence_penalty == -0.5
        assert merged.temperature is None

    def test_range_validation(self):
        with pytest.raises(Exception):
            GenerationConfig(temperature=3.0)

    def test_max_tokens_positive(self):
        with pytest.raises(Exception):
            GenerationConfig(max_tokens=0)

    def test_penalty_range_validation(self):
        with pytest.raises(Exception):
            GenerationConfig(frequency_penalty=3.0)
        with pytest.raises(Exception):
            GenerationConfig(presence_penalty=-3.0)


class TestUrlProtocolWarning:
    def test_mismatch_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExternalApiService(
                type="api",
                url="http://x/v1/chat/completions",
                protocol="completions",
            )
            assert len(w) >= 1
            assert "chat/completions" in str(w[0].message)

    def test_matching_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExternalApiService(
                type="api",
                url="http://x/v1/chat/completions",
                protocol="chat_completions",
            )
            assert len(w) == 0


# ── Sharding config tests ───────────────────────────────────────────


def _slurm_config(**cluster_overrides):
    """Helper to build a minimal shardable SLURM config."""
    cluster = {
        "type": "slurm",
        "walltime": "01:00:00",
        "node_pools": {"compute": {"partition": "batch", "nodes": 1, "gres": "gpu:1"}},
    }
    cluster.update(cluster_overrides)
    return {
        "services": {
            "model": {"type": "api", "url": "http://x/v1/chat/completions", "protocol": "chat_completions"},
        },
        "benchmarks": [{"name": "gsm8k", "solver": {"type": "simple", "service": "model"}}],
        "cluster": cluster,
    }


class TestShardsField:
    def test_shards_none_by_default(self):
        cfg = EvalConfig.model_validate(_slurm_config())
        assert cfg.cluster.shards is None

    def test_shards_accepted(self):
        cfg = EvalConfig.model_validate(_slurm_config(shards=4, auto_resume=False))
        assert cfg.cluster.shards == 4

    def test_shards_zero_rejected(self):
        with pytest.raises(Exception):
            EvalConfig.model_validate(_slurm_config(shards=0))

    def test_shards_negative_rejected(self):
        with pytest.raises(Exception):
            EvalConfig.model_validate(_slurm_config(shards=-1))


class TestShardsHetJobGuard:
    def test_shards_with_single_pool_ok(self):
        cfg = EvalConfig.model_validate(_slurm_config(shards=4, auto_resume=False))
        assert cfg.cluster.shards == 4

    def test_shards_with_multi_pool_raises(self):
        raw = {
            "services": {
                "model": {
                    "type": "vllm",
                    "model": "m",
                    "protocol": "chat_completions",
                    "port": 8000,
                    "node_pool": "gpu",
                },
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "model"},
                    "sandbox": {"type": "slurm", "image": "ubuntu:22.04", "node_pool": "cpu"},
                }
            ],
            "cluster": {
                "type": "slurm",
                "walltime": "01:00:00",
                "shards": 4,
                "auto_resume": False,
                "node_pools": {
                    "gpu": {"partition": "gpu", "nodes": 1, "gres": "gpu:4"},
                    "cpu": {"partition": "cpu", "nodes": 2},
                },
            },
        }
        with pytest.raises(ValueError, match="incompatible with heterogeneous"):
            EvalConfig.model_validate(raw)

    def test_shards_with_auto_resume_allowed(self):
        """Shards + auto_resume is valid: each shard is an independent job with its own resume chain."""
        cfg = EvalConfig.model_validate(_slurm_config(shards=4, auto_resume=True))
        assert cfg.cluster.shards == 4
        assert cfg.cluster.auto_resume is True


class TestApplyOverrideNull:
    """Test that -O key=null coerces to None for key deletion."""

    def test_null_override_sets_none(self):
        from nemo_evaluator.cli.eval import _apply_override

        data = {"output": {"dir": "/tmp", "extra": "remove-me"}}
        _apply_override(data, "output.extra=null")
        assert data["output"]["extra"] is None

    def test_null_case_variants(self):
        from nemo_evaluator.cli.eval import _apply_override

        for null_str in ("null", "Null", "NULL", "~"):
            data = {"key": "value"}
            _apply_override(data, f"key={null_str}")
            assert data["key"] is None, f"Failed for {null_str!r}"

    def test_none_string_not_coerced(self):
        from nemo_evaluator.cli.eval import _apply_override

        data = {"key": "value"}
        _apply_override(data, "key=none")
        assert data["key"] == "none"


# ---------------------------------------------------------------------------
# stateful sandbox flag
# ---------------------------------------------------------------------------


class TestStatefulSandboxFlag:
    def test_stateful_field_accepted_in_ecs_fargate(self):
        """EcsFargateSandbox parses stateful: true without error."""
        from nemo_evaluator.config import EcsFargateSandbox

        sb = EcsFargateSandbox(
            type="ecs_fargate",
            region="us-east-1",
            ecr_repository="123.dkr.ecr.us-east-1.amazonaws.com/repo",
            stateful=True,
        )
        assert sb.stateful is True

    def test_stateful_defaults_to_false(self):
        """stateful defaults to False for backward compatibility."""
        from nemo_evaluator.config import EcsFargateSandbox

        sb = EcsFargateSandbox(
            type="ecs_fargate",
            region="us-east-1",
            ecr_repository="123.dkr.ecr.us-east-1.amazonaws.com/repo",
        )
        assert sb.stateful is False

    def test_stateful_with_capture_cmd_warns(self):
        """Setting stateful=True alongside capture_cmd emits a UserWarning."""
        from nemo_evaluator.config import EcsFargateSandbox

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EcsFargateSandbox(
                type="ecs_fargate",
                region="us-east-1",
                ecr_repository="123.dkr.ecr.us-east-1.amazonaws.com/repo",
                stateful=True,
                capture_cmd="echo capture",
            )
        assert any("stateful" in str(w.message).lower() and issubclass(w.category, UserWarning) for w in caught)


class TestTerminalBenchPlaybook:
    def test_terminal_bench_playbook_loads(self):
        """terminal_bench_2 playbook parses cleanly and has stateful=True."""
        config = parse_eval_config(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "https://example.com/v1",
                        "protocol": "chat_completions",
                        "model": "test-model",
                        "api_key": "test-key",
                    },
                },
                "benchmarks": [
                    {
                        "playbook": "terminal_bench_2",
                        "sandbox": {
                            "region": "us-east-1",
                            "ecr_repository": "123.dkr.ecr.us-east-1.amazonaws.com/repo",
                        },
                    }
                ],
            }
        )
        bm = config.benchmarks[0]
        assert bm.name == "harbor://terminal-bench@2.0"
        assert bm.sandbox.stateful is True

    def test_terminal_bench_playbook_stateful_can_be_overridden(self):
        """User can override stateful: false to revert to stateless."""
        config = parse_eval_config(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "https://example.com/v1",
                        "protocol": "chat_completions",
                        "model": "test-model",
                        "api_key": "test-key",
                    },
                },
                "benchmarks": [
                    {
                        "playbook": "terminal_bench_2",
                        "sandbox": {
                            "region": "us-east-1",
                            "ecr_repository": "123.dkr.ecr.us-east-1.amazonaws.com/repo",
                            "stateful": False,
                        },
                    }
                ],
            }
        )
        assert config.benchmarks[0].sandbox.stateful is False


class TestNoSandboxStateful:
    def test_pick_lifecycle_with_nosandbox_config(self):
        """pick_lifecycle must work when sandbox_cfg is a NoSandbox (BYOB default).

        eval_loop.py:259-265 builds pick_lifecycle kwargs from sandbox_cfg. When
        no sandbox is configured, BenchmarkConfig defaults to NoSandbox(). This
        crashed with AttributeError because NoSandbox lacked the stateful field.
        """
        from unittest.mock import MagicMock

        from nemo_evaluator.config.sandboxes import NoSandbox
        from nemo_evaluator.sandbox.strategies import pick_lifecycle

        sandbox_cfg = NoSandbox()
        seed = MagicMock()
        seed.sandbox_spec = None
        seed.verify_sandbox_spec = None

        # This mirrors eval_loop.py:259-265 exactly
        pick_lifecycle(
            seed,
            None,
            config_capture_cmd=sandbox_cfg.capture_cmd,
            verify_timeout=sandbox_cfg.verify_timeout,
            force_stateful=sandbox_cfg.stateful,
        )


class TestHarborSolverTimeoutConfig:
    """Tests for timeout_strategy and max_agent_timeout config fields."""

    def test_timeout_strategy_defaults_to_override(self):
        from nemo_evaluator.config.solvers import HarborSolverConfig

        cfg = HarborSolverConfig(type="harbor", service="model", agent="terminus-2")
        assert cfg.timeout_strategy == "override"
        assert cfg.max_agent_timeout is None

    def test_timeout_strategy_task_parses(self):
        from nemo_evaluator.config.solvers import HarborSolverConfig

        cfg = HarborSolverConfig(
            type="harbor",
            service="model",
            agent="terminus-2",
            timeout_strategy="task",
            max_agent_timeout=7200,
        )
        assert cfg.timeout_strategy == "task"
        assert cfg.max_agent_timeout == 7200

    def test_timeout_strategy_max_parses(self):
        from nemo_evaluator.config.solvers import HarborSolverConfig

        cfg = HarborSolverConfig(
            type="harbor",
            service="model",
            agent="terminus-2",
            timeout_strategy="max",
            max_agent_timeout=3600,
        )
        assert cfg.timeout_strategy == "max"
        assert cfg.max_agent_timeout == 3600

    def test_timeout_strategy_invalid_rejected(self):
        import pydantic
        from nemo_evaluator.config.solvers import HarborSolverConfig

        with pytest.raises(pydantic.ValidationError, match="timeout_strategy"):
            HarborSolverConfig(
                type="harbor",
                service="model",
                agent="terminus-2",
                timeout_strategy="foo",
            )
