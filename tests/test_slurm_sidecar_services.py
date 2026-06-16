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
"""Tests that non-model services (gym) referenced by solvers appear in SLURM sidecar configs."""

import os

import pytest

from nemo_evaluator.config import EvalConfig
from nemo_evaluator.config.eval_config import parse_eval_config
from nemo_evaluator.orchestration.slurm_gen import generate_sbatch


def _make_slurm_config(services, benchmarks):
    return EvalConfig.model_validate(
        {
            "services": services,
            "benchmarks": benchmarks,
            "cluster": {
                "type": "slurm",
                "walltime": "02:00:00",
                "node_pools": {"gpu": {"partition": "batch", "nodes": 1, "gres": "gpu:4"}},
            },
        }
    )


def _validate_sidecar(sidecar: dict) -> EvalConfig:
    """Parse a sidecar dict through the real config loader (with env-var expansion).

    Raises if EvalConfig validation fails (e.g. missing service references).
    """
    env_patch = {
        "MODEL_URL": "http://localhost:8000",
        "MODEL_MODEL": "test-model",
        "NEL_OUTPUT_DIR": "/tmp/test",
    }
    orig = {k: os.environ.get(k) for k in env_patch}
    os.environ.update(env_patch)
    try:
        return parse_eval_config(sidecar)
    finally:
        for k, v in orig.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestSidecarGymServices:
    """Non-model services (gym) referenced by solvers must appear in sidecar configs."""

    def test_gym_delegation_includes_gym_service(self):
        """gym_service referenced by gym_delegation solver validates in sidecar."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "vllm",
                    "model": "nvidia/test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "port": 8000,
                    "node_pool": "gpu",
                },
                "gym": {"type": "gym", "url": "http://gym-server:8000"},
            },
            benchmarks=[
                {
                    "name": "swebench-multilingual",
                    "solver": {
                        "type": "gym_delegation",
                        "service": "model",
                        "gym_service": "gym",
                        "gym_agent": "openhands",
                    },
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        _validate_sidecar(list(sidecars.values())[0])

    def test_tool_calling_includes_resource_service(self):
        """resource_service referenced by tool_calling solver validates in sidecar."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "vllm",
                    "model": "nvidia/test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "port": 8000,
                    "node_pool": "gpu",
                },
                "tools": {"type": "gym", "port": 18099},
            },
            benchmarks=[
                {
                    "name": "finance-agent",
                    "solver": {
                        "type": "tool_calling",
                        "service": "model",
                        "resource_service": "tools",
                    },
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        _validate_sidecar(list(sidecars.values())[0])

    def test_managed_gym_gets_localhost_url(self):
        """Managed gym (no url set) gets localhost URL in sidecar."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "vllm",
                    "model": "nvidia/test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "port": 8000,
                    "node_pool": "gpu",
                },
                "gym": {"type": "gym", "port": 9090},
            },
            benchmarks=[
                {
                    "name": "swebench-multilingual",
                    "solver": {
                        "type": "gym_delegation",
                        "service": "model",
                        "gym_service": "gym",
                        "gym_agent": "openhands",
                    },
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        sidecar = _validate_sidecar(list(sidecars.values())[0])
        assert sidecar.services["gym"].url == "http://localhost:9090"

    def test_unreferenced_gym_excluded_from_sidecar(self):
        """Gym service not referenced by solver must not appear in sidecar."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
                "unused_gym": {"type": "gym", "url": "http://gym:8000"},
            },
            benchmarks=[
                {
                    "name": "harbor://swebench-verified@1.0",
                    "solver": {"type": "harbor", "service": "model", "agent": "openhands-sdk"},
                    "sandbox": {"type": "ecs_fargate", "region": "us-west-2"},
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        sidecar = _validate_sidecar(list(sidecars.values())[0])
        assert "unused_gym" not in sidecar.services

    def test_missing_gym_service_fails_validation(self):
        """A sidecar missing a referenced gym service fails EvalConfig validation."""
        sidecar_missing_gym = {
            "services": {
                "model": {
                    "type": "api",
                    "url": "http://localhost:8000/chat/completions",
                    "protocol": "chat_completions",
                    "model": "test-model",
                },
            },
            "benchmarks": [
                {
                    "name": "swebench-multilingual",
                    "solver": {
                        "type": "gym_delegation",
                        "service": "model",
                        "gym_service": "gym",
                        "gym_agent": "openhands",
                    },
                }
            ],
        }
        with pytest.raises(ValueError, match="gym_service='gym' not in services"):
            _validate_sidecar(sidecar_missing_gym)


class TestSimpleSolverSidecar:
    """SimpleSolver benchmarks now always get a sidecar YAML (no quick-mode)."""

    def test_simple_solver_round_trip(self):
        """All SimpleSolver fields survive the sidecar serialize / parse cycle."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            benchmarks=[
                {
                    "name": "gsm8k",
                    "solver": {
                        "type": "simple",
                        "service": "model",
                        "system_prompt": "You are a math tutor.",
                        "image_detail": "high",
                        "generation": {"temperature": 0.7, "max_tokens": 512},
                    },
                    "repeats": 3,
                    "max_problems": 50,
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        sidecar = list(sidecars.values())[0]
        parsed = _validate_sidecar(sidecar)

        bench = parsed.benchmarks[0]
        assert bench.solver.type == "simple"
        assert bench.solver.system_prompt == "You are a math tutor."
        assert bench.solver.image_detail == "high"
        assert bench.solver.generation.temperature == 0.7
        assert bench.solver.generation.max_tokens == 512
        assert bench.repeats == 3
        assert bench.max_problems == 50

    def test_simple_solver_minimal(self):
        """SimpleSolver with only required fields produces a valid sidecar."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            benchmarks=[
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "model"},
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        parsed = _validate_sidecar(list(sidecars.values())[0])
        assert parsed.benchmarks[0].solver.type == "simple"
        assert parsed.benchmarks[0].solver.image_detail == "auto"


class TestSidecarScoringServices:
    """Services referenced from scoring.metrics must appear in sidecar configs.

    Regression: PinchBench and other llm_judge / hybrid benchmarks silently
    lost their judge service in SLURM runs because ``_extract_bench_config``
    (a) stripped ``bench.scoring`` and (b) only copied solver-facing
    services.  Every judge-scored task then short-circuited with
    ``reward=0.0`` because ``judge_client`` was None.
    """

    def test_judge_service_and_scoring_survive_sidecar(self):
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "vllm",
                    "model": "nvidia/test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "port": 8000,
                    "node_pool": "gpu",
                },
                "judge": {
                    "type": "api",
                    "url": "https://inference-api.nvidia.com/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "aws/anthropic/claude-opus-4-5",
                    "api_key": "sk-test",
                    "generation": {"temperature": 0.0},
                    "proxy": {"extra_body": {"reasoning_effort": "low"}},
                },
            },
            benchmarks=[
                {
                    "name": "pinchbench",
                    "solver": {
                        "type": "openclaw",
                        "service": "model",
                    },
                    "sandbox": {"type": "ecs_fargate", "region": "us-west-2"},
                    "scoring": {
                        "metrics": [
                            {"type": "judge", "name": "pinchbench_rubric", "service": "judge"},
                        ]
                    },
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        sidecar_dict = list(sidecars.values())[0]

        assert "judge" in sidecar_dict["services"], sidecar_dict["services"]
        judge_svc = sidecar_dict["services"]["judge"]
        assert judge_svc["url"].startswith("https://inference-api.nvidia.com")
        assert judge_svc["model"] == "aws/anthropic/claude-opus-4-5"
        assert judge_svc["api_key"] == "sk-test"
        assert judge_svc["proxy"]["extra_body"] == {"reasoning_effort": "low"}

        bench_dict = sidecar_dict["benchmarks"][0]
        assert "scoring" in bench_dict, bench_dict
        assert bench_dict["scoring"]["metrics"][0]["service"] == "judge"

        parsed = _validate_sidecar(sidecar_dict)
        assert "judge" in parsed.services
        assert parsed.benchmarks[0].scoring.metrics[0].service == "judge"

    def test_unreferenced_judge_excluded_from_sidecar(self):
        """A judge service declared but not referenced from scoring stays out."""
        cfg = _make_slurm_config(
            services={
                "model": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
                "unused_judge": {
                    "type": "api",
                    "url": "https://judge/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "judge-model",
                },
            },
            benchmarks=[
                {
                    "name": "simpleqa",
                    "solver": {"type": "simple", "service": "model"},
                }
            ],
        )
        _, sidecars, _ = generate_sbatch(cfg)
        sidecar = _validate_sidecar(list(sidecars.values())[0])
        assert "unused_judge" not in sidecar.services


class TestSidecarSecretFilePermissions:
    """Sidecar YAMLs that embed resolved ``api_key`` values must be 0600 on disk."""

    def _cfg_with_judge(self, api_key: str | None):
        judge_svc = {
            "type": "api",
            "url": "https://judge/v1/chat/completions",
            "protocol": "chat_completions",
            "model": "claude-opus-4-5",
        }
        if api_key is not None:
            judge_svc["api_key"] = api_key
        return _make_slurm_config(
            services={
                "model": {
                    "type": "vllm",
                    "model": "nvidia/test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "port": 8000,
                    "node_pool": "gpu",
                },
                "judge": judge_svc,
            },
            benchmarks=[
                {
                    "name": "pinchbench",
                    "solver": {"type": "openclaw", "service": "model"},
                    "sandbox": {"type": "ecs_fargate", "region": "us-west-2"},
                    "scoring": {
                        "metrics": [
                            {"type": "judge", "name": "rubric", "service": "judge"},
                        ]
                    },
                }
            ],
        )

    def test_sidecar_with_api_key_is_0600(self, tmp_path):
        from nemo_evaluator.orchestration.slurm_gen import write_sbatch

        cfg = self._cfg_with_judge(api_key="sk-secret")
        _, extras = write_sbatch(cfg, output_dir=tmp_path)

        config_yamls = [p for p in extras if p.name.startswith("config_") and p.suffix == ".yaml"]
        assert config_yamls, f"expected a config_*.yaml in extras, got {extras}"
        for cfg_path in config_yamls:
            mode = cfg_path.stat().st_mode & 0o777
            assert mode == 0o600, f"{cfg_path} has mode {oct(mode)}, expected 0o600"
            assert "sk-secret" in cfg_path.read_text()

    def test_sidecar_without_secrets_keeps_default_perms(self, tmp_path):
        from nemo_evaluator.orchestration.slurm_gen import write_sbatch

        cfg = self._cfg_with_judge(api_key=None)
        _, extras = write_sbatch(cfg, output_dir=tmp_path)

        config_yamls = [p for p in extras if p.name.startswith("config_") and p.suffix == ".yaml"]
        assert config_yamls
        for cfg_path in config_yamls:
            mode = cfg_path.stat().st_mode & 0o777
            assert mode != 0o600, f"{cfg_path} unexpectedly chmoded to 0600 without secret"

    def test_helper_detects_api_key_in_any_service(self):
        from nemo_evaluator.orchestration.slurm_gen import _sidecar_contains_secret

        assert _sidecar_contains_secret({"services": {"a": {"api_key": "x"}}})
        assert not _sidecar_contains_secret({"services": {"a": {"api_key": ""}}})
        assert not _sidecar_contains_secret({"services": {"a": {}}})
        assert not _sidecar_contains_secret({"services": {}})
        assert not _sidecar_contains_secret({})
