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
