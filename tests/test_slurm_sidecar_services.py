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

from nemo_evaluator.config import EvalConfig
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


class TestSidecarGymServices:
    """Non-model services (gym) referenced by solvers must appear in sidecar configs."""

    def test_gym_delegation_includes_gym_service(self):
        """gym_service referenced by gym_delegation solver appears in sidecar."""
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
        sidecar = list(sidecars.values())[0]
        assert "gym" in sidecar["services"]
        assert sidecar["services"]["gym"]["type"] == "gym"
        assert sidecar["services"]["gym"]["url"] == "http://gym-server:8000"

    def test_tool_calling_includes_resource_service(self):
        """resource_service referenced by tool_calling solver appears in sidecar."""
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
        sidecar = list(sidecars.values())[0]
        assert "tools" in sidecar["services"]
        assert sidecar["services"]["tools"]["type"] == "gym"
        assert sidecar["services"]["tools"]["url"] == "http://localhost:18099"

    def test_managed_gym_gets_localhost_url(self):
        """Managed gym (no url set) gets http://localhost:<port> in sidecar."""
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
        sidecar = list(sidecars.values())[0]
        assert sidecar["services"]["gym"]["url"] == "http://localhost:9090"

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
        sidecar = list(sidecars.values())[0]
        assert "unused_gym" not in sidecar["services"]
        assert list(sidecar["services"].keys()) == ["model"]
