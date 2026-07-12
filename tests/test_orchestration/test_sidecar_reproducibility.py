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
"""Sidecar reproducibility: literal model id, proxy defaults, header."""

import pytest

from nemo_evaluator.config import EvalConfig
from nemo_evaluator.orchestration.slurm_gen import generate_sbatch, write_sbatch


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


_BENCH = [
    {
        "name": "swebench-multilingual",
        "solver": {
            "type": "gym_delegation",
            "service": "model",
            "gym_service": "gym",
            "gym_agent": "openhands",
        },
    }
]

_GYM = {"type": "gym", "port": 9090}


def _vllm(**extra):
    return {
        "type": "vllm",
        "model": "/ckpt/some-checkpoint",
        "protocol": "chat_completions",
        "tensor_parallel_size": 4,
        "port": 8000,
        "node_pool": "gpu",
        **extra,
    }


_EXTERNAL_API = {
    "type": "api",
    "url": "https://integrate.api.nvidia.com/v1/chat/completions",
    "protocol": "chat_completions",
    "model": "nvidia/nemotron-3-ultra-550b-a55b",
}


class TestSidecarModelLiteral:
    @pytest.mark.parametrize(
        ("model_service", "expected_literal"),
        [
            pytest.param(
                _vllm(served_model_name="nvidia/nemotron-test"),
                "nvidia/nemotron-test",
                id="served_name_inlined",
            ),
            pytest.param(_vllm(), None, id="env_ref_kept"),
            pytest.param(_EXTERNAL_API, "nvidia/nemotron-3-ultra-550b-a55b", id="external_api_inlined"),
        ],
    )
    def test_model_id_in_sidecar(self, model_service, expected_literal):
        cfg = _make_slurm_config(
            services={"model": model_service, "gym": _GYM},
            benchmarks=_BENCH,
        )
        _, sidecars, _ = generate_sbatch(cfg)
        svc = list(sidecars.values())[0]["services"]["model"]
        if expected_literal is None:
            assert svc["model"].startswith("${")
        else:
            assert svc["model"] == expected_literal
        if model_service.get("type") != "api":
            # URL is runtime-assigned for deployments: must stay an env ref.
            assert svc["url"].startswith("${")


class TestSidecarProxyDefaults:
    def test_proxy_defaults_recorded(self):
        """exclude_defaults must NOT strip effective proxy settings."""
        cfg = _make_slurm_config(
            services={
                "model": _vllm(proxy={"interceptors": [{"name": "reasoning"}]}),
                "gym": _GYM,
            },
            benchmarks=_BENCH,
        )
        _, sidecars, _ = generate_sbatch(cfg)
        proxy = list(sidecars.values())[0]["services"]["model"]["proxy"]
        # Default-valued fields must be present so the sidecar records the
        # effective configuration (request_timeout=120.0, max_retries=0, ...).
        assert "request_timeout" in proxy
        assert "max_retries" in proxy
        assert proxy["interceptors"][0]["name"] == "reasoning"


class TestSidecarHeader:
    def test_written_sidecar_has_provenance_header(self, tmp_path):
        cfg = _make_slurm_config(
            services={"model": _vllm(served_model_name="m"), "gym": _GYM},
            benchmarks=_BENCH,
        )
        write_sbatch(cfg, output_dir=tmp_path)
        sidecar_files = list(tmp_path.glob("config_*.yaml"))
        assert sidecar_files, "no sidecar config written"
        for sidecar in sidecar_files:
            text = sidecar.read_text()
            assert "full_config.yaml" in text, sidecar
