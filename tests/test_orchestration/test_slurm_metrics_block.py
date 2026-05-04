# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Prometheus /metrics scraper spawn in the sbatch template.

The scraper line is emitted by ``_metrics_block`` after the model service
health-check passes. Verifies the spawn is correct for vLLM/sglang
services, omitted for ExternalApiService, and skipped for het-group
services on non-batch-node pools (analogous to the existing
``_health_block`` rules).
"""

from __future__ import annotations

import pytest

from nemo_evaluator.config.services import (
    ExternalApiService,
    SglangService,
    VllmService,
)
from nemo_evaluator.orchestration.slurm_gen import _metrics_block


_IMAGE = "gitlab-master.nvidia.com/dl/joc/example:dev-test"


@pytest.mark.parametrize(
    "service_cls,port",
    [(VllmService, 8000), (SglangService, 30000)],
    ids=["vllm", "sglang"],
)
def test_metrics_block_emits_for_managed_service(service_cls, port) -> None:
    svc = service_cls(model="m", port=port, protocol="chat_completions")
    block = _metrics_block("svc", svc, eval_image=_IMAGE)
    assert "scrape_metrics" in block
    assert f'"http://localhost:{port}/metrics"' in block
    assert '"$OUTPUT_DIR/svc_engine_metrics.jsonl"' in block
    assert f"--container-image {_IMAGE}" in block
    assert "srun" in block
    assert block.rstrip().endswith("&")


def test_metrics_block_propagates_kill_switch_env() -> None:
    """The NEL_TRACING_METRICS_DISABLED kill-switch lives in scrape_metrics.py;
    Slurm containers don't inherit env vars unless explicitly passed via
    --container-env, so the spawn line must include it.
    """
    svc = VllmService(model="m", port=8000, protocol="chat_completions")
    block = _metrics_block("svc", svc, eval_image=_IMAGE)
    assert "--container-env=NEL_TRACING_METRICS_DISABLED" in block


def test_metrics_block_pins_to_master_ip_for_multi_node() -> None:
    """For multi-node services the api server binds on the head node only,
    so the scraper srun must pin to ``$MASTER_IP``. Without the pin, srun
    is free to land on a worker and the localhost URL would refuse-connect.
    """
    svc = VllmService(model="m", port=8000, protocol="chat_completions", num_nodes=4)
    block = _metrics_block("svc", svc, eval_image=_IMAGE)
    assert "-w $MASTER_IP" in block


def test_metrics_block_omits_master_ip_pin_for_single_node() -> None:
    """Single-node services don't export ``MASTER_IP`` (only the multi-node
    Ray preamble does). A bare ``-w $MASTER_IP`` would abort the sbatch
    under ``set -u``. For single-node configs srun lands on the only
    allocated node anyway, so the pin is unnecessary.

    Caught by SLURM 2554268 (line 136 unbound variable abort).
    """
    svc = VllmService(model="m", port=8000, protocol="chat_completions")
    block = _metrics_block("svc", svc, eval_image=_IMAGE)
    assert "-w $MASTER_IP" not in block, (
        "single-node block must not reference $MASTER_IP (set -u would abort the sbatch)"
    )


def test_metrics_block_omits_for_external_api() -> None:
    svc = ExternalApiService(url="http://api.example.com/v1/chat/completions", protocol="chat_completions")
    block = _metrics_block("api", svc, eval_image=_IMAGE)
    assert block == ""


@pytest.mark.parametrize(
    "node_pool,pool_to_het,expect_emit",
    [
        ("gpu", {"gpu": 1}, False),
        ("batch", {"batch": 0}, True),
        ("gpu", None, True),
    ],
    ids=["het_non_batch_pool_skips", "het_batch_pool_emits", "single_pool_emits"],
)
def test_metrics_block_pool_routing(node_pool, pool_to_het, expect_emit) -> None:
    svc = VllmService(model="m", port=8000, protocol="chat_completions", node_pool=node_pool)
    block = _metrics_block("svc", svc, eval_image=_IMAGE, pool_to_het=pool_to_het)
    if expect_emit:
        assert "scrape_metrics" in block
    else:
        assert block == ""


def test_metrics_block_safe_name_for_dotted_service_name() -> None:
    svc = VllmService(model="m", port=8000, protocol="chat_completions")
    block = _metrics_block("model-1.2", svc, eval_image=_IMAGE)
    assert "model_1_2_engine_metrics.jsonl" in block


def test_metrics_block_omits_when_no_eval_image() -> None:
    """Without an eval_image we can't run the scraper inside a container; rather
    than emit a broken bare-batch-node spawn (which would fail with
    ModuleNotFoundError on ``nemo_evaluator``), we skip cleanly.
    """
    svc = VllmService(model="m", port=8000, protocol="chat_completions")
    block = _metrics_block("svc", svc, eval_image=None)
    assert block == ""
