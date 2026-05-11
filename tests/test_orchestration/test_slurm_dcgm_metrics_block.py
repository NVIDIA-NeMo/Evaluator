# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-node GPU /metrics scraper spawn in the sbatch template.

The block is emitted by ``_dcgm_metrics_block`` once per sbatch (not per
service) right after the per-service ``_metrics_block`` loop. It iterates
over every node in ``$SLURM_JOB_NODELIST`` and runs the !109
``scrape_metrics`` CLI against ``localhost:9400`` (the standard
``dcgm-exporter`` port — assumed to be a system service on the cluster).
Output is one ``$OUTPUT_DIR/gpu_metrics_<host>.jsonl`` per node. Sibling
to ``test_slurm_metrics_block.py``.
"""

from __future__ import annotations

from nemo_evaluator.orchestration.slurm_gen import (
    _DCGM_EXPORTER_PORT,
    _dcgm_metrics_block,
)

_IMAGE = "registry.example.com/example:dev-test"


def test_dcgm_block_emits_when_eval_image_present() -> None:
    block = _dcgm_metrics_block(eval_image=_IMAGE)
    assert "scrape_metrics" in block
    assert f'"http://localhost:{_DCGM_EXPORTER_PORT}/metrics"' in block
    assert 'scontrol show hostnames "$SLURM_JOB_NODELIST"' in block
    assert '--nodelist="$_NODE"' in block
    assert "gpu_metrics_${_NODE}.jsonl" in block
    assert f"--container-image {_IMAGE}" in block
    assert block.rstrip().endswith("fi")


def test_dcgm_block_does_not_spawn_dcgm_exporter() -> None:
    """The cluster's system ``dcgm-exporter`` already binds :9400; spawning
    our own collides (``bind: address already in use``, confirmed on HSG
    by smoke 2554268). Just scrape the existing endpoint.
    """
    block = _dcgm_metrics_block(eval_image=_IMAGE)
    assert "dcgm-exporter --address" not in block, "must NOT spawn dcgm-exporter — the system one is on :9400 already"
    assert "nvcr.io/nvidia/k8s/dcgm-exporter" not in block


def test_dcgm_block_propagates_kill_switch_envs() -> None:
    """Two layers of opt-out:

    * ``NEL_GPU_METRICS_DISABLED`` short-circuits the entire block at
      sbatch-runtime (the bash ``if`` guard). Must use ``${VAR:-}`` safe
      expansion: the sbatch template runs under ``set -u`` so a bare
      ``$NEL_GPU_METRICS_DISABLED`` aborts the whole job when unset
      (caught by smoke 2553388: line 138 unbound variable).
    * ``NEL_TRACING_METRICS_DISABLED`` is forwarded into the scraper
      container so the !109 runtime kill-switch still applies.
    """
    block = _dcgm_metrics_block(eval_image=_IMAGE)
    assert 'if [ -z "${NEL_GPU_METRICS_DISABLED:-}" ]' in block
    assert '$NEL_GPU_METRICS_DISABLED"' not in block, "must use ${VAR:-} safe expansion to avoid set -u abort"
    assert "--container-env=NEL_TRACING_METRICS_DISABLED" in block


def test_dcgm_block_omits_when_no_eval_image() -> None:
    """Without an eval_image we can't run the scraper inside a container;
    skip cleanly. Mirrors the same rule in ``_metrics_block``.
    """
    block = _dcgm_metrics_block(eval_image=None)
    assert block == ""


def test_dcgm_block_iterates_per_node_not_master() -> None:
    """vLLM ``_metrics_block`` pins to ``$MASTER_IP`` (one api server,
    head node only). System ``dcgm-exporter`` runs on every node, so we
    iterate the full ``$SLURM_JOB_NODELIST`` and don't pin.
    """
    block = _dcgm_metrics_block(eval_image=_IMAGE)
    assert "-w $MASTER_IP" not in block
    assert "for _NODE in $(scontrol show hostnames" in block


def test_dcgm_block_mounts_output_dir_into_scraper_container() -> None:
    """The scraper writes ``$OUTPUT_DIR/gpu_metrics_<host>.jsonl``; that path
    must be visible inside the scraper container.
    """
    block = _dcgm_metrics_block(eval_image=_IMAGE)
    assert '--container-mounts="$OUTPUT_DIR:$OUTPUT_DIR"' in block


def test_dcgm_block_emits_one_srun_per_loop_iteration() -> None:
    """Single srun per node (just the scraper). Backgrounded so the loop
    continues without blocking.
    """
    block = _dcgm_metrics_block(eval_image=_IMAGE)
    srun_count = block.count("srun ")
    assert srun_count == 1, f"expected 1 srun line per node, got {srun_count}"
    background_count = block.count(" &\n")
    assert background_count == 1, f"expected 1 backgrounded spawn, got {background_count}"
