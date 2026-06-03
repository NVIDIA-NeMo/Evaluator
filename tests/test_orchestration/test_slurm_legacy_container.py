# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for slurm_gen's legacy `container://` direct-dispatch path.

These tests cover the v1 BC bridge generation: when a benchmark uses
``ContainerSolverConfig`` (``container://image#task``), slurm_gen pre-renders
``run_config.yaml`` at submit time and emits a single ``srun
--container-image=<harness>`` step — no nested Pyxis, no ``eval_image`` wrap.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from nemo_evaluator.config import EvalConfig
from nemo_evaluator.orchestration.slurm_gen import (
    _render_legacy_run_config,
    generate_sbatch,
    write_sbatch,
)


def _legacy_config(
    output_dir: str,
    *,
    with_eval_image: bool = False,
    with_native_too: bool = False,
    shards: int | None = None,
) -> EvalConfig:
    """Minimal valid config with one container:// benchmark."""
    cluster: dict = {
        "type": "slurm",
        "account": "test-account",
        "walltime": "00:30:00",
        "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
    }
    if with_eval_image:
        cluster["eval_image"] = "nvcr.io/nemo-evaluator:test"
    if shards is not None:
        cluster["shards"] = shards

    benchmarks: list[dict] = [
        {
            "name": "container://gitlab.example.com/ns/lm-evaluation-harness:25.11#ifeval",
            "params": {"limit_samples": 3, "parallelism": 1, "request_timeout": 300},
            "solver": {
                "type": "container",
                "service": "nemotron",
                "env_vars": {"HF_TOKEN": "hf_xxx", "OPENAI_API_KEY": "sk-xxx"},
            },
        }
    ]
    if with_native_too:
        benchmarks.append(
            {
                "name": "skills://gsm8k",
                "solver": {"type": "simple", "service": "nemotron"},
            }
        )

    return EvalConfig.model_validate(
        {
            "services": {
                "nemotron": {
                    "type": "api",
                    "url": "https://api.example.com/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "nemotron",
                    "api_key": "secret-token",
                }
            },
            "benchmarks": benchmarks,
            "cluster": cluster,
            "output": {"dir": output_dir},
        }
    )


def test_legacy_emits_direct_harness_srun_no_nested_wrap(tmp_path: Path):
    """Container:// benchmark renders one direct srun for the harness, no eval_image wrap."""
    cfg = _legacy_config(str(tmp_path), with_eval_image=True)
    script, _sidecars, _secrets = generate_sbatch(cfg)

    # The harness image appears in a srun --container-image step.
    assert "gitlab.example.com/ns/lm-evaluation-harness:25.11" in script
    assert "--container-image gitlab.example.com/ns/lm-evaluation-harness:25.11" in script
    # The eval-factory entrypoint (with nemo-evaluator fallback) is invoked
    # directly inside that srun, not via `nel eval run`.
    assert "eval-factory" in script
    assert "nemo-evaluator" in script
    # No `nel eval run` command targets the legacy benchmark — that's the
    # whole point of removing the nested-Pyxis wrap.
    assert "nel eval run" not in [
        line.strip()[: len("nel eval run")] for line in script.splitlines() if "container___gitlab" in line
    ]


def test_legacy_pre_renders_run_config_yaml(tmp_path: Path):
    """write_sbatch materializes run_config.yaml under {out}/{safe_name}/."""
    cfg = _legacy_config(str(tmp_path))
    write_sbatch(cfg)

    rc_paths = list(tmp_path.glob("**/run_config.yaml"))
    assert len(rc_paths) == 1, f"expected one run_config.yaml, got {rc_paths}"
    rc = yaml.safe_load(rc_paths[0].read_text())

    # Schema sanity: legacy v1 layout, with our solver fields written through.
    assert rc["config"]["type"] == "ifeval"
    assert rc["config"]["output_dir"] == "/results"
    assert rc["config"]["params"]["limit_samples"] == 3
    assert rc["target"]["api_endpoint"]["url"] == "https://api.example.com/v1/chat/completions"
    assert rc["target"]["api_endpoint"]["model_id"] == "nemotron"
    # api_key stays as the env var NAME; the value is injected via
    # --container-env=NEMO_API_KEY at runtime, never written to disk.
    assert rc["target"]["api_endpoint"]["api_key_name"] == "NEMO_API_KEY"
    assert "api_key" not in rc["target"]["api_endpoint"]


def test_legacy_rejects_cluster_shards(tmp_path: Path):
    cfg = _legacy_config(str(tmp_path), shards=2)

    with pytest.raises(ValueError, match=r"cluster\.shards.*container://"):
        write_sbatch(cfg)
    with pytest.raises(ValueError, match=r"cluster\.shards.*container://"):
        generate_sbatch(cfg, shard_idx=0, total_shards=2)


def test_legacy_allows_single_shard_noop(tmp_path: Path):
    cfg = _legacy_config(str(tmp_path), shards=1)

    script_paths, _extras = write_sbatch(cfg)

    assert len(script_paths) == 1
    assert "shard_0" not in str(script_paths[0])


def test_legacy_skips_when_results_already_exist(tmp_path: Path):
    """Generated sbatch is auto-resume safe: skips harness srun if results.yml exists."""
    cfg = _legacy_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)
    assert 'if [ -f "' in script, "missing auto-resume sentinel check"
    assert "results.yml" in script, "auto-resume sentinel does not check results.yml"
    assert "Already complete" in script, "missing 'Already complete' short-circuit message"


def test_legacy_omits_api_key_name_when_service_has_no_api_key(tmp_path: Path):
    """A self-deployed vLLM service has no ``api_key`` field — emitting
    ``api_key_name: NEMO_API_KEY`` then forces the harness inside the
    container to ``os.environ["NEMO_API_KEY"]`` (which is unset), and it
    raises ValueError("the value is not set").  v1 launcher's equivalent
    is ``api_key_name: null`` → harness skips the env-var lookup.  We
    mirror that: when the linked service has no resolved api_key, omit
    ``api_key_name`` from the run_config entirely."""
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "vllm": {
                    "type": "vllm",
                    "image": "/srv/vllm.sqsh",
                    "model": "/srv/model",
                    "served_model_name": "test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 1,
                    "node_pool": "gpu",
                },
            },
            "benchmarks": [
                {
                    "name": "container://x/y:1#ifeval",
                    "params": {"limit_samples": 3},
                    "solver": {"type": "container", "service": "vllm"},
                }
            ],
            "cluster": {
                "type": "slurm",
                "account": "test",
                "walltime": "00:30:00",
                "eval_image": "nvcr.io/nel:test",
                "node_pools": {"gpu": {"partition": "batch", "nodes": 1, "gpus_per_node": 4}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )
    write_sbatch(cfg)

    rc_paths = list(tmp_path.glob("**/run_config.yaml"))
    assert len(rc_paths) == 1
    rc = yaml.safe_load(rc_paths[0].read_text())
    assert "api_key_name" not in rc["target"]["api_endpoint"], (
        f"vLLM service has no api_key → api_key_name must be omitted; got: {rc['target']['api_endpoint']}"
    )


def test_legacy_api_service_without_model_uses_service_name(tmp_path: Path):
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "nemotron": {
                    "type": "api",
                    "url": "https://api.example.com/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            "benchmarks": [
                {
                    "name": "container://x/y:1#task",
                    "solver": {"type": "container", "service": "nemotron"},
                }
            ],
            "cluster": {
                "type": "slurm",
                "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )

    rc = _render_legacy_run_config(cfg.benchmarks[0], cfg.services)
    assert rc["target"]["api_endpoint"]["model_id"] == "nemotron"


def test_legacy_secrets_via_container_env_no_inline_values(tmp_path: Path):
    """Solver env_vars + service api_key flow through .secrets.env, never inline.

    Generated sbatch references env names via --container-env=NAME and
    re-exports values from secrets via mangled placeholder names; the raw
    values never appear as `export FOO=value` lines in the script.
    """
    cfg = _legacy_config(str(tmp_path))
    script, _, secrets = generate_sbatch(cfg)

    # Names appear via --container-env flags.
    assert "--container-env=NEMO_API_KEY" in script
    assert "--container-env=HF_TOKEN" in script
    assert "--container-env=OPENAI_API_KEY" in script

    # Raw values are NOT in the script — they live in secrets_content (which
    # is later written to .secrets.env at 0600 by _write_single_script).
    assert "secret-token" not in script
    assert "hf_xxx" not in script
    assert "sk-xxx" not in script
    # ... but they ARE in the secrets payload.
    assert "secret-token" in secrets.secrets_content
    assert "hf_xxx" in secrets.secrets_content
    assert "sk-xxx" in secrets.secrets_content


def test_legacy_mounts_run_config_and_results_dir(tmp_path: Path):
    """Generated srun bind-mounts the pre-rendered run_config.yaml and results dir."""
    cfg = _legacy_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)

    # run_config.yaml is mounted read-only at /config/run_config.yaml
    assert ":/config/run_config.yaml:ro" in script
    # The output bench dir houses both the run_config and results subdir
    assert "/run_config.yaml:/config/run_config.yaml:ro" in script
    assert "/results:/results" in script


def test_mixed_legacy_and_native_routes_each_correctly(tmp_path: Path):
    """A config with both container:// and native benches emits two distinct shapes."""
    cfg = _legacy_config(str(tmp_path), with_eval_image=True, with_native_too=True)
    script, sidecars, _ = generate_sbatch(cfg)

    # Native bench gets a sidecar (nel runs on compute for it).
    assert "skills___gsm8k" in sidecars
    # Legacy bench does not.
    assert not any("ifeval" in k for k in sidecars)
    # Both images are referenced.
    assert "lm-evaluation-harness:25.11" in script
    # The native bench routes through nel-next inside eval_image.
    assert "nvcr.io/nemo-evaluator:test" in script
    assert "nel eval run" in script


def test_legacy_export_runs_without_report_generation(tmp_path: Path):
    cfg = _legacy_config(str(tmp_path))
    cfg.output.report = []
    cfg.output.export = ["mlflow"]
    cfg.output.export_config = {"mlflow": {"tracking_uri": "file:///tmp/mlflow"}}

    scripts, extras = write_sbatch(cfg)
    script = scripts[0].read_text(encoding="utf-8")

    assert "nel eval report" not in script
    assert (
        'nel export "$OUTPUT_DIR" --dest mlflow --config "$OUTPUT_DIR/export_config.yaml" --output-dir "$OUTPUT_DIR"'
        in script
    )
    export_config_path = tmp_path / "export_config.yaml"
    assert export_config_path in extras
    assert export_config_path.exists()
    assert export_config_path.stat().st_mode & 0o777 == 0o600
    assert yaml.safe_load(export_config_path.read_text()) == cfg.output.export_config


def test_native_slurm_export_is_not_emitted_by_legacy_bridge(tmp_path: Path):
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "nemotron": {
                    "type": "api",
                    "url": "https://api.example.com/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "nemotron",
                }
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "solver": {"type": "simple", "service": "nemotron"},
                }
            ],
            "cluster": {
                "type": "slurm",
                "account": "test-account",
                "walltime": "00:30:00",
                "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
            },
            "output": {
                "dir": str(tmp_path),
                "export": ["mlflow"],
                "export_config": {"mlflow": {"tracking_uri": "file:///tmp/mlflow"}},
            },
        }
    )

    scripts, extras = write_sbatch(cfg)
    script = scripts[0].read_text(encoding="utf-8")

    assert "nel export" not in script
    export_config_path = tmp_path / "export_config.yaml"
    assert export_config_path not in extras
    assert not export_config_path.exists()


def _service_pre_cmd_config(output_dir: str) -> EvalConfig:
    """Multi-node vllm service with a user-supplied ``pre_cmd``.

    Stand-in for a v1 launcher gym deployment ports — the user owns the
    full service-side script (vLLM startup, fake health, gym poll-execute).
    """
    return EvalConfig.model_validate(
        {
            "services": {
                "vllm_ray": {
                    "type": "vllm",
                    "image": "/srv/vllm.sqsh",
                    "model": "/srv/model",
                    "served_model_name": "test-model",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "data_parallel_size": 8,
                    "num_nodes": 8,
                    "pre_cmd": "#!/bin/bash\nFAKE_HEALTH_PLACEHOLDER\nVLLM_PLACEHOLDER\nGYM_DANCE_PLACEHOLDER",
                    "node_pool": "gpu",
                },
            },
            "benchmarks": [
                {
                    "name": "container:///srv/nemo-gym.sqsh#nemo_gym",
                    "solver": {
                        "type": "container",
                        "service": "vllm_ray",
                    },
                }
            ],
            "cluster": {
                "type": "slurm",
                "account": "test",
                "walltime": "01:00:00",
                "eval_image": "nvcr.io/nel:test",
                "mount_home": False,
                "node_pools": {"gpu": {"partition": "batch", "nodes": 8, "gpus_per_node": 4}},
            },
            "output": {"dir": output_dir},
        }
    )


def test_pre_cmd_replaces_default_service_command(tmp_path: Path):
    """When ``pre_cmd`` is set, the script file the service execs contains
    only the user's script — no implicit ``vllm serve …`` line, no
    ``set -euo pipefail`` prefix."""
    cfg = _service_pre_cmd_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)

    # User markers present.
    assert "FAKE_HEALTH_PLACEHOLDER" in script
    assert "VLLM_PLACEHOLDER" in script
    assert "GYM_DANCE_PLACEHOLDER" in script
    # Default vllm command absent — `vllm serve <model> --port …`.
    # (The echo line "...Ray vllm server..." contains the substring "vllm s",
    # so we look for the actual command form.)
    assert "vllm serve /model --port 8000" not in script
    # set -euo pipefail prefix not added (user controls error semantics).
    # (We allow `set -uo pipefail` in the sbatch header — that's nel-next's,
    # not the per-service script.)
    svc_block_start = script.index("FAKE_HEALTH_PLACEHOLDER")
    # Walk back to the start of the heredoc and verify no `set -euo` line.
    heredoc_window = script[max(0, svc_block_start - 200) : svc_block_start]
    assert "set -euo pipefail" not in heredoc_window


def test_pre_cmd_keeps_default_health_check(tmp_path: Path):
    """``pre_cmd`` is the user's escape hatch for service startup, but
    nel-next still polls ``/health`` to gate the eval container.  That
    curl is the coordination signal v1 launcher's fake-health pattern
    relies on — one shot to unblock fake-health, then vLLM takes over
    port 8000.  Skipping it would leave fake-health on the port and
    crash vLLM."""
    cfg = _service_pre_cmd_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)
    assert "Skipping health check for vllm_ray (pre_cmd controls readiness)" not in script
    assert 'curl -sf "http://localhost:8000/health"' in script


def test_pre_cmd_propagates_nel_invocation_id(tmp_path: Path):
    """``NEL_INVOCATION_ID`` is exported in the sbatch header (so siblings
    in the auto-resume chain share the value) and forwarded into every
    Pyxis container via ``--container-env``."""
    cfg = _service_pre_cmd_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)

    # Exported once in the header with a hex token (not the literal placeholder).
    matches = re.findall(r"^export NEL_INVOCATION_ID=([0-9a-f]+)$", script, re.MULTILINE)
    assert len(matches) == 1, f"expected exactly one NEL_INVOCATION_ID export, found {matches}"
    assert len(matches[0]) == 32, f"expected uuid hex (32 chars), got {matches[0]!r}"
    # Forwarded into containers (we don't assert exact count — at least one
    # service-block srun and one eval-container srun should reference it).
    assert script.count("--container-env=NEL_INVOCATION_ID") >= 2


def test_solver_env_vars_broadcast_to_service_container_for_legacy_bench(tmp_path: Path):
    """v1 launcher's ``env_vars:`` block is container-agnostic — values reach
    both deployment and evaluation containers.  Our nel-next mirror is
    ``solver.env_vars`` on a container:// benchmark.  When that benchmark's
    linked service has its own container (vllm/sglang/gym Pattern A),
    ``solver.env_vars`` MUST reach the service container too, otherwise a
    user's gym deployment script that expands ``$INFERENCE_API_KEY`` etc.
    silently sees an empty string.
    """
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "vllm_ray": {
                    "type": "vllm",
                    "image": "/srv/vllm.sqsh",
                    "model": "/srv/model",
                    "served_model_name": "test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 4,
                    "num_nodes": 2,
                    "pre_cmd": "#!/bin/bash\necho $INFERENCE_API_KEY\n",
                    "node_pool": "gpu",
                },
            },
            "benchmarks": [
                {
                    "name": "container:///srv/nemo-gym.sqsh#nemo_gym",
                    "solver": {
                        "type": "container",
                        "service": "vllm_ray",
                        "env_vars": {"INFERENCE_API_KEY": "v1-style-key"},
                    },
                }
            ],
            "cluster": {
                "type": "slurm",
                "account": "test",
                "walltime": "01:00:00",
                "eval_image": "nvcr.io/nel:test",
                "node_pools": {"gpu": {"partition": "batch", "nodes": 2, "gpus_per_node": 4}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )
    script, _, _ = generate_sbatch(cfg)
    # The service srun is the multi-node one; both that and the eval srun
    # should advertise INFERENCE_API_KEY via --container-env.
    svc_lines = [
        line for line in script.splitlines() if "srun --mpi=pmix --overlap --mem=0" in line and "vllm.sqsh" in line
    ]
    assert svc_lines, "expected vllm_ray service srun line"
    assert any("--container-env=INFERENCE_API_KEY" in line for line in svc_lines), (
        "solver.env_vars must broadcast to the service container srun (v1 env_vars semantics); "
        f"got service srun heads: {svc_lines!r}"
    )


def test_pre_cmd_does_not_affect_default_service(tmp_path: Path):
    """Sanity: services without ``pre_cmd`` still get the default vllm/sglang
    template + health wait."""
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "vllm": {
                    "type": "vllm",
                    "image": "/srv/vllm.sqsh",
                    "model": "/srv/model",
                    "served_model_name": "test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 1,
                    "node_pool": "gpu",
                },
            },
            "benchmarks": [
                {
                    "name": "skills://gsm8k",
                    "solver": {"type": "simple", "service": "vllm"},
                }
            ],
            "cluster": {
                "type": "slurm",
                "account": "test",
                "walltime": "01:00:00",
                "eval_image": "nvcr.io/nel:test",
                "node_pools": {"gpu": {"partition": "batch", "nodes": 1, "gpus_per_node": 4}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )
    script, _, _ = generate_sbatch(cfg)
    assert "vllm serve" in script
    # Default health-wait is emitted (we look for the curl probe).
    assert 'curl -sf "http://localhost:8000/health"' in script
    assert "Skipping health check" not in script


def test_legacy_pins_to_master_ip_when_dependent_service_is_multinode(tmp_path: Path):
    """When a legacy ``container://`` benchmark depends on a multi-node
    service (e.g. vLLM with num_nodes>1), the harness srun must include
    ``-w $MASTER_IP`` so it lands on the same node as the service head.
    Pyxis on the same node shares the host network, so the eval-factory
    adapter proxy on 127.0.0.1:<port> in the harness container is then
    reachable from the service container's loopback (where, e.g., the
    gym CLI's HTTP client runs).  Without this, the harness drifts to a
    worker node and the cross-container loopback connect fails."""
    cfg = _service_pre_cmd_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)

    # The legacy block contains exactly one harness srun per benchmark.
    # We look for the srun line on its own (continuation backslashes split
    # the rest across follow-up lines, so we only need to match the head).
    srun_heads = [
        line
        for line in script.splitlines()
        if "srun --mpi=pmix --overlap --unbuffered" in line and "--ntasks 1" in line
    ]
    assert srun_heads, "expected at least one srun head line in the script"
    # At least one must carry ``-w $MASTER_IP`` (the legacy harness srun for
    # the multi-node-service-dependent benchmark).
    assert any("-w $MASTER_IP" in line for line in srun_heads), (
        f"expected ``-w $MASTER_IP`` on a srun head when bench depends on multi-node service; got: {srun_heads!r}"
    )


def test_service_gets_legacy_results_dir_mount(tmp_path: Path):
    """When a legacy ``container://`` benchmark depends on a service, the
    service container's srun must mount the bench's results dir at
    ``/results`` — otherwise harness writes from ``pre_cmd`` (gym
    Pattern A) land in ephemeral container storage and the eval-factory
    output parser sees an empty dir.  v1 launcher does this implicitly
    (executors/slurm/executor.py:826-828)."""
    cfg = _service_pre_cmd_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)

    # Find the service-block srun (rl image is the marker in this fixture).
    svc_lines = [
        line for line in script.splitlines() if "srun --mpi=pmix --overlap --mem=0" in line and "vllm.sqsh" in line
    ]
    assert svc_lines, "expected vllm_ray service srun head line"
    # Match the literal "<host_path_to_bench>/results:/results" tail.
    expected_fragment = "/results:/results"
    matched = any(expected_fragment in line for line in svc_lines)
    assert matched, (
        "service srun must include a legacy bench's <results_dir>:/results bind mount; got srun heads:\n  "
        + "\n  ".join(svc_lines)
    )


def test_service_no_legacy_mount_when_no_legacy_bench(tmp_path: Path):
    """Regression guard: services without any dependent legacy benchmark
    must NOT get a spurious ``/results`` bind from this code path.  Native
    benchmarks own their own mounts via the eval container."""
    # Use a config that has only a native (non-container://) benchmark.
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "vllm": {
                    "type": "vllm",
                    "image": "/srv/vllm.sqsh",
                    "model": "/srv/model",
                    "served_model_name": "test",
                    "protocol": "chat_completions",
                    "tensor_parallel_size": 1,
                    "node_pool": "gpu",
                },
            },
            "benchmarks": [
                {"name": "skills://gsm8k", "solver": {"type": "simple", "service": "vllm"}},
            ],
            "cluster": {
                "type": "slurm",
                "account": "test",
                "walltime": "01:00:00",
                "eval_image": "nvcr.io/nel:test",
                "node_pools": {"gpu": {"partition": "batch", "nodes": 1, "gpus_per_node": 4}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )
    script, _, _ = generate_sbatch(cfg)
    svc_lines = [
        line for line in script.splitlines() if "srun --mpi=pmix --overlap --mem=0" in line and "vllm.sqsh" in line
    ]
    assert svc_lines, "expected vllm service srun head line"
    for line in svc_lines:
        # No `<...>/results:/results` bind on the service srun for
        # native-only configs.
        assert "/results:/results" not in line, (
            f"native-only config must not get a /results bind on the service srun; got: {line!r}"
        )


def test_legacy_no_master_ip_pin_when_service_is_single_node(tmp_path: Path):
    """Regression guard: the ``-w $MASTER_IP`` flag must NOT appear on the
    legacy srun for a single-node service.  Single-node services run on
    the sbatch's only node anyway; pinning would be redundant and could
    fail in pools with strict node selection."""
    cfg = _legacy_config(str(tmp_path))
    script, _, _ = generate_sbatch(cfg)

    srun_heads = [
        line
        for line in script.splitlines()
        if "srun --mpi=pmix --overlap --unbuffered" in line and "--ntasks 1" in line
    ]
    assert srun_heads, "expected at least one srun head line"
    for line in srun_heads:
        assert "-w $MASTER_IP" not in line, (
            f"-w $MASTER_IP must not appear when service deps are single-node; got: {line!r}"
        )


def test_legacy_dedups_overlapping_mounts(tmp_path: Path):
    """If a user declares the same ``host:container`` mount in both
    ``cluster.container_mounts`` AND ``solver.mounts`` (easy to do when
    porting a v1 hydra config that has overlapping deployment/evaluation
    mount lists), Pyxis sees the same mount twice on the srun command and
    behaviour is undefined.  Dedupe so the mount appears once."""
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "nemotron": {
                    "type": "api",
                    "url": "https://api.example.com/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "nemotron",
                    "api_key": "tok",
                },
            },
            "benchmarks": [
                {
                    "name": "container://x/y:1#task",
                    "solver": {
                        "type": "container",
                        "service": "nemotron",
                        "mounts": {"/srv/shared": "/srv/shared"},
                    },
                }
            ],
            "cluster": {
                "type": "slurm",
                "account": "test",
                "walltime": "00:30:00",
                "container_mounts": ["/srv/shared:/srv/shared"],
                "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )
    script, _, _ = generate_sbatch(cfg)
    # Find the --container-mounts= line(s) for the legacy srun and verify
    # the duplicated mount only appears once.
    mount_lines = [line for line in script.splitlines() if "--container-mounts=" in line and "/srv/shared" in line]
    assert mount_lines, "expected at least one --container-mounts= line carrying /srv/shared"
    for line in mount_lines:
        mounts_arg = line.split("--container-mounts=")[1].split(" ")[0].rstrip("\\,")
        assert mounts_arg.count("/srv/shared:/srv/shared") == 1, (
            f"mount appears multiple times — dedup missing; got mounts arg: {mounts_arg!r}"
        )


def test_legacy_rejects_incompatible_service_type_at_config_load(tmp_path: Path):
    """A container:// benchmark needs a chat/completions endpoint.
    Pointing it at a ``type: gym`` (resource server) raises a clear
    ValueError at ``EvalConfig.model_validate`` time — before any sbatch
    rendering — so users see the misconfiguration immediately rather
    than after submit produces a broken ``run_config.yaml`` with empty
    ``model_id``."""
    with pytest.raises(ValueError, match="requires a model service"):
        EvalConfig.model_validate(
            {
                "services": {
                    "tool_server": {
                        "type": "gym",
                        "benchmark": "calendar",
                        "port": 9090,
                    },
                },
                "benchmarks": [
                    {
                        "name": "container://x/y:1#task",
                        "solver": {"type": "container", "service": "tool_server"},
                    }
                ],
                "cluster": {
                    "type": "slurm",
                    "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
                },
                "output": {"dir": str(tmp_path)},
            }
        )


def test_legacy_rejects_service_proxy_on_slurm_submit(tmp_path: Path):
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "nemotron": {
                    "type": "api",
                    "url": "https://api.example.com/v1/chat/completions",
                    "protocol": "chat_completions",
                    "model": "nemotron",
                    "proxy": {"extra_body": {"thinking": True}},
                },
            },
            "benchmarks": [
                {
                    "name": "container://x/y:1#task",
                    "solver": {"type": "container", "service": "nemotron"},
                }
            ],
            "cluster": {
                "type": "slurm",
                "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )

    with pytest.raises(ValueError, match=r"adapter proxy.*solver.adapter_config"):
        generate_sbatch(cfg)


def test_render_legacy_run_config_handles_managed_service(tmp_path: Path):
    """Managed model server (vllm) resolves to localhost URL with protocol suffix."""
    cfg = EvalConfig.model_validate(
        {
            "services": {
                "vllm-svc": {
                    "type": "vllm",
                    "model": "/path/model.bin",
                    "protocol": "completions",
                    "tensor_parallel_size": 1,
                    "port": 9000,
                    "node_pool": "cpu",
                }
            },
            "benchmarks": [
                {
                    "name": "container://x/y:1#task",
                    "solver": {
                        "type": "container",
                        "service": "vllm-svc",
                    },
                }
            ],
            "cluster": {
                "type": "slurm",
                "node_pools": {"cpu": {"partition": "cpu", "nodes": 1}},
            },
            "output": {"dir": str(tmp_path)},
        }
    )
    rc = _render_legacy_run_config(cfg.benchmarks[0], cfg.services)
    assert rc["target"]["api_endpoint"]["url"].startswith("http://localhost:9000/v1")
    assert rc["target"]["api_endpoint"]["url"].endswith("/completions")
    assert rc["target"]["api_endpoint"]["type"] == "completions"


def test_legacy_generated_script_is_bash_clean(tmp_path: Path):
    """Run `bash -n` on the generated script to confirm syntactic correctness."""
    cfg = _legacy_config(str(tmp_path), with_eval_image=True)
    script, _, _ = generate_sbatch(cfg)
    bash_path = shutil.which("bash")
    assert bash_path is not None, "bash not found in PATH"
    proc = subprocess.run([bash_path, "-n"], input=script.encode(), capture_output=True, check=False)
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr.decode()}"
