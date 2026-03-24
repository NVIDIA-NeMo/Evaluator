"""Tests for SLURM array job generation and eval_loop shard_info."""

from nemo_evaluator.orchestration.config import EvalConfig
from nemo_evaluator.orchestration.slurm_gen import generate_sbatch


def _make_slurm_config(shards=None, auto_resume=False):
    return EvalConfig.model_validate(
        {
            "services": {
                "model": {
                    "type": "api",
                    "url": "http://x/v1/chat/completions",
                    "protocol": "chat_completions",
                },
            },
            "benchmarks": [
                {"name": "gsm8k", "repeats": 5, "solver": {"type": "simple", "service": "model"}},
            ],
            "cluster": {
                "type": "slurm",
                "walltime": "02:00:00",
                "auto_resume": auto_resume,
                "shards": shards,
                "node_pools": {
                    "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                },
            },
        }
    )


class TestSbatchArrayGeneration:
    def test_no_shards_no_array(self):
        cfg = _make_slurm_config(shards=None)
        script, _, _ = generate_sbatch(cfg)
        assert "--array" not in script
        assert "NEL_SHARD_IDX" not in script
        assert "nel eval merge" not in script

    def test_shards_emits_array(self):
        cfg = _make_slurm_config(shards=4)
        script, _, _ = generate_sbatch(cfg)
        assert "#SBATCH --array=0-3" in script

    def test_shards_exports_env_vars(self):
        cfg = _make_slurm_config(shards=8)
        script, _, _ = generate_sbatch(cfg)
        assert "export NEL_SHARD_IDX=$SLURM_ARRAY_TASK_ID" in script
        assert "export NEL_TOTAL_SHARDS=$SLURM_ARRAY_TASK_COUNT" in script

    def test_shards_per_shard_output_dir(self):
        cfg = _make_slurm_config(shards=4)
        script, _, _ = generate_sbatch(cfg)
        assert 'OUTPUT_DIR="$OUTPUT_DIR/shard_$SLURM_ARRAY_TASK_ID"' in script

    def test_shards_skips_report(self):
        cfg = _make_slurm_config(shards=4)
        script, _, _ = generate_sbatch(cfg)
        assert "nel eval report" not in script

    def test_shards_skips_auto_resume(self):
        """auto_resume + shards is rejected at config level, but verify script
        doesn't contain resume logic when shards is set."""
        cfg = _make_slurm_config(shards=4)
        script, _, _ = generate_sbatch(cfg)
        assert "afternotok" not in script
        assert "sacct" not in script

    def test_shards_includes_merge_hint(self):
        cfg = _make_slurm_config(shards=4)
        script, _, _ = generate_sbatch(cfg)
        assert "nel eval merge" in script

    def test_no_shards_has_report(self):
        cfg = _make_slurm_config(shards=None)
        script, _, _ = generate_sbatch(cfg)
        assert "nel eval report" in script


class TestAutoresumePrologue:
    def test_autoresume_prologue_in_script(self):
        cfg = _make_slurm_config(auto_resume=True)
        script, _, _ = generate_sbatch(cfg)
        assert "sacct" in script
        assert "afternotok" in script
        assert "_retry_file" in script
        assert "_walltime_file" in script
        assert "_prev_slurm_job_id" in script

    def test_autoresume_disabled(self):
        cfg = _make_slurm_config(auto_resume=False)
        script, _, _ = generate_sbatch(cfg)
        assert "sacct" not in script
        assert "afternotok" not in script

    def test_max_walltime_block_present(self):
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {"name": "gsm8k", "repeats": 5, "solver": {"type": "simple", "service": "model"}},
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "auto_resume": True,
                    "max_walltime": "48:00:00",
                    "node_pools": {
                        "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                    },
                },
            }
        )
        script, _, _ = generate_sbatch(cfg)
        assert "172800" in script
        assert "Max walltime" in script

    def test_max_walltime_omitted_when_none(self):
        cfg = _make_slurm_config(auto_resume=True)
        script, _, _ = generate_sbatch(cfg)
        assert "Max walltime" not in script

    def test_max_retries_in_script(self):
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {"name": "gsm8k", "repeats": 5, "solver": {"type": "simple", "service": "model"}},
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "auto_resume": True,
                    "max_retries": 5,
                    "node_pools": {
                        "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                    },
                },
            }
        )
        script, _, _ = generate_sbatch(cfg)
        assert "Infra retry limit (5)" in script


    def test_cancelled_handling_in_script(self):
        cfg = _make_slurm_config(auto_resume=True)
        script, _, _ = generate_sbatch(cfg)
        assert "CANCELLED" in script
        assert "was cancelled" in script

    def test_sacct_retry_loop_in_script(self):
        cfg = _make_slurm_config(auto_resume=True)
        script, _, _ = generate_sbatch(cfg)
        assert "_sacct_try" in script
        assert "sleep 2" in script

    def test_sbatch_failure_warning_in_script(self):
        cfg = _make_slurm_config(auto_resume=True)
        script, _, _ = generate_sbatch(cfg)
        assert "WARNING: Failed to submit auto-resume" in script

    def test_prologue_before_cleanup_trap(self):
        cfg = _make_slurm_config(auto_resume=True)
        script, _, _ = generate_sbatch(cfg)
        prologue_pos = script.index("Auto-resume chain")
        trap_pos = script.index("trap cleanup EXIT")
        assert prologue_pos < trap_pos

    def test_max_walltime_days_format(self):
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {"name": "gsm8k", "repeats": 5, "solver": {"type": "simple", "service": "model"}},
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "auto_resume": True,
                    "max_walltime": "2-00:00:00",
                    "node_pools": {
                        "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                    },
                },
            }
        )
        script, _, _ = generate_sbatch(cfg)
        assert "172800" in script

    def test_max_walltime_invalid_format_rejected(self):
        import pytest

        with pytest.raises(Exception):
            EvalConfig.model_validate(
                {
                    "services": {
                        "model": {
                            "type": "api",
                            "url": "http://x/v1/chat/completions",
                            "protocol": "chat_completions",
                        },
                    },
                    "benchmarks": [
                        {"name": "gsm8k", "solver": {"type": "simple", "service": "model"}},
                    ],
                    "cluster": {
                        "type": "slurm",
                        "walltime": "02:00:00",
                        "max_walltime": "banana",
                        "node_pools": {
                            "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                        },
                    },
                }
            )

    def test_max_retries_negative_rejected(self):
        import pytest

        with pytest.raises(Exception):
            EvalConfig.model_validate(
                {
                    "services": {
                        "model": {
                            "type": "api",
                            "url": "http://x/v1/chat/completions",
                            "protocol": "chat_completions",
                        },
                    },
                    "benchmarks": [
                        {"name": "gsm8k", "solver": {"type": "simple", "service": "model"}},
                    ],
                    "cluster": {
                        "type": "slurm",
                        "walltime": "02:00:00",
                        "max_retries": -1,
                        "node_pools": {
                            "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                        },
                    },
                }
            )


class TestPerServiceLogFiles:
    def test_logs_dir_created(self):
        cfg = _make_slurm_config()
        script, _, _ = generate_sbatch(cfg)
        assert 'mkdir -p "$OUTPUT_DIR/logs"' in script

    def test_eval_task_tees_to_log_file(self):
        cfg = _make_slurm_config()
        script, _, _ = generate_sbatch(cfg)
        assert "tee" in script
        assert "logs/eval-" in script
        assert "PIPESTATUS" in script

    def test_vllm_service_log_redirect(self):
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "vllm",
                        "model": "m",
                        "protocol": "chat_completions",
                        "port": 8000,
                    },
                },
                "benchmarks": [
                    {"name": "gsm8k", "solver": {"type": "simple", "service": "model"}},
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "node_pools": {
                        "compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"},
                    },
                },
            }
        )
        script, _, _ = generate_sbatch(cfg)
        assert 'logs/server-model.log' in script


class TestSidecarConfigGeneration:
    """Complex benchmarks (harbor solver + ecs sandbox) generate sidecar YAML."""

    def test_harbor_bench_gets_sidecar(self):
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {
                        "name": "harbor://swebench-verified@1.0",
                        "solver": {"type": "harbor", "service": "model", "agent": "openhands-sdk"},
                        "sandbox": {"type": "ecs_fargate", "region": "us-west-2"},
                    }
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "node_pools": {"compute": {"partition": "batch", "nodes": 1}},
                },
            }
        )
        script, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 1
        key = list(sidecars.keys())[0]
        sidecar = sidecars[key]
        assert sidecar["services"]["model"]["type"] == "api"
        assert "${MODEL_URL}" in sidecar["services"]["model"]["url"]
        assert "${MODEL_MODEL}" in sidecar["services"]["model"]["model"]
        assert sidecar["benchmarks"][0]["solver"]["type"] == "harbor"
        assert sidecar["benchmarks"][0]["sandbox"]["type"] == "ecs_fargate"
        assert "nel eval run" in script
        assert f"config_{key}.yaml" in script
        assert "srun --overlap" not in script.split("nel eval run")[0].split("Benchmark 1/1")[-1]

    def test_simple_bench_uses_quick_mode(self):
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {
                        "name": "gsm8k",
                        "solver": {"type": "simple", "service": "model"},
                    }
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "node_pools": {"compute": {"partition": "batch", "nodes": 1}},
                },
            }
        )
        script, sidecars, _ = generate_sbatch(cfg)
        assert len(sidecars) == 0
        assert '--bench "gsm8k"' in script

    def test_secrets_not_in_sbatch_script(self):
        """Credentials must go to .secrets.env, not inline in the sbatch script."""
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {
                        "name": "gsm8k",
                        "solver": {"type": "simple", "service": "model"},
                    }
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "container_env": {
                        "HF_TOKEN": "hf_SUPERSECRET1234",
                        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG",
                    },
                    "node_pools": {"compute": {"partition": "batch", "nodes": 1}},
                },
            }
        )
        script, _, secrets_result = generate_sbatch(cfg)
        assert "hf_SUPERSECRET1234" not in script
        assert "wJalrXUtnFEMI/K7MDENG" not in script
        assert "source" in script and ".secrets.env" in script
        assert "hf_SUPERSECRET1234" in secrets_result.secrets_content
        assert "wJalrXUtnFEMI/K7MDENG" in secrets_result.secrets_content

    def test_per_service_env_isolation(self):
        """Different services with same env var name get disambiguated keys."""
        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "svc_a": {
                        "type": "vllm",
                        "model": "model-a",
                        "protocol": "chat_completions",
                        "port": 8000,
                        "extra_env": {"HF_TOKEN": "token_for_a"},
                    },
                    "svc_b": {
                        "type": "vllm",
                        "model": "model-b",
                        "protocol": "chat_completions",
                        "port": 8001,
                        "extra_env": {"HF_TOKEN": "token_for_b"},
                    },
                },
                "benchmarks": [
                    {
                        "name": "gsm8k",
                        "solver": {"type": "simple", "service": "svc_a"},
                    }
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "node_pools": {"compute": {"partition": "batch", "nodes": 1, "gres": "gpu:4"}},
                },
            }
        )
        script, _, secrets_result = generate_sbatch(cfg)
        # Both values in .secrets.env under disambiguated names
        assert "token_for_a" in secrets_result.secrets_content
        assert "token_for_b" in secrets_result.secrets_content
        # Neither plaintext value appears in the script
        assert "token_for_a" not in script
        assert "token_for_b" not in script
        # Re-export commands present before srun
        assert 'export HF_TOKEN="$' in script

    def test_shards_plus_sidecar_rejected(self):
        """Sharding with complex benchmarks must raise at generation time."""
        import pytest

        cfg = EvalConfig.model_validate(
            {
                "services": {
                    "model": {
                        "type": "api",
                        "url": "http://x/v1/chat/completions",
                        "protocol": "chat_completions",
                    },
                },
                "benchmarks": [
                    {
                        "name": "harbor://swebench-verified@1.0",
                        "solver": {"type": "harbor", "service": "model", "agent": "openhands"},
                        "sandbox": {"type": "ecs_fargate", "region": "us-west-2"},
                    }
                ],
                "cluster": {
                    "type": "slurm",
                    "walltime": "02:00:00",
                    "shards": 4,
                    "auto_resume": False,
                    "node_pools": {"compute": {"partition": "batch", "nodes": 1}},
                },
            }
        )
        with pytest.raises(ValueError, match="incompatible"):
            generate_sbatch(cfg)


class TestShardInfoComputation:
    """Test that shard_info correctly computes problem_range via get_shard_range."""

    def test_shard_info_computes_range(self):
        from nemo_evaluator.engine.sharding import get_shard_range

        ds_size = 100
        shard_idx, total_shards = 0, 4
        problem_range = get_shard_range(ds_size, shard_idx, total_shards)
        assert problem_range == (0, 25)

    def test_shard_info_with_max_problems(self):
        from nemo_evaluator.engine.sharding import get_shard_range

        ds_size = 1000
        max_problems = 50
        effective = min(ds_size, max_problems)
        problem_range = get_shard_range(effective, 1, 4)
        assert problem_range == (13, 26)

    def test_shard_info_respects_effective_size(self):
        """Sharding must use min(ds_size, max_problems), not raw ds_size."""
        from nemo_evaluator.engine.sharding import get_shard_range

        ranges = [get_shard_range(50, i, 4) for i in range(4)]
        all_indices = set()
        for s, e in ranges:
            all_indices.update(range(s, e))
        assert len(all_indices) == 50
        assert min(all_indices) == 0
        assert max(all_indices) == 49
