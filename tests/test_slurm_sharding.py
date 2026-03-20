"""Tests for SLURM array job generation and eval_loop shard_info."""
from nemo_evaluator.eval.config import EvalConfig
from nemo_evaluator.eval.slurm_gen import generate_sbatch


def _make_slurm_config(shards=None, auto_resume=False):
    return EvalConfig.model_validate({
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
    })


class TestSbatchArrayGeneration:
    def test_no_shards_no_array(self):
        cfg = _make_slurm_config(shards=None)
        script = generate_sbatch(cfg)
        assert "--array" not in script
        assert "NEL_SHARD_IDX" not in script
        assert "nel eval merge" not in script

    def test_shards_emits_array(self):
        cfg = _make_slurm_config(shards=4)
        script = generate_sbatch(cfg)
        assert "#SBATCH --array=0-3" in script

    def test_shards_exports_env_vars(self):
        cfg = _make_slurm_config(shards=8)
        script = generate_sbatch(cfg)
        assert "export NEL_SHARD_IDX=$SLURM_ARRAY_TASK_ID" in script
        assert "export NEL_TOTAL_SHARDS=$SLURM_ARRAY_TASK_COUNT" in script

    def test_shards_per_shard_output_dir(self):
        cfg = _make_slurm_config(shards=4)
        script = generate_sbatch(cfg)
        assert 'OUTPUT_DIR="$OUTPUT_DIR/shard_$SLURM_ARRAY_TASK_ID"' in script

    def test_shards_skips_report(self):
        cfg = _make_slurm_config(shards=4)
        script = generate_sbatch(cfg)
        assert "nel eval report" not in script

    def test_shards_skips_auto_resume(self):
        """auto_resume + shards is rejected at config level, but verify script
        doesn't contain resume logic when shards is set."""
        cfg = _make_slurm_config(shards=4)
        script = generate_sbatch(cfg)
        assert "ATTEMPT_FILE" not in script
        assert "resubmitting" not in script.lower()

    def test_shards_includes_merge_hint(self):
        cfg = _make_slurm_config(shards=4)
        script = generate_sbatch(cfg)
        assert "nel eval merge" in script

    def test_no_shards_has_report(self):
        cfg = _make_slurm_config(shards=None)
        script = generate_sbatch(cfg)
        assert "nel eval report" in script


class TestShardInfoComputation:
    """Test that shard_info correctly computes problem_range via get_shard_range."""

    def test_shard_info_computes_range(self):
        from nemo_evaluator.runner.sharding import get_shard_range

        ds_size = 100
        shard_idx, total_shards = 0, 4
        problem_range = get_shard_range(ds_size, shard_idx, total_shards)
        assert problem_range == (0, 25)

    def test_shard_info_with_max_problems(self):
        from nemo_evaluator.runner.sharding import get_shard_range

        ds_size = 1000
        max_problems = 50
        effective = min(ds_size, max_problems)
        problem_range = get_shard_range(effective, 1, 4)
        assert problem_range == (13, 26)

    def test_shard_info_respects_effective_size(self):
        """Sharding must use min(ds_size, max_problems), not raw ds_size."""
        from nemo_evaluator.runner.sharding import get_shard_range

        ranges = [get_shard_range(50, i, 4) for i in range(4)]
        all_indices = set()
        for s, e in ranges:
            all_indices.update(range(s, e))
        assert len(all_indices) == 50
        assert min(all_indices) == 0
        assert max(all_indices) == 49
