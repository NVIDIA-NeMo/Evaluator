import pytest
from pydantic import ValidationError

from nemo_evaluator.models import (
    EvalBundle,
    EvalConfig,
    BenchmarkResult,
    ShardConfig,
    RetryConfig,
)


class TestEvalBundle:
    def test_empty_run_id_rejected(self):
        with pytest.raises(ValidationError):
            EvalBundle(
                run_id="",
                config_hash="sha256:abc123",
                sdk_version="0.1.0",
                config=EvalConfig(),
                benchmark=BenchmarkResult(name="test", samples=0),
            )

    def test_auto_timestamp(self):
        b = EvalBundle(
            run_id="test",
            config_hash="sha256:abc",
            sdk_version="0.1.0",
            config=EvalConfig(),
            benchmark=BenchmarkResult(name="test", samples=0),
        )
        assert "T" in b.timestamp


class TestShardConfig:
    def test_negative_idx_rejected(self):
        with pytest.raises(ValidationError):
            ShardConfig(shard_idx=-1, total_shards=8, problem_range=(0, 100))

    def test_zero_shards_rejected(self):
        with pytest.raises(ValidationError):
            ShardConfig(shard_idx=0, total_shards=0, problem_range=(0, 100))


class TestRetryConfig:
    def test_defaults_include_common_retryable_codes(self):
        rc = RetryConfig()
        assert 429 in rc.retry_on_status
        assert 503 in rc.retry_on_status
