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
import pytest
from pydantic import ValidationError

from nemo_evaluator.schemas import (
    BenchmarkResult,
    EvalBundle,
    RetryConfig,
    RunSnapshot,
    ShardConfig,
)


class TestEvalBundle:
    def test_empty_run_id_rejected(self):
        with pytest.raises(ValidationError):
            EvalBundle(
                run_id="",
                config_hash="sha256:abc123",
                sdk_version="0.1.0",
                config=RunSnapshot(),
                benchmark=BenchmarkResult(name="test", samples=0),
            )

    def test_auto_timestamp(self):
        b = EvalBundle(
            run_id="test",
            config_hash="sha256:abc",
            sdk_version="0.1.0",
            config=RunSnapshot(),
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
