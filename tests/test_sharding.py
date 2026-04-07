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

from nemo_evaluator.engine.sharding import get_shard_range, shard_from_env


class TestGetShardRange:
    def test_even_split(self):
        assert get_shard_range(100, 0, 4) == (0, 25)
        assert get_shard_range(100, 1, 4) == (25, 50)
        assert get_shard_range(100, 2, 4) == (50, 75)
        assert get_shard_range(100, 3, 4) == (75, 100)

    def test_uneven_split(self):
        # 10 problems across 3 shards: 4, 3, 3
        assert get_shard_range(10, 0, 3) == (0, 4)
        assert get_shard_range(10, 1, 3) == (4, 7)
        assert get_shard_range(10, 2, 3) == (7, 10)

    def test_coverage(self):
        total = 14001
        shards = 16
        ranges = [get_shard_range(total, i, shards) for i in range(shards)]
        # All problems covered
        all_indices = set()
        for s, e in ranges:
            all_indices.update(range(s, e))
        assert len(all_indices) == total
        # No overlaps
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_total_zero(self):
        assert get_shard_range(0, 0, 4) == (0, 0)

    def test_more_shards_than_problems(self):
        # 3 problems, 10 shards: first 3 get 1 each, rest get empty
        assert get_shard_range(3, 0, 10) == (0, 1)
        assert get_shard_range(3, 2, 10) == (2, 3)
        assert get_shard_range(3, 3, 10) == (3, 3)  # empty
        assert get_shard_range(3, 9, 10) == (3, 3)  # empty

    def test_single_shard(self):
        assert get_shard_range(100, 0, 1) == (0, 100)

    def test_invalid_shard_idx(self):
        with pytest.raises(ValueError):
            get_shard_range(100, -1, 4)
        with pytest.raises(ValueError):
            get_shard_range(100, 4, 4)


class TestShardFromEnv:
    def test_nel_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEL_SHARD_IDX", "3")
        monkeypatch.setenv("NEL_TOTAL_SHARDS", "8")
        assert shard_from_env() == (3, 8)

    def test_slurm_env_vars(self, monkeypatch):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "16")
        assert shard_from_env() == (5, 16)

    def test_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("NEL_SHARD_IDX", raising=False)
        monkeypatch.delenv("NEL_TOTAL_SHARDS", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
        assert shard_from_env() is None

    def test_invalid_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEL_SHARD_IDX", "abc")
        monkeypatch.setenv("NEL_TOTAL_SHARDS", "xyz")
        assert shard_from_env() is None
