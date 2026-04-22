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
"""Tests for response caching."""

from nemo_evaluator.engine.cache import ResponseCache


class TestResponseCache:
    def test_miss_then_hit(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        assert cache.get("m", "prompt", None, 0.0, 100) is None
        assert cache.stats["misses"] == 1

        cache.put("m", "prompt", None, 0.0, 100, {"choices": [{"message": {"content": "hi"}}]})
        result = cache.get("m", "prompt", None, 0.0, 100)
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "hi"
        assert cache.stats["hits"] == 1

    def test_nonzero_temperature_not_cached(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        cache.put("m", "prompt", None, 0.7, 100, {"data": "should not persist"})
        assert cache.get("m", "prompt", None, 0.7, 100) is None

    def test_different_prompts_different_keys(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        cache.put("m", "a", None, 0.0, 100, {"answer": "alpha"})
        cache.put("m", "b", None, 0.0, 100, {"answer": "beta"})
        assert cache.get("m", "a", None, 0.0, 100)["answer"] == "alpha"
        assert cache.get("m", "b", None, 0.0, 100)["answer"] == "beta"

    def test_different_stop_sequences_different_keys(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        resp_a = {"answer": "with_stop_a"}
        resp_b = {"answer": "with_stop_b"}
        cache.put("m", "prompt", None, 0.0, 100, resp_a, stop=["END"])
        cache.put("m", "prompt", None, 0.0, 100, resp_b, stop=["STOP"])
        assert cache.get("m", "prompt", None, 0.0, 100, stop=["END"])["answer"] == "with_stop_a"
        assert cache.get("m", "prompt", None, 0.0, 100, stop=["STOP"])["answer"] == "with_stop_b"
        assert cache.get("m", "prompt", None, 0.0, 100, stop=None) is None

    def test_different_penalties_different_keys(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        resp_a = {"answer": "penalty_a"}
        resp_b = {"answer": "penalty_b"}
        cache.put("m", "prompt", None, 0.0, 100, resp_a, frequency_penalty=0.5)
        cache.put("m", "prompt", None, 0.0, 100, resp_b, frequency_penalty=1.0)
        assert cache.get("m", "prompt", None, 0.0, 100, frequency_penalty=0.5)["answer"] == "penalty_a"
        assert cache.get("m", "prompt", None, 0.0, 100, frequency_penalty=1.0)["answer"] == "penalty_b"

    def test_stop_list_order_does_not_matter(self, tmp_path):
        cache = ResponseCache(tmp_path / "cache")
        resp = {"answer": "ordered"}
        cache.put("m", "prompt", None, 0.0, 100, resp, stop=["A", "B"])
        assert cache.get("m", "prompt", None, 0.0, 100, stop=["A", "B"]) is not None
        assert cache.get("m", "prompt", None, 0.0, 100, stop=["B", "A"]) is not None
