"""Tests for response caching."""
import json
import tempfile
from pathlib import Path

import pytest
from nemo_evaluator.runner.cache import ResponseCache


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
