# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for seed_cache_dir functionality in CachingInterceptor."""

import pytest

from nemo_evaluator.adapters.caching.diskcaching import Cache
from nemo_evaluator.adapters.interceptors.caching_interceptor import CachingInterceptor


def _make_cache_key(data: dict) -> str:
    """Generate cache key the same way the interceptor does."""
    return CachingInterceptor._generate_cache_key(data)


def _populate_seed_cache(seed_dir: str, entries: dict[str, tuple[bytes, dict]]) -> None:
    """Populate a seed cache directory with test entries.

    Args:
        seed_dir: Root seed cache directory.
        entries: Mapping of cache_key -> (response_content, headers_dict).
    """
    responses_cache = Cache(directory=f"{seed_dir}/responses")
    headers_cache = Cache(directory=f"{seed_dir}/headers")
    for key, (content, headers) in entries.items():
        responses_cache[key] = content
        headers_cache[key] = headers
    responses_cache.close()
    headers_cache.close()


class TestSeedCacheFallback:
    """Test that seed cache is used as a read-only fallback."""

    def test_seed_cache_hit_on_primary_miss(self, tmp_path):
        """Primary cache miss + seed cache hit → returns seed data."""
        primary_dir = str(tmp_path / "primary")
        seed_dir = str(tmp_path / "seed")

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        cache_key = _make_cache_key(request_data)
        seed_content = b'{"choices": [{"message": {"content": "from seed"}}]}'
        seed_headers = {"content-type": "application/json"}

        _populate_seed_cache(seed_dir, {cache_key: (seed_content, seed_headers)})

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=seed_dir,
                reuse_cached_responses=True,
            )
        )

        result = interceptor._get_from_cache(cache_key)
        assert result is not None
        content, headers = result
        assert content == seed_content
        assert headers == seed_headers

    def test_primary_cache_takes_precedence(self, tmp_path):
        """Primary cache hit → returns primary data, seed not used."""
        primary_dir = str(tmp_path / "primary")
        seed_dir = str(tmp_path / "seed")

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        cache_key = _make_cache_key(request_data)

        primary_content = b'{"choices": [{"message": {"content": "from primary"}}]}'
        primary_headers = {"content-type": "application/json", "x-source": "primary"}
        seed_content = b'{"choices": [{"message": {"content": "from seed"}}]}'
        seed_headers = {"content-type": "application/json", "x-source": "seed"}

        _populate_seed_cache(seed_dir, {cache_key: (seed_content, seed_headers)})

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=seed_dir,
                reuse_cached_responses=True,
            )
        )

        # Populate primary cache directly
        interceptor.responses_cache[cache_key] = primary_content
        interceptor.headers_cache[cache_key] = primary_headers

        result = interceptor._get_from_cache(cache_key)
        assert result is not None
        content, headers = result
        assert content == primary_content
        assert headers == primary_headers

    def test_both_caches_miss(self, tmp_path):
        """Both primary and seed miss → returns None."""
        primary_dir = str(tmp_path / "primary")
        seed_dir = str(tmp_path / "seed")

        # Create empty seed cache directories
        _populate_seed_cache(seed_dir, {})

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=seed_dir,
                reuse_cached_responses=True,
            )
        )

        cache_key = _make_cache_key({"model": "test", "messages": []})
        result = interceptor._get_from_cache(cache_key)
        assert result is None

    def test_no_seed_cache_configured(self, tmp_path):
        """seed_cache_dir=None → only primary cache checked."""
        primary_dir = str(tmp_path / "primary")

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=None,
                reuse_cached_responses=True,
            )
        )

        assert interceptor.seed_responses_cache is None
        assert interceptor.seed_headers_cache is None

        cache_key = _make_cache_key({"model": "test", "messages": []})
        result = interceptor._get_from_cache(cache_key)
        assert result is None

    def test_seed_cache_nonexistent_dir(self, tmp_path):
        """seed_cache_dir points to nonexistent dir → no crash, seed ignored."""
        primary_dir = str(tmp_path / "primary")
        seed_dir = str(tmp_path / "nonexistent_seed")

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=seed_dir,
                reuse_cached_responses=True,
            )
        )

        assert interceptor.seed_responses_cache is None
        assert interceptor.seed_headers_cache is None

    def test_new_responses_saved_to_primary_only(self, tmp_path):
        """New cache entries go to primary cache, not seed."""
        primary_dir = str(tmp_path / "primary")
        seed_dir = str(tmp_path / "seed")

        _populate_seed_cache(seed_dir, {})

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=seed_dir,
                reuse_cached_responses=True,
                save_responses=True,
            )
        )

        cache_key = _make_cache_key(
            {"model": "test", "messages": [{"role": "user", "content": "new"}]}
        )
        new_content = b'{"choices": [{"message": {"content": "new response"}}]}'
        new_headers = {"content-type": "application/json"}

        interceptor._save_to_cache(cache_key, new_content, new_headers)

        # Verify it's in primary
        assert interceptor.responses_cache[cache_key] == new_content
        assert interceptor.headers_cache[cache_key] == new_headers

        # Verify it's NOT in seed
        with pytest.raises(KeyError):
            _ = interceptor.seed_responses_cache[cache_key]
        with pytest.raises(KeyError):
            _ = interceptor.seed_headers_cache[cache_key]

    def test_seed_cache_not_modified(self, tmp_path):
        """Verify seed cache is never written to during normal operation."""
        primary_dir = str(tmp_path / "primary")
        seed_dir = str(tmp_path / "seed")

        original_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        cache_key = _make_cache_key(original_data)
        seed_content = b'{"choices": [{"message": {"content": "original seed"}}]}'
        seed_headers = {"content-type": "application/json"}

        _populate_seed_cache(seed_dir, {cache_key: (seed_content, seed_headers)})

        interceptor = CachingInterceptor(
            CachingInterceptor.Params(
                cache_dir=primary_dir,
                seed_cache_dir=seed_dir,
                reuse_cached_responses=True,
                save_responses=True,
            )
        )

        # Read from seed (populates nothing in seed)
        result = interceptor._get_from_cache(cache_key)
        assert result is not None

        # Save a different entry
        other_key = _make_cache_key(
            {"model": "test", "messages": [{"role": "user", "content": "other"}]}
        )
        interceptor._save_to_cache(other_key, b"other", {"x": "y"})

        # Verify seed still has exactly the original entry and nothing else
        assert interceptor.seed_responses_cache[cache_key] == seed_content
        with pytest.raises(KeyError):
            _ = interceptor.seed_responses_cache[other_key]


class TestSeedCacheLegacyConfig:
    """Test that seed_cache_dir is passed through legacy config conversion."""

    def test_seed_cache_dir_in_legacy_config(self):
        """seed_cache_dir flows from legacy config to interceptor config."""
        from nemo_evaluator.adapters.adapter_config import AdapterConfig

        legacy = {"seed_cache_dir": "/path/to/seed/cache"}
        config = AdapterConfig.from_legacy_config(legacy)

        caching_configs = [ic for ic in config.interceptors if ic.name == "caching"]
        assert len(caching_configs) == 1
        assert caching_configs[0].config["seed_cache_dir"] == "/path/to/seed/cache"

    def test_no_seed_cache_dir_in_legacy_config(self):
        """When seed_cache_dir is not set, it's absent from interceptor config."""
        from nemo_evaluator.adapters.adapter_config import AdapterConfig

        config = AdapterConfig.from_legacy_config({})

        caching_configs = [ic for ic in config.interceptors if ic.name == "caching"]
        assert len(caching_configs) == 1
        assert "seed_cache_dir" not in caching_configs[0].config
