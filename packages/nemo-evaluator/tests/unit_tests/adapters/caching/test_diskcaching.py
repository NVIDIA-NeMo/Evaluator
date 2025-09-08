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

import json
import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from nemo_evaluator.adapters.caching.diskcaching import Cache


@pytest.fixture
def cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        import shutil

        shutil.rmtree(temp_dir)


def test_basic_operations(cache_dir):
    """Test basic cache operations."""
    cache = Cache(directory=cache_dir)
    try:
        # Test setting and getting items
        cache["key1"] = "value1"
        cache["key2"] = "value2"

        assert cache["key1"] == "value1"
        assert cache["key2"] == "value2"

        # Test length
        assert len(cache) == 2

        # Test contains
        assert "key1" in cache
        assert "nonexistent" not in cache

        # Test deletion
        del cache["key1"]
        assert "key1" not in cache
        assert len(cache) == 1
    finally:
        cache.close()


def test_persistence(cache_dir):
    """Test data persistence."""
    # Write data
    cache1 = Cache(directory=cache_dir)
    try:
        cache1["test"] = "value"
    finally:
        cache1.close()

    # Verify data persists after closing and reopening
    cache2 = Cache(directory=cache_dir)
    try:
        assert cache2["test"] == "value"
    finally:
        cache2.close()


@pytest.mark.parametrize(
    "test_data",
    [
        {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
            "boolean": True,
            "none": None,
        }
    ],
)
def test_different_types(cache_dir, test_data):
    """Test storing different types of data."""
    # Given
    cache = Cache(directory=cache_dir)

    try:
        # When
        for key, value in test_data.items():
            cache[key] = value

        # Then
        for key, value in test_data.items():
            cached_value = cache[key]
            if isinstance(value, dict):
                assert json.dumps(cached_value, sort_keys=True) == json.dumps(
                    value, sort_keys=True
                )
            else:
                assert cached_value == value
    finally:
        cache.close()


@pytest.mark.parametrize("test_data", [{"a": 1, "b": 2, "c": 3}])
def test_iteration(cache_dir, test_data):
    """Test cache iteration methods."""
    # Given
    cache = Cache(directory=cache_dir)

    try:
        # When
        for key, value in test_data.items():
            cache[key] = value

        # Then
        assert set(cache) == set(test_data.keys())

        # And
        cached_values = {key: cache[key] for key in cache}
        assert cached_values == test_data
    finally:
        cache.close()


def test_thread_safety(cache_dir):
    """Test concurrent access from multiple threads."""
    num_threads = 10
    operations_per_thread = 100
    cache = Cache(directory=cache_dir)

    def worker(thread_id):
        """Worker function that performs random read/write operations."""
        results = []
        for i in range(operations_per_thread):
            op = random.choice(["read", "write"])
            key = f"key_{thread_id}_{i}"
            value = f"value_{thread_id}_{i}"

            try:
                if op == "write":
                    cache[key] = value
                    results.append(("write", key, value, True))
                else:
                    # Try to read a random key
                    read_thread = random.randint(0, thread_id)
                    read_op = random.randint(0, i)
                    read_key = f"key_{read_thread}_{read_op}"
                    try:
                        value = cache[read_key]
                        results.append(("read", read_key, value, True))
                    except KeyError:
                        results.append(("read", read_key, None, False))
            except Exception as e:
                results.append(("error", key, str(e), False))
        return results

    try:
        # Run operations in multiple threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_thread = {
                executor.submit(worker, thread_id): thread_id
                for thread_id in range(num_threads)
            }

            # Collect results
            all_results = []
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    assert False, f"Thread {thread_id} failed: {str(e)}"

        # Verify results
        write_ops = [r for r in all_results if r[0] == "write"]
        assert all(r[3] for r in write_ops), "Some write operations failed"

        for op, key, value, success in write_ops:
            assert cache[key] == value, f"Value mismatch for key {key}"
    finally:
        cache.close()


def test_concurrent_updates(cache_dir):
    """Test concurrent updates to the same keys."""
    num_threads = 5
    updates_per_thread = 50
    test_keys = ["shared_key_1", "shared_key_2", "shared_key_3"]
    cache = Cache(directory=cache_dir)
    threads = []

    def worker(thread_id):
        for _ in range(updates_per_thread):
            key = random.choice(test_keys)
            value = f"thread_{thread_id}_update_{time.time()}"
            cache[key] = value
            time.sleep(0.001)  # Small sleep to increase chance of thread interleaving

    try:
        # Create and start threads
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify final state
        for key in test_keys:
            assert key in cache
            value = cache[key]
            assert isinstance(value, str)
            assert value.startswith("thread_")

        assert len(cache) == len(test_keys)
    finally:
        cache.close()


def test_concurrent_read_write(cache_dir):
    """Test concurrent reads while writing is in progress."""
    num_threads = 8
    operations_per_thread = 50
    cache = Cache(directory=cache_dir)
    threads = []

    def writer(thread_id):
        for i in range(operations_per_thread):
            key = f"key_{thread_id}_{i}"
            value = f"value_{time.time()}"
            cache[key] = value
            time.sleep(0.002)

    def reader():
        for _ in range(operations_per_thread):
            try:
                keys = list(cache)
                for key in keys:
                    _ = cache[key]
                time.sleep(0.001)
            except Exception as e:
                assert False, f"Reader failed: {str(e)}"

    try:
        # Create writer threads
        for i in range(num_threads // 2):
            thread = threading.Thread(target=writer, args=(i,))
            threads.append(thread)
            thread.start()

        # Create reader threads
        for i in range(num_threads // 2):
            thread = threading.Thread(target=reader)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    finally:
        cache.close()


@pytest.mark.parametrize(
    "test_cases",
    [
        [
            (b'{"result": "test"}', "JSON bytes"),
            (b"\x00\x01\x02\x03", "Raw binary data"),
            ("Hello".encode(), "UTF-8 encoded string"),
            (b"\xff\xfe\xfd\xfc", "Binary with special bytes"),
            (b"a" * 1024, "Large binary content"),
        ]
    ],
)
def test_raw_bytes_storage(cache_dir, test_cases):
    """Test storing and retrieving raw bytes."""
    # Given
    cache = Cache(directory=cache_dir)

    try:
        for content, desc in test_cases:
            # When
            key = f"binary_{desc}"
            cache[key] = content

            # Then
            retrieved = cache[key]
            assert isinstance(retrieved, bytes), (
                f"Retrieved content should be bytes for {desc}"
            )
            assert retrieved == content, f"Content mismatch for {desc}"

            # And
            assert len(retrieved) == len(content), f"Length mismatch for {desc}"
            for b1, b2 in zip(retrieved, content):
                assert b1 == b2, f"Byte mismatch in {desc}"
    finally:
        cache.close()


@pytest.mark.parametrize(
    "binary_content,json_content", [(b'{"raw": "data"}', ["header1", "header2"])]
)
def test_mixed_storage(cache_dir, binary_content, json_content):
    """Test storing both binary and JSON data."""
    # Given
    cache = Cache(directory=cache_dir)

    try:
        # When
        binary_key = "binary_data"
        cache[binary_key] = binary_content

        json_key = "json_data"
        cache[json_key] = json_content

        # Then
        retrieved_binary = cache[binary_key]
        assert isinstance(retrieved_binary, bytes)
        assert retrieved_binary == binary_content

        # And
        retrieved_json = cache[json_key]
        assert isinstance(retrieved_json, list)
        assert retrieved_json == json_content
    finally:
        cache.close()


@pytest.mark.parametrize("sizes", [[1024, 10 * 1024, 100 * 1024]])  # 1KB, 10KB, 100KB
def test_large_binary_storage(cache_dir, sizes):
    """Test storing and retrieving large binary content."""
    # Given
    cache = Cache(directory=cache_dir)

    try:
        for size in sizes:
            # When
            content = bytes([i % 256 for i in range(size)])
            key = f"large_binary_{size}"
            cache[key] = content

            # Then
            retrieved = cache[key]
            assert isinstance(retrieved, bytes)
            assert len(retrieved) == size
            assert retrieved == content

            # And
            for i, byte in enumerate(retrieved):
                assert byte == (i % 256), (
                    f"Byte mismatch at position {i} for size {size}"
                )
    finally:
        cache.close()
