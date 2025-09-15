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


import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from nemo_evaluator.api.api_dataclasses import EvaluationResult
from nemo_evaluator.logging.utils import logger


def get_token_usage_from_cache_db(cache_db_path: str | Path) -> dict:
    """Extract token usage metrics from the cache database.

    Args:
        cache_db_path: Path to the SQLite cache database file

    Returns:
        Dictionary containing token usage metrics:
        {
            "total_tokens": int,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_cached_requests": int
        }
        Returns empty dict if no data found or error occurs.
    """
    try:
        with sqlite3.connect(cache_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    SUM(json_extract(value, '$.usage.total_tokens')) as total_tokens,
                    SUM(json_extract(value, '$.usage.prompt_tokens')) as prompt_tokens,
                    SUM(json_extract(value, '$.usage.completion_tokens')) as completion_tokens,
                    COUNT(*) as total_requests
                FROM cache
                WHERE json_extract(value, '$.usage') IS NOT NULL
            """
            )
            row = cursor.fetchone()

            if row and row[0] is not None:  # Check if we got any results
                return {
                    "total_tokens": row[0],
                    "prompt_tokens": row[1],
                    "completion_tokens": row[2],
                    "total_cached_requests": row[3],
                }
    except Exception as e:
        logger.warning(f"Failed to read token usage from cache: {e}")

    return {}


def get_token_usage_from_cache(cache_dir: str) -> dict:
    """Extract token usage metrics from the cache database."""
    cache_db_path = Path(cache_dir) / "responses" / "cache.db"
    if not cache_db_path.exists():
        return {}

    return get_token_usage_from_cache_db(cache_db_path)


def monitor_memory_usage(
    func, *args, interval_ms, cache_dir: str | None = None, **kwargs
) -> tuple[EvaluationResult, dict[str, Any]]:
    """
    Run func(*args, **kwargs) while polling RSS via psutil.
    Returns (func_return_value, peak_rss_bytes, peak_tree_rss_bytes) where:
    - peak_rss_bytes: peak memory usage of the main process
    - peak_tree_rss_bytes: peak memory usage of the entire process tree (main + children)
    """
    proc = psutil.Process(os.getpid())
    peak = 0
    peak_tree = 0
    stop = False
    ret = None

    def get_tree_memory(process):
        """Get total memory usage of process and all its children."""
        try:
            memory = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    memory += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return memory
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def sampler():
        nonlocal peak, peak_tree
        while not stop:
            # Get memory for current process
            rss = proc.memory_info().rss
            peak = max(peak, rss)

            # Get memory for entire process tree
            tree_rss = get_tree_memory(proc)
            peak_tree = max(peak_tree, tree_rss)

            time.sleep(interval_ms / 1000.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    start_time = time.time()

    try:
        # Filter out cache_dir from kwargs since the target function doesn't expect it
        func_kwargs = {k: v for k, v in kwargs.items() if k != "cache_dir"}
        ret = func(*args, **func_kwargs)
    finally:
        stop = True  # thread safe
        th.join()
    end_time = time.time()

    runtime_seconds = end_time - start_time

    # Get token usage from cache if cache_dir is provided
    token_usage = None
    if cache_dir:
        try:
            token_usage = get_token_usage_from_cache(cache_dir)
        except Exception as e:
            logger.warning(f"Failed to get token usage from cache: {e}")

    metrics = {
        "runtime_seconds": runtime_seconds,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(start_time)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(end_time)),
        "token_usage": token_usage,
        "peak_memory_bytes": peak,  # Memory of main process
        "peak_tree_memory_bytes": peak_tree,  # Memory of entire process tree
    }

    return ret, metrics
