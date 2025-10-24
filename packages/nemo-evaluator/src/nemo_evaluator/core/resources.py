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
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from nemo_evaluator.api.api_dataclasses import EvaluationResult
from nemo_evaluator.logging.utils import logger

__all__ = []


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
        logger.warning("Failed to read token usage from cache", error=str(e))

    return {}


def get_token_usage_from_cache(cache_dir: str) -> dict:
    """Extract token usage metrics from the cache database."""
    cache_db_path = Path(cache_dir) / "responses" / "cache.db"
    if not cache_db_path.exists():
        return {}

    return get_token_usage_from_cache_db(cache_db_path)


def aggregate_runtime_metrics(output_dir: str) -> dict[str, Any]:
    """Aggregate all run data from run_times directory."""
    run_times_dir = Path(output_dir) / "run_times"
    aggregated_metrics = {}

    if not run_times_dir.exists():
        return aggregated_metrics

    total_runtime = 0
    earliest_start = None
    latest_end = None
    max_peak_memory = 0
    max_peak_tree_memory = 0
    run_count = 0

    for run_file in run_times_dir.glob("runtime_*.json"):
        try:
            with open(run_file, "r") as f:
                run_data = json.load(f)
                total_runtime += run_data.get("runtime_seconds", 0)
                run_count += 1

                # Track earliest start and latest end
                run_start = run_data.get("start_time", "")
                run_end = run_data.get("end_time", "")
                if earliest_start is None or run_start < earliest_start:
                    earliest_start = run_start
                if latest_end is None or run_end > latest_end:
                    latest_end = run_end

                # Track peak memory across all runs
                max_peak_memory = max(
                    max_peak_memory, run_data.get("peak_memory_bytes", 0)
                )
                max_peak_tree_memory = max(
                    max_peak_tree_memory, run_data.get("peak_tree_memory_bytes", 0)
                )
        except Exception:
            pass

    if run_count > 0:
        aggregated_metrics = {
            "runtime_seconds": total_runtime,
            "start_time": earliest_start,
            "end_time": latest_end,
            "peak_memory_bytes": max_peak_memory,
            "peak_tree_memory_bytes": max_peak_tree_memory,
            "total_runs": run_count,
        }

        # Try to get inference time from response stats and calculate scoring time
        try:
            metrics_file = Path(output_dir) / "eval_factory_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                    response_stats = metrics_data.get("response_stats", {})
                    inference_time = response_stats.get("inference_time", 0.0)

                    # Calculate scoring time as runtime - inference time
                    scoring_time = max(0.0, total_runtime - inference_time)
                    aggregated_metrics["inference_time_seconds"] = inference_time
                    aggregated_metrics["scoring_time_seconds"] = scoring_time
        except Exception as e:
            # If we can't read response stats, just continue without scoring time
            logger.warning(
                "Could not extract inference time from response stats", error=str(e)
            )

    return aggregated_metrics


def _update_persistent_metrics(
    output_dir: str,
    start_time: float,
    peak_memory: int,
    peak_tree_memory: int,
    run_id: str,
) -> None:
    """Save individual run data and update peak memory only."""
    try:
        # Create run_times directory
        run_times_dir = Path(output_dir) / "run_times"
        run_times_dir.mkdir(exist_ok=True)

        # Save individual run runtime
        current_time = time.time()
        current_runtime = current_time - start_time
        run_file = run_times_dir / f"runtime_{run_id}.json"

        with open(run_file, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "start_time": time.strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(start_time)
                    ),
                    "end_time": time.strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time)
                    ),
                    "runtime_seconds": current_runtime,
                    "peak_memory_bytes": peak_memory,
                    "peak_tree_memory_bytes": peak_tree_memory,
                },
                f,
            )

    except Exception as e:
        logger.warning(
            "Failed to update persistent metrics", error=str(e), run_id=run_id
        )


def monitor_memory_usage(
    func,
    *args,
    interval_ms,
    cache_dir: str | None = None,
    output_dir: str | None = None,
    **kwargs,
) -> tuple[EvaluationResult, dict[str, Any]]:
    """
    Run func(*args, **kwargs) while polling RSS via psutil.
    Returns (func_return_value, peak_rss_bytes, peak_tree_rss_bytes) where:
    - peak_rss_bytes: peak memory usage of the main process
    - peak_tree_rss_bytes: peak memory usage of the entire process tree (main + children)
    """
    proc = psutil.Process(os.getpid())

    # Generate meaningful run ID (counter or date)
    if output_dir:
        run_times_dir = Path(output_dir) / "run_times"
        run_times_dir.mkdir(exist_ok=True)
        # Count existing runs to get next ID
        existing_runs = list(run_times_dir.glob("runtime_*.json"))
        run_id = str(len(existing_runs))
    else:
        run_id = "0"

    # Initialize values
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
        last_save_time = 0
        save_interval = 5.0  # Save every 5 seconds

        while not stop:
            # Get memory for current process
            rss = proc.memory_info().rss
            peak = max(peak, rss)

            # Get memory for entire process tree
            tree_rss = get_tree_memory(proc)
            peak_tree = max(peak_tree, tree_rss)

            # Update persistent metrics file if output_dir is provided and enough time has passed
            if output_dir:
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    _update_persistent_metrics(
                        output_dir, start_time, peak, peak_tree, run_id
                    )
                    last_save_time = current_time

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
            logger.warning("Failed to get token usage from cache", error=str(e))

    metrics = {
        "runtime_seconds": runtime_seconds,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(start_time)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(end_time)),
        "token_usage": token_usage,
        "peak_memory_bytes": peak,
        "peak_tree_memory_bytes": peak_tree,
    }

    return ret, metrics
