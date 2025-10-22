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

"""
Benchmark script for ResponseStatsInterceptor to identify performance bottlenecks.

Usage:
    python benchmark_response_stats.py [--num-responses 1000] [--profile] [--threads 1]

This script measures:
- Overall intercept_response performance
- Time spent in each method (lock acquisition, JSON parsing, disk I/O, etc.)
- Memory usage
- Thread contention analysis
"""

import argparse
import cProfile
import io
import json
import pstats
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nemo_evaluator.adapters.interceptors.response_stats_interceptor import (
    ResponseStatsInterceptor,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)


@dataclass
class TimingResult:
    """Store timing results for a specific operation."""

    operation: str
    total_time: float
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    percentage: float


class PerformanceProfiler:
    """Profile performance of ResponseStatsInterceptor methods."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure execution time of an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self.lock:
                if operation not in self.timings:
                    self.timings[operation] = []
                self.timings[operation].append(elapsed)

    def get_results(self, total_time: float) -> List[TimingResult]:
        """Get timing results for all operations."""
        results = []
        for operation, times in self.timings.items():
            total = sum(times)
            results.append(
                TimingResult(
                    operation=operation,
                    total_time=total,
                    call_count=len(times),
                    avg_time=total / len(times) if times else 0,
                    min_time=min(times) if times else 0,
                    max_time=max(times) if times else 0,
                    percentage=(total / total_time * 100) if total_time > 0 else 0,
                )
            )
        return sorted(results, key=lambda x: x.total_time, reverse=True)

    def print_results(self, total_time: float):
        """Print formatted results."""
        results = self.get_results(total_time)

        print("\n" + "=" * 100)
        print("PERFORMANCE PROFILE RESULTS")
        print("=" * 100)
        print(
            f"{'Operation':<40} {'Total (s)':<12} {'Calls':<10} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'%':<8}"
        )
        print("-" * 100)

        for result in results:
            print(
                f"{result.operation:<40} "
                f"{result.total_time:<12.6f} "
                f"{result.call_count:<10} "
                f"{result.avg_time * 1000:<12.6f} "
                f"{result.min_time * 1000:<12.6f} "
                f"{result.max_time * 1000:<12.6f} "
                f"{result.percentage:<8.2f}"
            )

        print("-" * 100)
        print(f"{'TOTAL':<40} {total_time:<12.6f}")
        print("=" * 100)


class InstrumentedResponseStatsInterceptor(ResponseStatsInterceptor):
    """Instrumented version of ResponseStatsInterceptor for performance profiling."""

    def __init__(self, params, profiler: PerformanceProfiler):
        self.profiler = profiler
        super().__init__(params)

    def _update_time_tracking(self, current_time: float) -> None:
        with self.profiler.measure("_update_time_tracking"):
            super()._update_time_tracking(current_time)

    def _update_basic_stats(self, resp: AdapterResponse, current_time: float) -> None:
        with self.profiler.measure("_update_basic_stats"):
            super()._update_basic_stats(resp, current_time)

    def _add_basic_response_stats(
        self, adapter_response, context: AdapterGlobalContext
    ) -> None:
        with self.profiler.measure("_add_basic_response_stats"):
            super()._add_basic_response_stats(adapter_response, context)

    def _extract_detailed_response_stats(self, response_data: dict) -> dict:
        with self.profiler.measure("_extract_detailed_response_stats"):
            return super()._extract_detailed_response_stats(response_data)

    def _update_response_stats(self, individual_stats: dict) -> None:
        with self.profiler.measure("_update_response_stats"):
            super()._update_response_stats(individual_stats)

    def _cache_request_stats(self, request_id: str, stats: dict) -> None:
        with self.profiler.measure("_cache_request_stats"):
            super()._cache_request_stats(request_id, stats)

    def _save_aggregated_stats_to_cache(self) -> None:
        with self.profiler.measure("_save_aggregated_stats_to_cache"):
            super()._save_aggregated_stats_to_cache()

    def _load_interceptor_state(self) -> dict:
        with self.profiler.measure("_load_interceptor_state"):
            return super()._load_interceptor_state()

    def _save_interceptor_state(self, state: dict) -> None:
        with self.profiler.measure("_save_interceptor_state"):
            super()._save_interceptor_state(state)

    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        with self.profiler.measure("intercept_response_total"):
            # Measure JSON parsing separately
            original_json = resp.r.json

            def instrumented_json():
                with self.profiler.measure("json_parsing"):
                    return original_json()

            resp.r.json = instrumented_json

            return super().intercept_response(resp, context)


def create_mock_response(
    status_code: int = 200,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    include_tool_calls: bool = False,
) -> AdapterResponse:
    """Create a mock AdapterResponse for testing."""
    mock_resp = Mock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}

    response_data = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens": completion_tokens,
        },
        "choices": [{"finish_reason": "stop", "message": {}}],
    }

    if include_tool_calls:
        response_data["choices"][0]["message"]["tool_calls"] = [
            {"type": "function", "function": {"name": "test_function"}}
        ]

    mock_resp.json = Mock(return_value=response_data)

    return AdapterResponse(
        r=mock_resp,
        rctx=AdapterRequestContext(request_id=f"req_{time.time_ns()}"),
        latency_ms=100.0,
    )


def create_real_response(
    endpoint_url: str, request_id: str, prompt: str = "Hi"
) -> AdapterResponse:
    """Create an AdapterResponse from a real API call."""
    start_time = time.perf_counter()

    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(
        endpoint_url, json=payload, headers={"Content-Type": "application/json"}
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    return AdapterResponse(
        r=resp, rctx=AdapterRequestContext(request_id=request_id), latency_ms=latency_ms
    )


def benchmark_single_threaded(
    num_responses: int,
    cache_dir: Path,
    save_individuals: bool = True,
    endpoint_url: str = None,
) -> tuple[float, PerformanceProfiler]:
    """Benchmark single-threaded performance."""
    profiler = PerformanceProfiler()

    params = ResponseStatsInterceptor.Params(
        cache_dir=str(cache_dir),
        save_individuals=save_individuals,
        stats_file_saving_interval=None,  # Disable periodic file saving
        logging_aggregated_stats_interval=10000,  # Disable periodic logging
    )

    interceptor = InstrumentedResponseStatsInterceptor(params, profiler)
    api_url = endpoint_url or "http://test.api.com/v1/chat/completions"
    context = AdapterGlobalContext(output_dir=str(cache_dir), url=api_url)

    start_time = time.perf_counter()

    for i in range(num_responses):
        if endpoint_url:
            response = create_real_response(endpoint_url, f"req_{i}", prompt="Hi")
        else:
            response = create_mock_response(
                prompt_tokens=50 + (i % 100),
                completion_tokens=25 + (i % 50),
                include_tool_calls=(i % 5 == 0),
            )
        interceptor.intercept_response(response, context)

    elapsed = time.perf_counter() - start_time

    return elapsed, profiler


def benchmark_multi_threaded(
    num_responses: int,
    cache_dir: Path,
    num_threads: int,
    save_individuals: bool = True,
    endpoint_url: str = None,
) -> tuple[float, PerformanceProfiler]:
    """Benchmark multi-threaded performance."""
    profiler = PerformanceProfiler()

    params = ResponseStatsInterceptor.Params(
        cache_dir=str(cache_dir),
        save_individuals=save_individuals,
        stats_file_saving_interval=None,
        logging_aggregated_stats_interval=10000,
    )

    interceptor = InstrumentedResponseStatsInterceptor(params, profiler)
    api_url = endpoint_url or "http://test.api.com/v1/chat/completions"
    context = AdapterGlobalContext(output_dir=str(cache_dir), url=api_url)

    def process_responses(thread_id: int, responses_per_thread: int):
        for i in range(responses_per_thread):
            if endpoint_url:
                response = create_real_response(
                    endpoint_url, f"req_t{thread_id}_{i}", prompt="Hi"
                )
            else:
                response = create_mock_response(
                    prompt_tokens=50 + (i % 100),
                    completion_tokens=25 + (i % 50),
                    include_tool_calls=(i % 5 == 0),
                )
            interceptor.intercept_response(response, context)

    start_time = time.perf_counter()

    responses_per_thread = num_responses // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_responses, i, responses_per_thread)
            for i in range(num_threads)
        ]
        for future in futures:
            future.result()

    elapsed = time.perf_counter() - start_time

    return elapsed, profiler


def benchmark_with_cprofile(num_responses: int, cache_dir: Path):
    """Benchmark using cProfile for detailed call statistics."""
    params = ResponseStatsInterceptor.Params(
        cache_dir=str(cache_dir),
        save_individuals=True,
        stats_file_saving_interval=None,
        logging_aggregated_stats_interval=10000,
    )

    interceptor = ResponseStatsInterceptor(params)
    context = AdapterGlobalContext(
        output_dir=str(cache_dir), url="http://test.api.com/v1/chat/completions"
    )

    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(num_responses):
        response = create_mock_response(
            prompt_tokens=50 + (i % 100),
            completion_tokens=25 + (i % 50),
            include_tool_calls=(i % 5 == 0),
        )
        interceptor.intercept_response(response, context)

    profiler.disable()

    # Print results
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(50)  # Top 50 functions

    print("\n" + "=" * 100)
    print("CPROFILE RESULTS (Top 50 functions by cumulative time)")
    print("=" * 100)
    print(s.getvalue())


def benchmark_cache_operations(num_operations: int, cache_dir: Path):
    """Benchmark cache operations separately."""
    from nemo_evaluator.adapters.caching.diskcaching import Cache

    cache = Cache(cache_dir)
    profiler = PerformanceProfiler()

    # Benchmark cache writes
    test_data = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "status_code": 200,
        "timestamp": time.time(),
    }
    test_json = json.dumps(test_data)

    start = time.perf_counter()
    for i in range(num_operations):
        with profiler.measure("cache_set"):
            cache.set(f"key_{i}", test_json)
    write_time = time.perf_counter() - start

    # Benchmark cache reads
    start = time.perf_counter()
    for i in range(num_operations):
        with profiler.measure("cache_get"):
            cache.get(f"key_{i}")
    read_time = time.perf_counter() - start

    print("\n" + "=" * 100)
    print("CACHE OPERATIONS BENCHMARK")
    print("=" * 100)
    print(f"Operations: {num_operations}")
    print(f"Cache writes: {write_time:.6f}s ({num_operations / write_time:.2f} ops/s)")
    print(f"Cache reads:  {read_time:.6f}s ({num_operations / read_time:.2f} ops/s)")
    print(
        f"Avg write time: {write_time / num_operations * 1000:.6f}ms per operation"
    )
    print(f"Avg read time: {read_time / num_operations * 1000:.6f}ms per operation")
    print("=" * 100)


def analyze_lock_contention(
    num_responses: int, cache_dir: Path, num_threads: int, endpoint_url: str = None
):
    """Analyze lock contention with different thread counts."""
    print("\n" + "=" * 100)
    print("LOCK CONTENTION ANALYSIS")
    print("=" * 100)

    results = []
    for threads in range(1, num_threads + 1):
        elapsed, profiler = benchmark_multi_threaded(
            num_responses, cache_dir, threads, save_individuals=False, endpoint_url=endpoint_url
        )
        throughput = num_responses / elapsed
        results.append((threads, elapsed, throughput))
        print(
            f"Threads: {threads:2d} | Time: {elapsed:8.4f}s | Throughput: {throughput:8.2f} req/s | Speedup: {results[0][1]/elapsed:.2f}x"
        )

    print("=" * 100)
    print("\nAnalysis:")
    if len(results) > 1:
        speedup = results[0][1] / results[-1][1]
        ideal_speedup = num_threads
        efficiency = (speedup / ideal_speedup) * 100
        print(
            f"Speedup from 1 to {num_threads} threads: {speedup:.2f}x (ideal: {ideal_speedup}x)"
        )
        print(f"Parallel efficiency: {efficiency:.1f}%")
        if efficiency < 80:
            print("⚠️  Low efficiency suggests significant lock contention!")
        else:
            print("✓ Good parallel efficiency")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ResponseStatsInterceptor performance"
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=1000,
        help="Number of responses to process (default: 1000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for multi-threaded benchmark (default: 4)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Real endpoint URL to test (e.g., http://localhost:8000/v1/chat/completions). If not provided, uses mock responses.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run detailed cProfile analysis",
    )
    parser.add_argument(
        "--cache-benchmark",
        action="store_true",
        help="Run cache operations benchmark",
    )
    parser.add_argument(
        "--contention-analysis",
        action="store_true",
        help="Analyze lock contention with different thread counts",
    )
    parser.add_argument(
        "--no-individuals",
        action="store_true",
        help="Disable individual request caching",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("RESPONSE STATS INTERCEPTOR BENCHMARK")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Responses: {args.num_responses}")
    print(f"  Threads: {args.threads}")
    print(f"  Endpoint: {args.endpoint or 'Mock responses'}")
    print(f"  Save individuals: {not args.no_individuals}")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"

        # Single-threaded benchmark
        print("\n[1/4] Running single-threaded benchmark...")
        elapsed, profiler = benchmark_single_threaded(
            args.num_responses,
            cache_dir,
            save_individuals=not args.no_individuals,
            endpoint_url=args.endpoint,
        )
        print(f"\nSingle-threaded results:")
        print(f"  Total time: {elapsed:.6f}s")
        print(f"  Throughput: {args.num_responses / elapsed:.2f} requests/second")
        print(
            f"  Average time per request: {elapsed / args.num_responses * 1000:.6f}ms"
        )
        profiler.print_results(elapsed)

        # Multi-threaded benchmark
        cache_dir = Path(tmpdir) / "cache_mt"
        print(f"\n[2/4] Running multi-threaded benchmark ({args.threads} threads)...")
        elapsed_mt, profiler_mt = benchmark_multi_threaded(
            args.num_responses,
            cache_dir,
            args.threads,
            save_individuals=not args.no_individuals,
            endpoint_url=args.endpoint,
        )
        print(f"\nMulti-threaded results ({args.threads} threads):")
        print(f"  Total time: {elapsed_mt:.6f}s")
        print(f"  Throughput: {args.num_responses / elapsed_mt:.2f} requests/second")
        print(
            f"  Average time per request: {elapsed_mt / args.num_responses * 1000:.6f}ms"
        )
        print(f"  Speedup vs single-threaded: {elapsed / elapsed_mt:.2f}x")
        profiler_mt.print_results(elapsed_mt)

        # Cache operations benchmark
        if args.cache_benchmark:
            print("\n[3/4] Running cache operations benchmark...")
            cache_dir = Path(tmpdir) / "cache_bench"
            benchmark_cache_operations(args.num_responses, cache_dir)
        else:
            print("\n[3/4] Skipping cache benchmark (use --cache-benchmark to enable)")

        # Lock contention analysis
        if args.contention_analysis:
            print("\n[4/4] Running lock contention analysis...")
            cache_dir = Path(tmpdir) / "cache_contention"
            analyze_lock_contention(
                args.num_responses, cache_dir, args.threads, endpoint_url=args.endpoint
            )
        else:
            print(
                "\n[4/4] Skipping contention analysis (use --contention-analysis to enable)"
            )

        # Detailed cProfile
        if args.profile:
            print("\nRunning detailed cProfile analysis...")
            cache_dir = Path(tmpdir) / "cache_profile"
            benchmark_with_cprofile(min(args.num_responses, 100), cache_dir)

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)
    print("\nKey findings:")
    print(
        f"1. Single-threaded throughput: {args.num_responses / elapsed:.2f} req/s"
    )
    print(
        f"2. Multi-threaded throughput: {args.num_responses / elapsed_mt:.2f} req/s ({args.threads} threads)"
    )
    print(f"3. Speedup: {elapsed / elapsed_mt:.2f}x")
    print(
        "\nUse --profile for detailed function-level profiling"
    )
    print("Use --cache-benchmark to analyze cache performance")
    print("Use --contention-analysis to analyze lock contention")


if __name__ == "__main__":
    main()

