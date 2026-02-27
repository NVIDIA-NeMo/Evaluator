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

"""Tests for ResponseStatsInterceptor."""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from nemo_evaluator.adapters.interceptors.response_stats_interceptor import (
    ResponseStatsInterceptor,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)


class TestResponseStatsInterceptor:
    """Test ResponseStatsInterceptor functionality."""

    @pytest.fixture
    def interceptor(self, tmp_path):
        """Create a response stats interceptor for testing."""
        cache_dir = tmp_path / "test_cache"
        return ResponseStatsInterceptor(
            ResponseStatsInterceptor.Params(
                save_individuals=False, cache_dir=str(cache_dir)
            )
        )

    @pytest.fixture
    def context(self):
        """Create a mock context for testing."""
        return AdapterGlobalContext(
            output_dir="/tmp/test_output", url="http://test.api.com/v1/chat/completions"
        )

    @pytest.mark.parametrize(
        "params,expected",
        [
            (
                ResponseStatsInterceptor.Params(),
                {
                    "collect_token_stats": True,
                    "collect_finish_reasons": True,
                    "collect_tool_calls": True,
                    "stats_file_saving_interval": None,
                    "save_individuals": True,
                    "has_cache": True,
                },
            ),
            (
                ResponseStatsInterceptor.Params(
                    collect_token_stats=False,
                    collect_finish_reasons=False,
                    collect_tool_calls=False,
                    stats_file_saving_interval=10,
                ),
                {
                    "collect_token_stats": False,
                    "collect_finish_reasons": False,
                    "collect_tool_calls": False,
                    "stats_file_saving_interval": 10,
                    "save_individuals": True,
                    "has_cache": True,
                },
            ),
        ],
    )
    def test_initialization(self, params, expected):
        """Test interceptor initialization with different parameters."""
        interceptor = ResponseStatsInterceptor(params)

        assert interceptor.collect_token_stats == expected["collect_token_stats"]
        assert interceptor.collect_finish_reasons == expected["collect_finish_reasons"]
        assert interceptor.collect_tool_calls == expected["collect_tool_calls"]
        assert (
            interceptor.stats_file_saving_interval
            == expected["stats_file_saving_interval"]
        )
        assert interceptor.save_individuals == expected["save_individuals"]
        assert (interceptor._request_stats_cache is not None) == expected["has_cache"]

        # Check stats structure
        assert "count" in interceptor._stats
        assert "avg_prompt_tokens" in interceptor._stats
        assert "sum_prompt_tokens" in interceptor._stats
        assert "max_prompt_tokens" in interceptor._stats

    def test_add_basic_response_stats(self, interceptor, context):
        """Test basic response stats collection."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200

        adapter_response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(), latency_ms=0.0
        )

        # Get the initial count before adding stats
        initial_count = interceptor._stats["count"]
        initial_status_200_count = interceptor._stats["status_codes"].get(200, 0)

        interceptor._add_basic_response_stats(adapter_response, context)

        # Count should be incremented by 1
        assert interceptor._stats["count"] == initial_count + 1
        assert interceptor._stats["status_codes"][200] == initial_status_200_count + 1

    @pytest.mark.parametrize(
        "status_code,response_data,expected_tokens,expected_finish_reason,expected_tool_calls",
        [
            # Success case with complete data
            (
                200,
                {
                    "usage": {
                        "prompt_tokens": 25,
                        "total_tokens": 35,
                        "completion_tokens": 10,
                    },
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "tool_calls": [
                                    {"type": "function", "function": {"name": "test"}}
                                ]
                            },
                        }
                    ],
                },
                {"avg": 25, "max": 25},
                {"stop": 1},
                {"tool_calls": 1, "function_calls": 0},
            ),
            # Failed response - should not collect detailed stats
            (
                500,
                {
                    "usage": {
                        "prompt_tokens": 25,
                        "total_tokens": 35,
                        "completion_tokens": 10,
                    }
                },
                None,
                {},
                {"tool_calls": 0, "function_calls": 0},
            ),
            # Missing usage field - should default to 0
            (
                200,
                {"choices": [{"finish_reason": "stop", "message": {}}]},
                {"avg": 0, "max": 0},
                {"stop": 1},
                {"tool_calls": 0, "function_calls": 0},
            ),
            # Malformed data - should handle gracefully
            (
                200,
                {
                    "usage": {
                        "prompt_tokens": "invalid",
                        "total_tokens": None,
                        "completion_tokens": [],
                    },
                    "choices": "not_a_list",
                },
                None,
                {},
                {"tool_calls": 0, "function_calls": 0},
            ),
        ],
    )
    def test_detailed_response_stats(
        self,
        interceptor,
        context,
        status_code,
        response_data,
        expected_tokens,
        expected_finish_reason,
        expected_tool_calls,
    ):
        """Test detailed response stats collection for various scenarios."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = status_code
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = response_data

        adapter_response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(), latency_ms=0.0
        )

        initial_count = interceptor._stats["count"]
        _ = interceptor.intercept_response(adapter_response, context)

        # Basic stats should always be updated
        assert interceptor._stats["count"] == initial_count + 1
        assert interceptor._stats["status_codes"][status_code] == 1

        # Token stats depend on success and data validity
        if expected_tokens is None:
            assert interceptor._stats["avg_prompt_tokens"] is None
            assert interceptor._stats["avg_total_tokens"] is None
            assert interceptor._stats["avg_completion_tokens"] is None
        else:
            assert interceptor._stats["avg_prompt_tokens"] == expected_tokens["avg"]
            assert interceptor._stats["max_prompt_tokens"] == expected_tokens["max"]

        # Finish reason and tool calls
        assert interceptor._stats["finish_reason"] == expected_finish_reason
        assert (
            interceptor._stats["tool_calls_count"] == expected_tool_calls["tool_calls"]
        )
        assert (
            interceptor._stats["function_calls_count"]
            == expected_tool_calls["function_calls"]
        )

    @pytest.mark.parametrize(
        "status_code,json_side_effect,expected_status_count,expected_tokens",
        [
            # Success case
            (
                200,
                {
                    "usage": {
                        "prompt_tokens": 30,
                        "total_tokens": 45,
                        "completion_tokens": 15,
                    },
                    "choices": [
                        {"finish_reason": "stop", "message": {"tool_calls": []}}
                    ],
                },
                {200: 1},
                {"avg": 30, "max": 30},
            ),
            # Non-JSON response
            (404, json.JSONDecodeError("Not JSON", "", 0), {404: 1}, None),
            # Exception during JSON parsing
            (200, Exception("Unexpected error"), {200: 1}, None),
        ],
    )
    def test_intercept_response_scenarios(
        self,
        interceptor,
        context,
        status_code,
        json_side_effect,
        expected_status_count,
        expected_tokens,
    ):
        """Test response interception for various scenarios."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = status_code
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}

        if isinstance(json_side_effect, Exception):
            mock_resp.json.side_effect = json_side_effect
        else:
            mock_resp.json.return_value = json_side_effect

        adapter_response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(), latency_ms=0.0
        )
        initial_count = interceptor._stats["count"]
        result = interceptor.intercept_response(adapter_response, context)

        # Should always return the original response
        assert result == adapter_response
        # Count should always be incremented
        assert interceptor._stats["count"] == initial_count + 1

        # Status code should be recorded
        for status, count in expected_status_count.items():
            assert interceptor._stats["status_codes"][status] == count

        # Token stats only for successful JSON responses
        if expected_tokens is None:
            assert interceptor._stats["avg_prompt_tokens"] is None
        else:
            assert interceptor._stats["avg_prompt_tokens"] == expected_tokens["avg"]
            assert interceptor._stats["max_prompt_tokens"] == expected_tokens["max"]

    @pytest.mark.parametrize(
        "interval,file_side_effect,expected_file_exists,expected_count",
        [
            # Normal file saving
            (5, None, True, 3),
            # Interval-based saving
            (3, None, True, 5),
        ],
    )
    def test_file_saving_scenarios(
        self,
        interceptor,
        context,
        tmp_path,
        interval,
        file_side_effect,
        expected_file_exists,
        expected_count,
    ):
        """Test file saving for various scenarios."""
        interceptor.stats_file_saving_interval = interval
        context.output_dir = str(tmp_path)

        # Add some stats
        initial_count = interceptor._stats["count"]
        interceptor._stats["count"] = initial_count + expected_count
        interceptor._stats["avg_prompt_tokens"] = 50

        if file_side_effect:
            with patch("builtins.open", side_effect=file_side_effect):
                # Should not raise an exception, should handle gracefully
                interceptor._save_stats_to_file(context)
        else:
            interceptor._save_stats_to_file(context)

        # Check file existence
        metrics_path = Path(context.output_dir) / "eval_factory_metrics.json"
        if expected_file_exists:
            assert metrics_path.exists(), f"File {metrics_path} was not created"

            # Check file content
            with open(metrics_path, "r") as f:
                saved_stats = json.load(f)

                assert "response_stats" in saved_stats
                assert (
                    saved_stats["response_stats"]["count"]
                    == initial_count + expected_count
                )
                assert saved_stats["response_stats"]["avg_prompt_tokens"] == 50
        else:
            assert not metrics_path.exists()

    @pytest.mark.parametrize(
        "add_stats,existing_metrics,expected_file_exists,expected_count,expected_merge",
        [
            # With stats - should create file
            (True, None, True, 10, False),
            # No stats - should not create file
            (False, None, False, 0, False),
            # With stats and existing metrics - should merge
            (True, {"existing_data": {"value": 42}}, True, 5, True),
        ],
    )
    def test_post_eval_hook_scenarios(
        self,
        interceptor,
        tmp_path,
        add_stats,
        existing_metrics,
        expected_file_exists,
        expected_count,
        expected_merge,
    ):
        """Test post evaluation hook for various scenarios."""
        context = AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

        # Create existing metrics file if specified
        if existing_metrics:
            metrics_path = Path(context.output_dir) / "eval_factory_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(existing_metrics, f)

        if add_stats:
            # Add some test data
            initial_count = interceptor._stats["count"]
            interceptor._stats["count"] = initial_count + expected_count
            interceptor._stats["avg_prompt_tokens"] = 200

        interceptor.post_eval_hook(context)

        # Check file existence
        metrics_path = Path(context.output_dir) / "eval_factory_metrics.json"
        if expected_file_exists:
            assert metrics_path.exists()

            # Check file content
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

                assert "response_stats" in metrics
                assert (
                    metrics["response_stats"]["count"] == initial_count + expected_count
                )
                assert metrics["response_stats"]["avg_prompt_tokens"] == 200

                # Check if existing metrics were preserved
                if expected_merge:
                    assert "existing_data" in metrics
                    assert metrics["existing_data"]["value"] == 42
        else:
            assert not metrics_path.exists()

    def test_thread_safety(self, interceptor, context):
        """Test that stats collection is thread-safe."""
        import threading
        import time

        def process_responses():
            for _ in range(100):
                mock_resp = Mock(spec=requests.Response)
                mock_resp.status_code = 200
                mock_resp.json = Mock(return_value={"choices": []})
                mock_resp.headers = {"date": "2025-08-28T12:00:00Z"}

                adapter_response = AdapterResponse(
                    r=mock_resp,
                    rctx=AdapterRequestContext(request_id=f"test-{_}"),
                    latency_ms=0.0,
                )

                interceptor.intercept_response(adapter_response, context)
                time.sleep(0.001)  # Small delay to increase race condition chance

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=process_responses)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have exactly 500 responses (5 threads * 100 each)
        assert interceptor._stats["count"] >= 500
        assert interceptor._stats["status_codes"][200] >= 500

    @pytest.mark.parametrize(
        "cache_hit,expected_total_responses,expected_successful_responses",
        [
            # Cached response - should NOT be counted (skipped by interceptor)
            (True, 0, 0),
            # Normal response (not cached)
            (False, 1, 1),
        ],
    )
    def test_cached_response_stats_behavior(
        self,
        tmp_path,
        context,
        cache_hit,
        expected_total_responses,
        expected_successful_responses,
    ):
        """Test that cached responses are properly skipped in stats counting."""
        # Create a unique cache directory for each test to avoid state pollution
        import uuid

        unique_cache_dir = tmp_path / f"test_cache_{uuid.uuid4().hex[:8]}"

        interceptor = ResponseStatsInterceptor(
            ResponseStatsInterceptor.Params(
                save_individuals=False, cache_dir=str(unique_cache_dir)
            )
        )

        # Create a mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "test"}
        mock_response._content = b'{"result": "test"}'

        # Setup request context with cache hit status
        mock_rctx = Mock()
        mock_rctx.cache_hit = cache_hit

        adapter_response = AdapterResponse(r=mock_response, rctx=mock_rctx)

        # Process response
        interceptor.intercept_response(adapter_response, context)

        # Verify stats counting behavior
        assert interceptor._stats["count"] == expected_total_responses
        assert interceptor._stats["successful_count"] == expected_successful_responses


    def test_average_precision_with_many_requests(self, tmp_path):
        """Test that averages remain precise over many requests.

        Previously, running averages with round(..., 2) at each step accumulated
        floating-point error. With sum-based computation, the average should be
        exact (within floating-point precision of a single division).
        """
        cache_dir = tmp_path / "test_cache_precision"
        interceptor = ResponseStatsInterceptor(
            ResponseStatsInterceptor.Params(
                save_individuals=False, cache_dir=str(cache_dir)
            )
        )
        context = AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

        # Simulate 10,000 requests with varying token counts
        total_completion = 0
        num_requests = 10000
        for i in range(num_requests):
            completion_tokens = 100 + (i % 997)  # Varying values
            total_completion += completion_tokens

            mock_resp = Mock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.headers = {}
            mock_resp.json.return_value = {
                "usage": {
                    "prompt_tokens": 50,
                    "total_tokens": 50 + completion_tokens,
                    "completion_tokens": completion_tokens,
                },
                "choices": [{"finish_reason": "stop", "message": {}}],
            }
            adapter_response = AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id=f"req_{i}"),
                latency_ms=10.0,
            )
            interceptor.intercept_response(adapter_response, context)

        # Verify exact sum
        assert interceptor._stats["sum_completion_tokens"] == total_completion

        # Verify average is precise (should match exact computation)
        expected_avg = total_completion / num_requests
        assert abs(interceptor._stats["avg_completion_tokens"] - expected_avg) < 1e-10

        # Verify count
        assert interceptor._stats["successful_count"] == num_requests

    def test_total_fields_in_saved_file(self, tmp_path):
        """Test that saved metrics include exact total_* fields."""
        cache_dir = tmp_path / "test_cache_totals"
        interceptor = ResponseStatsInterceptor(
            ResponseStatsInterceptor.Params(
                save_individuals=False, cache_dir=str(cache_dir)
            )
        )
        context = AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

        # Process a few requests
        token_values = [100, 200, 333]
        for i, ct in enumerate(token_values):
            mock_resp = Mock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.headers = {}
            mock_resp.json.return_value = {
                "usage": {
                    "prompt_tokens": 50,
                    "total_tokens": 50 + ct,
                    "completion_tokens": ct,
                },
                "choices": [{"finish_reason": "stop", "message": {}}],
            }
            adapter_response = AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id=f"req_{i}"),
                latency_ms=10.0,
            )
            interceptor.intercept_response(adapter_response, context)

        # Save to file
        interceptor.post_eval_hook(context)

        # Read and verify
        metrics_path = Path(tmp_path) / "eval_factory_metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        rs = metrics["response_stats"]

        # Exact totals should be present
        assert rs["total_completion_tokens"] == sum(token_values)
        assert rs["total_prompt_tokens"] == 50 * len(token_values)

        # Averages should be rounded to 2dp
        expected_avg_ct = round(sum(token_values) / len(token_values), 2)
        assert rs["avg_completion_tokens"] == expected_avg_ct

        # Internal sum_* fields should NOT be in the output
        assert "sum_completion_tokens" not in rs
        assert "sum_prompt_tokens" not in rs
        assert "sum_latency_ms" not in rs


class TestResponseStatsInterceptorCache:
    """Test ResponseStatsInterceptor caching and aggregation functionality."""

    @pytest.mark.parametrize(
        "latency_ms,expected_avg,expected_max",
        [
            (100.0, 100.0, 100.0),
            (None, None, None),
            (0.0, 0.0, 0.0),
            (500.0, 500.0, 500.0),
        ],
    )
    def test_basic_functionality(
        self, tmp_path, latency_ms, expected_avg, expected_max
    ):
        """Test basic interceptor functionality with different latency values."""
        cache_dir = tmp_path / "test_cache"
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=True
        )
        context = AdapterGlobalContext(output_dir=str(tmp_path), url="http://test.com")
        interceptor = ResponseStatsInterceptor(params)

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {"choices": [{"finish_reason": "stop"}]}

        adapter_response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(), latency_ms=latency_ms
        )
        interceptor.intercept_response(adapter_response, context)

        # Verify basic stats
        assert interceptor._stats["count"] == 1
        assert interceptor._stats["successful_count"] == 1
        assert interceptor._stats["status_codes"][200] == 1
        assert interceptor._stats["finish_reason"]["stop"] == 1

        # Verify latency handling
        if latency_ms is None:
            assert interceptor._stats["avg_latency_ms"] is None
            assert interceptor._stats["max_latency_ms"] is None
        else:
            assert interceptor._stats["avg_latency_ms"] == expected_avg
            assert interceptor._stats["max_latency_ms"] == expected_max

    def test_caching_and_aggregation_with_multiple_runs(self, tmp_path):
        """Test comprehensive caching and aggregation across multiple runs with inference_time, count, and latency."""
        cache_dir = tmp_path / "test_cache"
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=True
        )
        context = AdapterGlobalContext(output_dir=str(tmp_path), url="http://test.com")

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {"choices": [{"finish_reason": "stop"}]}

        # Run 1: 2 requests
        interceptor1 = ResponseStatsInterceptor(params)
        interceptor1.intercept_response(
            AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id="req_1"),
                latency_ms=100.0,
            ),
            context,
        )
        time.sleep(0.1)
        interceptor1.intercept_response(
            AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id="req_2"),
                latency_ms=200.0,
            ),
            context,
        )

        # Verify Run 1 stats
        assert interceptor1._stats["count"] == 2
        assert interceptor1._stats["successful_count"] == 2
        assert interceptor1._stats["status_codes"][200] == 2
        assert interceptor1._stats["finish_reason"]["stop"] == 2
        assert interceptor1._stats["run_id"] == 0
        assert interceptor1._stats["avg_latency_ms"] == 150.0  # (100 + 200) / 2
        assert interceptor1._stats["max_latency_ms"] == 200.0
        run1_inference_time = interceptor1._stats["inference_time"]
        # With latency-based estimation: sleep_time + latency_adjustment
        # Expected: ~0.1s (sleep) + ~0.1s (first request latency) = ~0.2s
        assert 0.15 <= run1_inference_time <= 0.25, (
            f"Run 1 inference time {run1_inference_time} not in expected range"
        )

        # Run 2: 2 requests (new instance, should load cached data)
        time.sleep(0.05)  # Gap between runs
        interceptor2 = ResponseStatsInterceptor(params)
        interceptor2.intercept_response(
            AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id="req_3"),
                latency_ms=300.0,
            ),
            context,
        )
        time.sleep(0.08)
        interceptor2.intercept_response(
            AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id="req_4"),
                latency_ms=400.0,
            ),
            context,
        )

        # Verify aggregation across runs
        assert interceptor2._stats["count"] == 4  # 2 + 2
        assert interceptor2._stats["successful_count"] == 4
        assert interceptor2._stats["status_codes"][200] == 4
        assert interceptor2._stats["finish_reason"]["stop"] == 4
        assert interceptor2._stats["run_id"] == 1  # Should increment

        # Verify latency aggregation
        assert (
            interceptor2._stats["avg_latency_ms"] == 250.0
        )  # (100 + 200 + 300 + 400) / 4
        assert interceptor2._stats["max_latency_ms"] == 400.0

        # Verify inference_time aggregation (sum of all runs)
        total_inference_time = 0.0
        for run_data in interceptor2._stats["inference_run_times"].values():
            if "inference_time" in run_data:
                total_inference_time += run_data["inference_time"]
        assert abs(interceptor2._stats["inference_time"] - total_inference_time) < 0.001
        assert total_inference_time > 0.1, (
            f"Total inference time {total_inference_time} should be > 0.1"
        )

        # Verify individual request caching
        assert interceptor2._request_stats_cache.get("req_1") is not None
        assert interceptor2._request_stats_cache.get("req_4") is not None

        # Verify run_ids are preserved
        cached_state = interceptor2._load_interceptor_state()
        run_ids = [run["run_id"] for run in cached_state["run_ids"]]
        assert 0 in run_ids and 1 in run_ids

    @pytest.mark.parametrize(
        "save_individuals,test_individual_cache,test_aggregated_cache,test_persistence",
        [
            # Individual caching disabled
            (False, False, True, False),
            # Individual caching enabled
            (True, True, True, False),
            # Persistence test
            (True, True, True, True),
        ],
    )
    def test_cache_functionality(
        self,
        tmp_path,
        save_individuals,
        test_individual_cache,
        test_aggregated_cache,
        test_persistence,
    ):
        """Test cache functionality for various scenarios."""
        cache_dir = tmp_path / "test_cache"
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=save_individuals
        )
        interceptor = ResponseStatsInterceptor(params)

        # Test individual request caching
        if test_individual_cache:
            request_id = "test_request_123"
            stats = {"status_code": 200, "prompt_tokens": 100}
            interceptor._cache_request_stats(request_id, stats)

            cached_stats = interceptor._request_stats_cache.get(request_id)
            if save_individuals:
                assert cached_stats is not None
                assert cached_stats["status_code"] == 200
            else:
                assert cached_stats is None

        # Test aggregated stats caching
        if test_aggregated_cache:
            interceptor._stats["count"] = 5
            interceptor._stats["successful_count"] = 4
            interceptor._stats["avg_prompt_tokens"] = 150.0
            interceptor._stats["status_codes"][200] = 4

            interceptor._save_aggregated_stats_to_cache()
            cached_state = interceptor._load_interceptor_state()

            assert "aggregated_stats" in cached_state
            assert cached_state["aggregated_stats"]["count"] == 5
            assert cached_state["aggregated_stats"]["successful_count"] == 4

        # Test persistence across instances
        if test_persistence:
            # Create second interceptor (should load cached data)
            interceptor2 = ResponseStatsInterceptor(params)

            # Verify data was loaded from cache
            assert interceptor2._stats["count"] == 5
            assert interceptor2._stats["successful_count"] == 4
            assert interceptor2._stats["avg_prompt_tokens"] == 150.0

    @pytest.mark.parametrize(
        "test_scenario,expected_count,expected_success,test_inference_time,latency_ms,status_code",
        [
            # Missing data - should initialize with defaults
            ("missing_data", 0, 0, False, None, 200),
            # Multiple requests accumulation
            ("accumulation", 3, 3, False, 100.0, 200),
            # Status code handling
            ("status_codes", 1, 1, False, 100.0, 200),
            ("status_codes", 1, 0, False, 100.0, 400),
            ("status_codes", 1, 0, False, 100.0, 201),
            # Inference time aggregation across multiple run_ids
            ("inference_time_multiple_runs", 6, 6, True, 100.0, 200),
        ],
    )
    def test_comprehensive_cache_scenarios(
        self,
        tmp_path,
        test_scenario,
        expected_count,
        expected_success,
        test_inference_time,
        latency_ms,
        status_code,
    ):
        """Test comprehensive cache functionality including inference_time aggregation."""
        cache_dir = tmp_path / "test_cache"

        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=True
        )
        context = AdapterGlobalContext(output_dir=str(tmp_path), url="http://test.com")

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = status_code
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {"choices": [{"finish_reason": "stop"}]}

        if test_scenario == "missing_data":
            # Test missing data handling
            interceptor = ResponseStatsInterceptor(params)
            assert interceptor._stats["count"] == 0
            assert interceptor._stats["successful_count"] == 0
            assert interceptor._stats["run_id"] == 0

        elif test_scenario == "accumulation":
            # Test multiple requests accumulation
            interceptor1 = ResponseStatsInterceptor(params)

            # Process 2 requests
            for i in range(2):
                adapter_response = AdapterResponse(
                    r=mock_resp,
                    rctx=AdapterRequestContext(request_id=f"req_{i}"),
                    latency_ms=latency_ms,
                )
                interceptor1.intercept_response(adapter_response, context)

            # Create second interceptor (should load cached data)
            interceptor2 = ResponseStatsInterceptor(params)
            assert interceptor2._stats["count"] == 2
            assert interceptor2._stats["successful_count"] == 2
            assert interceptor2._stats["run_id"] == 1  # Should increment

            # Process third request
            adapter_response = AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id="req_2"),
                latency_ms=latency_ms,
            )
            interceptor2.intercept_response(adapter_response, context)

            assert interceptor2._stats["count"] == 3
            assert interceptor2._stats["successful_count"] == 3

        elif test_scenario == "status_codes":
            # Test status code handling
            interceptor = ResponseStatsInterceptor(params)
            adapter_response = AdapterResponse(
                r=mock_resp, rctx=AdapterRequestContext(), latency_ms=latency_ms
            )
            interceptor.intercept_response(adapter_response, context)

            assert interceptor._stats["count"] == 1
            assert interceptor._stats["successful_count"] == expected_success
            assert interceptor._stats["status_codes"][status_code] == 1

        elif test_scenario == "inference_time_multiple_runs":
            # Test comprehensive inference time aggregation across multiple run_ids
            interceptor1 = ResponseStatsInterceptor(params)

            # Run 1: 2 requests
            interceptor1.intercept_response(
                AdapterResponse(
                    r=mock_resp, rctx=AdapterRequestContext(), latency_ms=100.0
                ),
                context,
            )
            time.sleep(0.1)
            interceptor1.intercept_response(
                AdapterResponse(
                    r=mock_resp, rctx=AdapterRequestContext(), latency_ms=200.0
                ),
                context,
            )

            # Verify Run 1 inference time
            run1_time = interceptor1._stats["inference_time"]
            # With latency-based estimation: sleep_time + latency_adjustment
            # Expected: ~0.1s (sleep) + ~0.1s (first request latency) = ~0.2s
            assert 0.15 <= run1_time <= 0.25, (
                f"Run 1 time {run1_time} not in expected range"
            )
            assert interceptor1._stats["run_id"] == 0
            # Run 1 uses integer keys (not yet cached)
            assert 0 in interceptor1._stats["inference_run_times"]
            assert "inference_time" in interceptor1._stats["inference_run_times"][0]

            # Run 2: 2 requests (new instance, should load cached data)
            time.sleep(0.05)
            interceptor2 = ResponseStatsInterceptor(params)
            interceptor2.intercept_response(
                AdapterResponse(
                    r=mock_resp, rctx=AdapterRequestContext(), latency_ms=300.0
                ),
                context,
            )
            time.sleep(0.08)
            interceptor2.intercept_response(
                AdapterResponse(
                    r=mock_resp, rctx=AdapterRequestContext(), latency_ms=400.0
                ),
                context,
            )

            # Verify Run 2 inference time
            # All run_ids should be integers after cache loading fix
            run2_time = interceptor2._stats["inference_run_times"][1]["inference_time"]
            # With latency-based estimation: sleep_time + latency_adjustment
            # Expected: ~0.08s (sleep) + ~0.3s (first request latency) = ~0.38s
            assert 0.35 <= run2_time <= 0.45, (
                f"Run 2 time {run2_time} not in expected range"
            )
            assert interceptor2._stats["run_id"] == 1
            assert 1 in interceptor2._stats["inference_run_times"]
            assert "inference_time" in interceptor2._stats["inference_run_times"][1]

            # Run 3: 2 more requests (third instance)
            time.sleep(0.05)
            interceptor3 = ResponseStatsInterceptor(params)
            interceptor3.intercept_response(
                AdapterResponse(
                    r=mock_resp, rctx=AdapterRequestContext(), latency_ms=500.0
                ),
                context,
            )
            time.sleep(0.06)
            interceptor3.intercept_response(
                AdapterResponse(
                    r=mock_resp, rctx=AdapterRequestContext(), latency_ms=600.0
                ),
                context,
            )

            # Verify Run 3 inference time
            # All run_ids should be integers after cache loading fix
            run3_time = interceptor3._stats["inference_run_times"][2]["inference_time"]
            # With latency-based estimation: sleep_time + latency_adjustment
            # Expected: ~0.06s (sleep) + ~0.5s (first request latency) = ~0.56s
            assert 0.50 <= run3_time <= 0.65, (
                f"Run 3 time {run3_time} not in expected range"
            )
            assert interceptor3._stats["run_id"] == 2
            assert 2 in interceptor3._stats["inference_run_times"]
            assert "inference_time" in interceptor3._stats["inference_run_times"][2]

            # Verify total aggregation across all runs
            assert interceptor3._stats["count"] == 6
            assert interceptor3._stats["successful_count"] == 6

            # Verify global inference_time is sum of all run inference times
            total_inference_time = 0.0
            for run_id, run_data in interceptor3._stats["inference_run_times"].items():
                if "inference_time" in run_data:
                    total_inference_time += run_data["inference_time"]
                    # Verify each run has reasonable inference time
                    assert run_data["inference_time"] > 0.05, (
                        f"Run {run_id} inference time too small: {run_data['inference_time']}"
                    )

            assert (
                abs(interceptor3._stats["inference_time"] - total_inference_time)
                < 0.001
            ), (
                f"Global inference time {interceptor3._stats['inference_time']} != sum of runs {total_inference_time}"
            )
            assert total_inference_time > 0.15, (
                f"Total inference time {total_inference_time} should be > 0.15"
            )

            # Verify all run_ids are tracked (should all be integers after cache loading fix)
            actual_run_ids = set(interceptor3._stats["inference_run_times"].keys())
            expected_run_ids = {0, 1, 2}
            assert actual_run_ids == expected_run_ids, (
                f"Expected run_ids {expected_run_ids}, got {actual_run_ids}"
            )
