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
Regression tests for ResponseStatsInterceptor to ensure correctness during optimization.

These tests verify that any performance optimizations maintain the same behavior
and output as the original implementation.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock

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


class TestResponseStatsRegressionBasic:
    """Basic regression tests for core functionality."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a unique cache directory for each test."""
        return tmp_path / "cache"

    @pytest.fixture
    def context(self, tmp_path):
        """Create test context."""
        return AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

    def create_response(
        self,
        request_id: str,
        status_code: int = 200,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        latency_ms: float = 100.0,
    ) -> AdapterResponse:
        """Helper to create test responses."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = status_code
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_tokens": completion_tokens,
            },
            "choices": [{"finish_reason": "stop", "message": {}}],
        }

        return AdapterResponse(
            r=mock_resp,
            rctx=AdapterRequestContext(request_id=request_id),
            latency_ms=latency_ms,
        )

    def test_exact_token_counts(self, cache_dir, context):
        """Verify exact token count calculations."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        # Test case 1: Single response
        response = self.create_response("req1", prompt_tokens=100, completion_tokens=50)
        interceptor.intercept_response(response, context)

        assert interceptor._stats["avg_prompt_tokens"] == 100
        assert interceptor._stats["avg_completion_tokens"] == 50
        assert interceptor._stats["avg_total_tokens"] == 150
        assert interceptor._stats["max_prompt_tokens"] == 100
        assert interceptor._stats["successful_count"] == 1

        # Test case 2: Second response with different tokens
        response = self.create_response("req2", prompt_tokens=200, completion_tokens=100)
        interceptor.intercept_response(response, context)

        assert interceptor._stats["avg_prompt_tokens"] == 150.0  # (100 + 200) / 2
        assert interceptor._stats["avg_completion_tokens"] == 75.0  # (50 + 100) / 2
        assert interceptor._stats["avg_total_tokens"] == 225.0  # (150 + 300) / 2
        assert interceptor._stats["max_prompt_tokens"] == 200
        assert interceptor._stats["successful_count"] == 2

        # Test case 3: Third response
        response = self.create_response("req3", prompt_tokens=150, completion_tokens=75)
        interceptor.intercept_response(response, context)

        assert interceptor._stats["avg_prompt_tokens"] == 150.0  # (100+200+150)/3
        assert interceptor._stats["avg_completion_tokens"] == 75.0  # (50+100+75)/3
        assert interceptor._stats["avg_total_tokens"] == 225.0  # (150+300+225)/3
        assert interceptor._stats["max_prompt_tokens"] == 200
        assert interceptor._stats["successful_count"] == 3

    def test_exact_status_code_counts(self, cache_dir, context):
        """Verify exact status code counting."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        # Process responses with different status codes
        for status in [200, 200, 404, 200, 500, 200]:
            response = self.create_response(f"req_{status}", status_code=status)
            interceptor.intercept_response(response, context)

        assert interceptor._stats["count"] == 6
        assert interceptor._stats["status_codes"][200] == 4
        assert interceptor._stats["status_codes"][404] == 1
        assert interceptor._stats["status_codes"][500] == 1
        assert interceptor._stats["successful_count"] == 4  # Only 200s

    def test_exact_finish_reason_counts(self, cache_dir, context):
        """Verify exact finish reason counting."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        finish_reasons = ["stop", "stop", "length", "stop", "tool_calls"]

        for i, reason in enumerate(finish_reasons):
            mock_resp = Mock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
            mock_resp.json.return_value = {
                "usage": {"prompt_tokens": 100, "total_tokens": 150, "completion_tokens": 50},
                "choices": [{"finish_reason": reason, "message": {}}],
            }

            response = AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id=f"req_{i}"),
                latency_ms=100.0,
            )
            interceptor.intercept_response(response, context)

        assert interceptor._stats["finish_reason"]["stop"] == 3
        assert interceptor._stats["finish_reason"]["length"] == 1
        assert interceptor._stats["finish_reason"]["tool_calls"] == 1

    def test_exact_latency_calculations(self, cache_dir, context):
        """Verify exact latency calculations."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        latencies = [100.0, 200.0, 150.0]

        for i, latency in enumerate(latencies):
            response = self.create_response(f"req_{i}", latency_ms=latency)
            interceptor.intercept_response(response, context)

        assert interceptor._stats["avg_latency_ms"] == 150.0  # (100+200+150)/3
        assert interceptor._stats["max_latency_ms"] == 200.0

    def test_cached_responses_not_counted(self, cache_dir, context):
        """Verify that cached responses are skipped correctly."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        # Normal response
        response = self.create_response("req1")
        interceptor.intercept_response(response, context)

        # Cached response
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_rctx = Mock()
        mock_rctx.cache_hit = True
        cached_response = AdapterResponse(r=mock_resp, rctx=mock_rctx)
        interceptor.intercept_response(cached_response, context)

        # Should only count the non-cached response
        assert interceptor._stats["count"] == 1
        assert interceptor._stats["successful_count"] == 1


class TestResponseStatsRegressionPersistence:
    """Regression tests for persistence and caching behavior."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a unique cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def context(self, tmp_path):
        """Create test context."""
        return AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

    def create_response(self, request_id: str, prompt_tokens: int = 100) -> AdapterResponse:
        """Helper to create test responses."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + 50,
                "completion_tokens": 50,
            },
            "choices": [{"finish_reason": "stop", "message": {}}],
        }

        return AdapterResponse(
            r=mock_resp,
            rctx=AdapterRequestContext(request_id=request_id),
            latency_ms=100.0,
        )

    def test_state_persistence_across_instances(self, cache_dir, context):
        """Verify state persists correctly across interceptor instances."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=True
        )

        # First instance - process 3 responses
        interceptor1 = ResponseStatsInterceptor(params)
        for i in range(3):
            response = self.create_response(f"req1_{i}", prompt_tokens=100 + i * 10)
            interceptor1.intercept_response(response, context)

        assert interceptor1._stats["count"] == 3
        assert interceptor1._stats["successful_count"] == 3
        assert interceptor1._stats["run_id"] == 0
        run1_avg_prompt_tokens = interceptor1._stats["avg_prompt_tokens"]

        # Second instance - should load state and continue
        time.sleep(0.1)
        interceptor2 = ResponseStatsInterceptor(params)

        # Verify loaded state
        assert interceptor2._stats["count"] == 3
        assert interceptor2._stats["successful_count"] == 3
        assert interceptor2._stats["run_id"] == 1  # Should increment
        assert interceptor2._stats["avg_prompt_tokens"] == run1_avg_prompt_tokens

        # Process 2 more responses
        for i in range(2):
            response = self.create_response(f"req2_{i}", prompt_tokens=150 + i * 10)
            interceptor2.intercept_response(response, context)

        assert interceptor2._stats["count"] == 5
        assert interceptor2._stats["successful_count"] == 5

    def test_individual_request_caching(self, cache_dir, context):
        """Verify individual request stats are cached correctly."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=True
        )
        interceptor = ResponseStatsInterceptor(params)

        test_requests = [
            ("req_1", 100),
            ("req_2", 200),
            ("req_3", 150),
        ]

        for request_id, prompt_tokens in test_requests:
            response = self.create_response(request_id, prompt_tokens=prompt_tokens)
            interceptor.intercept_response(response, context)

        # Verify all requests are cached
        for request_id, expected_prompt_tokens in test_requests:
            cached_data = interceptor._request_stats_cache.get(request_id)
            assert cached_data is not None
            assert cached_data["prompt_tokens"] == expected_prompt_tokens
            assert cached_data["request_id"] == request_id

    def test_run_id_incrementing(self, cache_dir, context):
        """Verify run_id increments correctly across instances."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )

        run_ids = []
        for i in range(5):
            interceptor = ResponseStatsInterceptor(params)
            run_ids.append(interceptor._stats["run_id"])
            # Process at least one response to trigger save
            response = self.create_response(f"req_{i}")
            interceptor.intercept_response(response, context)
            time.sleep(0.05)

        assert run_ids == [0, 1, 2, 3, 4]


class TestResponseStatsRegressionConcurrency:
    """Regression tests for thread safety and concurrent access."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a unique cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def context(self, tmp_path):
        """Create test context."""
        return AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

    def create_response(self, request_id: str) -> AdapterResponse:
        """Helper to create test responses."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {
            "usage": {"prompt_tokens": 100, "total_tokens": 150, "completion_tokens": 50},
            "choices": [{"finish_reason": "stop", "message": {}}],
        }

        return AdapterResponse(
            r=mock_resp,
            rctx=AdapterRequestContext(request_id=request_id),
            latency_ms=100.0,
        )

    def test_concurrent_count_accuracy(self, cache_dir, context):
        """Verify count accuracy under concurrent access."""
        import threading

        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False, logging_aggregated_stats_interval=10000
        )
        interceptor = ResponseStatsInterceptor(params)

        responses_per_thread = 100
        num_threads = 10

        def process_responses(thread_id: int):
            for i in range(responses_per_thread):
                response = self.create_response(f"req_t{thread_id}_{i}")
                interceptor.intercept_response(response, context)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=process_responses, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        expected_count = responses_per_thread * num_threads
        assert (
            interceptor._stats["count"] == expected_count
        ), f"Expected {expected_count}, got {interceptor._stats['count']}"
        assert (
            interceptor._stats["successful_count"] == expected_count
        ), f"Expected {expected_count}, got {interceptor._stats['successful_count']}"

    def test_concurrent_status_code_counting(self, cache_dir, context):
        """Verify status code counting accuracy under concurrent access."""
        import threading

        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False, logging_aggregated_stats_interval=10000
        )
        interceptor = ResponseStatsInterceptor(params)

        num_threads = 5
        responses_per_thread = 50

        def process_responses(thread_id: int):
            for i in range(responses_per_thread):
                # Alternate between status codes
                status_code = 200 if i % 2 == 0 else 404
                mock_resp = Mock(spec=requests.Response)
                mock_resp.status_code = status_code
                mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
                mock_resp.json.return_value = {"choices": []}

                response = AdapterResponse(
                    r=mock_resp,
                    rctx=AdapterRequestContext(request_id=f"req_t{thread_id}_{i}"),
                    latency_ms=100.0,
                )
                interceptor.intercept_response(response, context)

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=process_responses, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        total_responses = num_threads * responses_per_thread
        expected_200s = num_threads * (responses_per_thread // 2 + responses_per_thread % 2)
        expected_404s = num_threads * (responses_per_thread // 2)

        assert interceptor._stats["count"] == total_responses
        assert interceptor._stats["status_codes"][200] == expected_200s
        assert interceptor._stats["status_codes"][404] == expected_404s


class TestResponseStatsRegressionEdgeCases:
    """Regression tests for edge cases and error handling."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a unique cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def context(self, tmp_path):
        """Create test context."""
        return AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

    def test_zero_token_responses(self, cache_dir, context):
        """Handle responses with zero tokens."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {
            "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0},
            "choices": [{"finish_reason": "stop", "message": {}}],
        }

        response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(request_id="req1"), latency_ms=100.0
        )
        interceptor.intercept_response(response, context)

        assert interceptor._stats["avg_prompt_tokens"] == 0
        assert interceptor._stats["avg_completion_tokens"] == 0
        assert interceptor._stats["successful_count"] == 1

    def test_missing_usage_field(self, cache_dir, context):
        """Handle responses with missing usage field."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {"choices": [{"finish_reason": "stop", "message": {}}]}

        response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(request_id="req1"), latency_ms=100.0
        )
        interceptor.intercept_response(response, context)

        # Should handle gracefully with zero tokens
        assert interceptor._stats["successful_count"] == 1
        assert interceptor._stats["avg_prompt_tokens"] == 0

    def test_malformed_json_response(self, cache_dir, context):
        """Handle responses with malformed JSON."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(request_id="req1"), latency_ms=100.0
        )
        interceptor.intercept_response(response, context)

        # Should still count basic stats
        assert interceptor._stats["count"] == 1
        assert interceptor._stats["successful_count"] == 0  # No detailed stats
        assert interceptor._stats["avg_prompt_tokens"] is None

    def test_none_latency(self, cache_dir, context):
        """Handle responses with None latency."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {
            "usage": {"prompt_tokens": 100, "total_tokens": 150, "completion_tokens": 50},
            "choices": [{"finish_reason": "stop", "message": {}}],
        }

        response = AdapterResponse(
            r=mock_resp, rctx=AdapterRequestContext(request_id="req1"), latency_ms=None
        )
        interceptor.intercept_response(response, context)

        assert interceptor._stats["count"] == 1
        assert interceptor._stats["avg_latency_ms"] is None


class TestResponseStatsRegressionFileOutput:
    """Regression tests for file output and metrics."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a unique cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def context(self, tmp_path):
        """Create test context."""
        return AdapterGlobalContext(
            output_dir=str(tmp_path), url="http://test.api.com/v1/chat/completions"
        )

    def create_response(self, request_id: str, prompt_tokens: int = 100) -> AdapterResponse:
        """Helper to create test responses."""
        mock_resp = Mock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
        mock_resp.json.return_value = {
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + 50,
                "completion_tokens": 50,
            },
            "choices": [{"finish_reason": "stop", "message": {}}],
        }

        return AdapterResponse(
            r=mock_resp,
            rctx=AdapterRequestContext(request_id=request_id),
            latency_ms=100.0,
        )

    def test_metrics_file_format(self, cache_dir, context):
        """Verify metrics file has correct format."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        # Process some responses
        for i in range(5):
            response = self.create_response(f"req_{i}", prompt_tokens=100 + i * 10)
            interceptor.intercept_response(response, context)

        # Save to file
        interceptor.post_eval_hook(context)

        # Verify file exists and has correct structure
        metrics_file = Path(context.output_dir) / "eval_factory_metrics.json"
        assert metrics_file.exists()

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        assert "response_stats" in metrics
        stats = metrics["response_stats"]

        # Verify all required fields are present
        required_fields = [
            "count",
            "successful_count",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
            "status_codes",
            "finish_reason",
        ]

        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"

    def test_metrics_file_values_accuracy(self, cache_dir, context):
        """Verify metrics file contains accurate values."""
        params = ResponseStatsInterceptor.Params(
            cache_dir=str(cache_dir), save_individuals=False
        )
        interceptor = ResponseStatsInterceptor(params)

        # Process responses with known values
        test_data = [
            (100, 50),  # prompt, completion
            (200, 100),
            (150, 75),
        ]

        for i, (prompt, completion) in enumerate(test_data):
            mock_resp = Mock(spec=requests.Response)
            mock_resp.status_code = 200
            mock_resp.headers = {"date": "2023-01-01T00:00:00Z"}
            mock_resp.json.return_value = {
                "usage": {
                    "prompt_tokens": prompt,
                    "total_tokens": prompt + completion,
                    "completion_tokens": completion,
                },
                "choices": [{"finish_reason": "stop", "message": {}}],
            }

            response = AdapterResponse(
                r=mock_resp,
                rctx=AdapterRequestContext(request_id=f"req_{i}"),
                latency_ms=100.0 + i * 50,
            )
            interceptor.intercept_response(response, context)

        # Save to file
        interceptor.post_eval_hook(context)

        # Load and verify
        metrics_file = Path(context.output_dir) / "eval_factory_metrics.json"
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        stats = metrics["response_stats"]

        # Verify exact values
        assert stats["count"] == 3
        assert stats["successful_count"] == 3
        assert stats["avg_prompt_tokens"] == 150.0  # (100+200+150)/3
        assert stats["avg_completion_tokens"] == 75.0  # (50+100+75)/3
        assert stats["max_prompt_tokens"] == 200
        assert stats["status_codes"]["200"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

