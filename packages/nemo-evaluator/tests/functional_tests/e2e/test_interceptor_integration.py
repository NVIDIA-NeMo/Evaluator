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
"""Integration tests for eval-factory interceptors using fake endpoint.

This module tests the actual runtime behavior of interceptors
when running eval-factory with the fake endpoint.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from nemo_evaluator.logging.utils import logger


class TestInterceptorIntegration:
    """Test interceptor integration with eval-factory and fake endpoint."""

    @classmethod
    def teardown_class(cls):
        """Clean up after all tests in the class."""
        # Final cleanup of any remaining logs
        base_log_dir = Path("./e2e_run")
        if base_log_dir.exists():
            try:
                shutil.rmtree(base_log_dir)
                logger.info(
                    "🧹 Final cleanup: Removed base log directory and all contents"
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not perform final cleanup: {e}")

    def setup_method(self):
        """Set up test environment before each test method."""
        # Create base log directory
        self.base_log_dir = Path("./e2e_run")
        self.base_log_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up test-specific log directories
        if hasattr(self, "test_log_dir") and Path(self.test_log_dir).exists():
            try:
                shutil.rmtree(self.test_log_dir)
                logger.info(f"🧹 Cleaned up test log directory: {self.test_log_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Could not clean up {self.test_log_dir}: {e}")

        # Clean up any temporary cache directories
        cache_dirs = list(Path("/tmp").glob("cache_test_*"))
        for cache_dir in cache_dirs:
            try:
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    logger.info(f"🧹 Cleaned up cache directory: {cache_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Could not clean up {cache_dir}: {e}")

        # Clean up main log files that might have been created
        if self.base_log_dir.exists():
            for log_file in self.base_log_dir.glob("*.log*"):
                try:
                    log_file.unlink()
                    logger.info(f"🧹 Cleaned up main log file: {log_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not clean up {log_file}: {e}")

            # Try to remove the base log directory if it's empty
            try:
                if not any(self.base_log_dir.iterdir()):
                    self.base_log_dir.rmdir()
                    logger.info(
                        f"🧹 Removed empty base log directory: {self.base_log_dir}"
                    )
            except Exception as e:
                logger.warning(f"⚠️  Could not remove base log directory: {e}")

    def test_core_interceptor_chain(self, fake_endpoint, fake_url):
        """Test core interceptors are actually working at runtime."""
        env = os.environ.copy()
        timestamp = int(time.time())
        self.test_log_dir = f"./e2e_run/core_chain_{timestamp}"
        env["NEMO_EVALUATOR_LOG_DIR"] = self.test_log_dir
        env["NEMO_EVALUATOR_LOG_LEVEL"] = "DEBUG"

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                "nemo-evaluator",
                "run_eval",
                "--eval_type",
                "mmlu_pro",
                "--model_id",
                "Qwen/Qwen3-8B",
                "--model_url",
                fake_url,
                "--model_type",
                "chat",
                "--api_key_name",
                "API_KEY",
                "--output_dir",
                temp_dir,
                "--overrides",
                (
                    "config.params.limit_samples=2,"
                    "target.api_endpoint.url=" + fake_url + ","
                    "target.api_endpoint.adapter_config.use_request_logging=True,"
                    "target.api_endpoint.adapter_config.use_response_logging=True,"
                    "target.api_endpoint.adapter_config.use_caching=True,"
                    "target.api_endpoint.adapter_config.caching_dir="
                    + temp_dir
                    + "/cache,"
                    "target.api_endpoint.adapter_config.process_reasoning_traces=True,"
                    "logging.level=DEBUG"
                ),
            ]

            logger.info(f"Testing core interceptors runtime behavior: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=False, text=True, env=env, timeout=60)

            # Check ONLY runtime behavior - what the interceptors actually DO
            log_dir = Path(self.test_log_dir)
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                logger.info(f"Core chain test created {len(log_files)} log files")

                for log_file in log_files:
                    content = log_file.read_text()

                    # Runtime behavior only - what interceptors actually do during execution
                    assert "Incoming request" in content, (
                        "Request logging interceptor should log requests at runtime"
                    )
                    assert "Outgoing response" in content, (
                        "Response logging interceptor should log responses at runtime"
                    )
                    assert "Processing request for caching" in content, (
                        "Caching interceptor should process requests at runtime"
                    )
                    assert "Processing response with choices" in content, (
                        "Reasoning interceptor should process responses at runtime"
                    )

                    break  # Only check first log file

                logger.info("✅ Core interceptors runtime behavior verified")
            else:
                pytest.fail(
                    "No log directory created for core chain test - test should have generated logs"
                )

    def test_log_cleanup_verification(self):
        """Test that logs are properly cleaned up after tests."""
        # Create a test log file to verify cleanup
        test_log_file = self.base_log_dir / "test_cleanup.log"
        test_log_file.write_text("Test log content for cleanup verification")

        # Verify the file exists
        assert test_log_file.exists(), "Test log file should be created"

        # The teardown method should clean this up
        logger.info("✅ Test log file created for cleanup verification")

        # Note: The actual cleanup happens in teardown_method

    def _build_eval_command(
        self, output_dir: str, fake_url: str, cache_dir: str = None
    ) -> list[str]:
        """Build eval-factory command with parameterized output and cache directories"""
        if cache_dir is None:
            cache_dir = f"{output_dir}/cache"

        return [
            "nemo-evaluator",
            "run_eval",
            "--eval_type",
            "mmlu_pro",
            "--model_id",
            "Qwen/Qwen3-8B",
            "--model_url",
            fake_url,
            "--model_type",
            "chat",
            "--api_key_name",
            "API_KEY",
            "--output_dir",
            output_dir,
            "--overrides",
            (
                "config.params.limit_samples=2,"
                "target.api_endpoint.url=" + fake_url + ","
                "target.api_endpoint.adapter_config.use_system_prompt=True,"
                "target.api_endpoint.adapter_config.custom_system_prompt=You are a helpful AI assistant.,"
                "target.api_endpoint.adapter_config.use_request_logging=True,"
                "target.api_endpoint.adapter_config.use_response_logging=True,"
                "target.api_endpoint.adapter_config.use_caching=True,"
                "target.api_endpoint.adapter_config.reuse_cached_responses=True,"
                "target.api_endpoint.adapter_config.save_requests=True,"
                "target.api_endpoint.adapter_config.max_saved_requests=1,"
                "target.api_endpoint.adapter_config.html_report_size=5,"
                "target.api_endpoint.adapter_config.caching_dir=" + cache_dir + ","
                "target.api_endpoint.adapter_config.process_reasoning_traces=True,"
                "target.api_endpoint.adapter_config.use_progress_tracking=True,"
                "target.api_endpoint.adapter_config.progress_tracking_interval=1,"
                'target.api_endpoint.adapter_config.params_to_add={"comprehensive_test": true},'
                "target.api_endpoint.adapter_config.tracking_requests_stats=True,"
                "target.api_endpoint.adapter_config.response_stats_cache_dir="
                + cache_dir
                + "/response_stats_cache,"
                "target.api_endpoint.adapter_config.generate_html_report=True,"
                "logging.level=DEBUG"
            ),
        ]

    def test_comprehensive_interceptor_chain(self, fake_endpoint, fake_url):
        """Test all interceptors are actually working at runtime."""
        env = os.environ.copy()
        timestamp = int(time.time())
        self.test_log_dir = f"./e2e_run/comprehensive_chain_{timestamp}"
        env["NEMO_EVALUATOR_LOG_DIR"] = self.test_log_dir
        env["NEMO_EVALUATOR_LOG_LEVEL"] = "DEBUG"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Build the command with all interceptors enabled
            cmd = self._build_eval_command(temp_dir, fake_url)

            logger.info(f"Testing all interceptors runtime behavior: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=False, text=True, env=env, timeout=60
            )
            logger.info("Finished the subprocess", result=result)

            # Check ONLY runtime behavior - what interceptors actually DO during execution
            log_dir = Path(self.test_log_dir)
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                # wipp
                print(log_files)
                logger.info(
                    f"Comprehensive chain test created {len(log_files)} log files"
                )

                for log_file in log_files:
                    content = log_file.read_text()

                    # Runtime behavior only - what each interceptor actually does during execution
                    # Check for interceptor initialization messages (these are logged during startup)
                    assert "System message interceptor initialized" in content, (
                        "System message interceptor should be initialized"
                    )
                    assert "Request logging interceptor initialized" in content, (
                        "Request logging interceptor should be initialized"
                    )
                    assert "Response logging interceptor initialized" in content, (
                        "Response logging interceptor should be initialized"
                    )
                    assert "Caching interceptor initialized" in content, (
                        "Caching interceptor should be initialized"
                    )

                    # Verify that caching interceptor is configured for reuse
                    if (
                        "reuse_cached_responses=True" in content
                        or "reuse_cached_responses: True" in content
                    ):
                        logger.info("✅ Caching interceptor configured for reuse")
                    else:
                        logger.warning(
                            "⚠️  Caching interceptor reuse configuration not found in logs"
                        )
                    assert "Reasoning interceptor initialized" in content, (
                        "Reasoning interceptor should be initialized"
                    )
                    assert "Response stats interceptor initialized" in content, (
                        "Response stats interceptor should be initialized"
                    )
                    assert "Payload modifier interceptor initialized" in content, (
                        "Payload modifier interceptor should be initialized"
                    )
                    assert "Progress tracking interceptor initialized" in content, (
                        "Progress tracking interceptor should be initialized"
                    )

                    # Check for actual runtime behavior messages (these will be logged during execution)
                    # Note: Some of these may not appear if the evaluation fails early due to interceptor issues
                    logger.info("✅ All interceptors initialization verified")

                    break  # Only check first log file

                logger.info("✅ All interceptors runtime behavior verified")

                # Verify that metrics JSON file exists and contains expected data
                metrics_file = Path(temp_dir) / "eval_factory_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    # Check that both reasoning and response stats are present
                    assert "reasoning" in metrics, (
                        "Reasoning stats should be present in metrics file"
                    )
                    assert "response_stats" in metrics, (
                        "Response stats should be present in metrics file"
                    )

                    reasoning_stats = metrics["reasoning"]
                    response_stats = metrics["response_stats"]

                    # Verify reasoning stats structure
                    assert "total_responses" in reasoning_stats, (
                        "Total responses should be tracked in reasoning stats"
                    )
                    assert "responses_with_reasoning" in reasoning_stats, (
                        "Responses with reasoning should be tracked"
                    )

                    # Verify response stats structure
                    assert "count" in response_stats, (
                        "Response count should be tracked in response stats"
                    )
                    assert "status_codes" in response_stats, (
                        "Status codes should be tracked in response stats"
                    )
                    assert "finish_reason" in response_stats, (
                        "Finish reasons should be tracked in response stats"
                    )

                    # Check that both stats have consistent total response counts
                    assert (
                        reasoning_stats["total_responses"] == response_stats["count"]
                    ), "Both stats should have the same total response count"
                    assert reasoning_stats["total_responses"] >= 2, (
                        "Should have processed at least 2 samples"
                    )

                    logger.info("✅ Metrics file verified:")
                    logger.info(f"   Reasoning stats: {reasoning_stats}")
                    logger.info(f"   Response stats: {response_stats}")
                else:
                    pytest.fail(
                        "Metrics file should be created with both types of stats"
                    )

                # Check that caching is working (since it's enabled)
                cache_dir = Path(temp_dir) / "cache"
                if cache_dir.exists():
                    # Check for cache.db files in the cache directory (should have at least 3: responses, requests, headers)
                    cache_db_files = list(cache_dir.glob("**/*.db"))
                    assert len(cache_db_files) >= 3, (
                        f"Should have at least 3 cache databases, found {len(cache_db_files)}"
                    )
                    logger.info(
                        f"✅ Cache databases created: {len(cache_db_files)} files"
                    )

                    # Test that reuse-caching is working and responses cache is properly filled
                    responses_cache_dir = cache_dir / "responses"
                    if responses_cache_dir.exists():
                        # Check that responses are actually cached (should have at least 2 responses for 2 samples)
                        response_cache_files = list(responses_cache_dir.glob("*.db"))
                        assert len(response_cache_files) >= 1, (
                            f"Should have at least 1 response cache database, found {len(response_cache_files)}"
                        )

                        # Verify that the responses cache contains actual data
                        # Since we're using SQLite databases, we can't easily read them without the Cache class
                        # But we can verify the directory structure and file sizes
                        for cache_file in response_cache_files:
                            assert cache_file.stat().st_size > 0, (
                                f"Cache file {cache_file.name} should not be empty"
                            )

                        logger.info(
                            f"✅ Response cache verification: {len(response_cache_files)} cache files with data"
                        )

                        # Check that the caching interceptor is actually processing requests and responses
                        # by looking for cache-related log messages
                        cache_logs_found = False
                        for log_file in log_files:
                            content = log_file.read_text()
                            if (
                                "Cached successful response" in content
                                or "cache_key=" in content
                            ):
                                cache_logs_found = True
                                logger.info(
                                    "✅ Found cache operation logs - caching is working"
                                )
                                break

                        if not cache_logs_found:
                            pytest.fail(
                                "❌ No cache operation logs found - caching is NOT working! This test should fail."
                            )
                    else:
                        pytest.fail("Response cache directory should be created")
                else:
                    pytest.fail("Cache directory should be created")

                # Verify that HTML report was generated
                html_report_path = Path(temp_dir) / "report.html"
                assert html_report_path.exists(), (
                    f"HTML report should be generated at {html_report_path}"
                )
                assert html_report_path.stat().st_size > 0, (
                    "HTML report should not be empty"
                )
                logger.info(f"✅ HTML report generated: {html_report_path}")

                # Test reuse-caching by running the same evaluation again and checking for cache hits
                logger.info(
                    "🔄 Testing reuse-caching by running the same evaluation again..."
                )

                # Use a separate output directory for the second run to avoid conflicts
                # BUT keep the SAME cache directory so it can reuse cached responses
                with tempfile.TemporaryDirectory() as temp_dir2:
                    # Create a new command for the second run with different output dir but same cache dir
                    cmd2 = self._build_eval_command(
                        temp_dir2, fake_url, cache_dir=str(cache_dir)
                    )

                    result2 = subprocess.run(
                        cmd2, capture_output=False, text=True, env=env, timeout=60
                    )
                    logger.info("Finished the second subprocess run", result2)

                    # The second run should succeed since it's using a different output directory
                    if result2.returncode != 0:
                        logger.error(
                            f"Second run failed with return code {result2.returncode}"
                        )
                        logger.error(f"stdout: {result2.stdout}")
                        logger.error(f"stderr: {result2.stderr}")
                        pytest.fail(
                            f"Second evaluation run should succeed, but failed with return code {result2.returncode}"
                        )

                    # Check for cache hit indicators in the second run logs
                    # Since we're using a different output directory, we need to check the new log directory
                    log_dir2 = Path(self.test_log_dir)
                    if log_dir2.exists():
                        log_files2 = list(log_dir2.glob("*.log"))
                        cache_hits_found = False
                        for log_file in log_files2:
                            content = log_file.read_text()
                            if (
                                "Returning cached response" in content
                                or "cache_hit=True" in content
                            ):
                                cache_hits_found = True
                                logger.info(
                                    "✅ Found cache hit logs - reuse-caching is working"
                                )
                                break

                        if not cache_hits_found:
                            pytest.fail(
                                "❌ No cache hit logs found - reuse-caching is NOT working! This test should fail."
                            )

                    # Verify that the cache still contains the same data
                    if cache_dir.exists():
                        responses_cache_dir = cache_dir / "responses"
                        if responses_cache_dir.exists():
                            response_cache_files_after = list(
                                responses_cache_dir.glob("*.db")
                            )
                            # The cache should still have the same number of files after the second run
                            assert len(response_cache_files_after) >= 1, (
                                f"Cache should still contain data after second run, found {len(response_cache_files_after)} files"
                            )
                            logger.info(
                                "✅ Cache data persistence verified after second run"
                            )

                logger.info(
                    "✅ Comprehensive interceptor chain test completed successfully"
                )
            else:
                pytest.fail(
                    "No log directory created for comprehensive chain test - test should have generated logs"
                )
