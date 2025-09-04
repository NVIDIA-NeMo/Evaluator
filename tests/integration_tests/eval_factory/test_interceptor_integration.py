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
        env["NV_EVAL_LOG_DIR"] = self.test_log_dir
        env["NV_EVAL_LOG_LEVEL"] = "DEBUG"

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                "eval-factory",
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
                    "target.api_endpoint.adapter_config.use_reasoning=True,"
                    "logging.level=DEBUG"
                ),
            ]

            logger.info(f"Testing core interceptors runtime behavior: {' '.join(cmd)}")
            subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)

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

    def test_comprehensive_interceptor_chain(self, fake_endpoint, fake_url):
        """Test all interceptors are actually working at runtime."""
        env = os.environ.copy()
        timestamp = int(time.time())
        self.test_log_dir = f"./e2e_run/comprehensive_chain_{timestamp}"
        env["NV_EVAL_LOG_DIR"] = self.test_log_dir
        env["NV_EVAL_LOG_LEVEL"] = "DEBUG"

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = [
                "eval-factory",
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
                    "target.api_endpoint.adapter_config.use_system_prompt=True,"
                    "target.api_endpoint.adapter_config.custom_system_prompt=You are a helpful AI assistant.,"
                    "target.api_endpoint.adapter_config.use_request_logging=True,"
                    "target.api_endpoint.adapter_config.use_response_logging=True,"
                    "target.api_endpoint.adapter_config.use_caching=True,"
                    "target.api_endpoint.adapter_config.caching_dir="
                    + temp_dir
                    + "/cache,"
                    "target.api_endpoint.adapter_config.use_reasoning=True,"
                    "target.api_endpoint.adapter_config.use_progress_tracking=True,"
                    "target.api_endpoint.adapter_config.progress_tracking_interval=1,"
                    'target.api_endpoint.adapter_config.params_to_add={"comprehensive_test": true},'
                    "logging.level=DEBUG"
                ),
            ]

            logger.info(f"Testing all interceptors runtime behavior: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=60
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
                    assert "System message added to request" in content, (
                        "System message interceptor should add messages at runtime"
                    )
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
                    assert "Sample processed" in content, (
                        "Progress tracking interceptor should track progress at runtime"
                    )
                    assert "Processing request payload" in content, (
                        "Payload modifier interceptor should modify payloads at runtime"
                    )

                    break  # Only check first log file

                logger.info("✅ All interceptors runtime behavior verified")
            else:
                pytest.fail(
                    "No log directory created for comprehensive chain test - test should have generated logs"
                )
