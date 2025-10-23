# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for logging_utils.py functionality."""

import os
from unittest.mock import patch

import pytest

from nemo_evaluator_launcher.common.logging_utils import (
    _DEFAULT_LOG_LEVEL,
    _LOG_LEVEL_ENV_VAR,
    _get_env_log_level,
    redact_processor,
)


class TestGetEnvLogLevel:
    """Test cases for _get_env_log_level function."""

    def test_default_log_level_when_no_env_vars(self):
        """Test default log level when no environment variables are set."""

        with patch.dict(os.environ, {}, clear=True):
            result = _get_env_log_level()
            assert result == _DEFAULT_LOG_LEVEL

    def test_nemo_evaluator_log_level_takes_precedence(self):
        """Test that NEMO_EVALUATOR_LOG_LEVEL takes precedence over LOG_LEVEL."""

        with patch.dict(
            os.environ,
            {_LOG_LEVEL_ENV_VAR: "ERROR", "LOG_LEVEL": "DEBUG"},
            clear=True,
        ):
            result = _get_env_log_level()
            assert result == "ERROR"

    def test_fallback_to_log_level(self):
        """Test fallback to LOG_LEVEL when NEMO_EVALUATOR_LOG_LEVEL is not set."""

        with patch.dict(os.environ, {"LOG_LEVEL": "INFO"}, clear=True):
            result = _get_env_log_level()
            assert result == "INFO"

    @pytest.mark.parametrize(
        "input_level,expected_output",
        [
            ("d", "DEBUG"),
            ("i", "INFO"),
            ("w", "WARNING"),
            ("e", "ERROR"),
            ("f", "CRITICAL"),
            ("D", "DEBUG"),
            ("I", "INFO"),
            ("W", "WARNING"),
            ("E", "ERROR"),
            ("F", "CRITICAL"),
            ("debug", "DEBUG"),
            ("info", "INFO"),
            ("warning", "WARNING"),
            ("error", "ERROR"),
            ("critical", "CRITICAL"),
            ("DEBUG", "DEBUG"),
            ("INFO", "INFO"),
            ("WARNING", "WARNING"),
            ("ERROR", "ERROR"),
            ("CRITICAL", "CRITICAL"),
        ],
    )
    def test_case_single_letters_and_other(
        self, input_level: str, expected_output: str
    ):
        """Test all cases"""

        with patch.dict(os.environ, {_LOG_LEVEL_ENV_VAR: input_level}, clear=True):
            result = _get_env_log_level()
            assert result == expected_output

    @pytest.mark.parametrize(
        "invalid_level",
        [
            "INVALID",
            "TRACE",
            "VERBOSE",
            "123",
            "DEBUG_EXTRA",
        ],
    )
    def test_invalid_log_levels_returned_unchanged(self, invalid_level: str):
        """Test that invalid/unknown log levels are returned unchanged (uppercase)."""

        with patch.dict(os.environ, {_LOG_LEVEL_ENV_VAR: invalid_level}, clear=True):
            result = _get_env_log_level()
            assert result == invalid_level.upper()

    def test_empty_nemo_evaluator_log_level_defaults(self):
        """Test that empty NEMO_EVALUATOR_LOG_LEVEL defaults to _DEFAULT_LOG_LEVEL."""

        with patch.dict(
            os.environ,
            {_LOG_LEVEL_ENV_VAR: "", "LOG_LEVEL": "ERROR"},
            clear=True,
        ):
            result = _get_env_log_level()
            assert result == _DEFAULT_LOG_LEVEL

    def test_empty_both_env_vars_defaults(self):
        """Test that empty values for both env vars defaults to _DEFAULT_LOG_LEVEL."""

        with patch.dict(
            os.environ, {_LOG_LEVEL_ENV_VAR: "", "LOG_LEVEL": ""}, clear=True
        ):
            result = _get_env_log_level()
            assert result == _DEFAULT_LOG_LEVEL

    def test_unset_nemo_evaluator_log_level_falls_back_to_log_level(self):
        """Test that unset NEMO_EVALUATOR_LOG_LEVEL properly falls back to LOG_LEVEL."""

        # Only LOG_LEVEL is set - NEMO_EVALUATOR_LOG_LEVEL is not in environment
        with patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"}, clear=True):
            result = _get_env_log_level()
            assert result == "CRITICAL"

    def test_log_level_env_var_fallback_priority(self):
        """Test the complete fallback chain: NEMO_EVALUATOR_LOG_LEVEL -> LOG_LEVEL -> _DEFAULT_LOG_LEVEL."""

        # Test 1: Only LOG_LEVEL set
        with patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"}, clear=True):
            result = _get_env_log_level()
            assert result == "CRITICAL"

        # Test 2: Both set - NEMO_EVALUATOR_LOG_LEVEL takes precedence
        with patch.dict(
            os.environ,
            {_LOG_LEVEL_ENV_VAR: "DEBUG", "LOG_LEVEL": "CRITICAL"},
            clear=True,
        ):
            result = _get_env_log_level()
            assert result == "DEBUG"

        # Test 3: Neither set - defaults to _DEFAULT_LOG_LEVEL
        with patch.dict(os.environ, {}, clear=True):
            result = _get_env_log_level()
            assert result == _DEFAULT_LOG_LEVEL


class TestRedactProcessor:
    """Test cases for redact_processor function."""

    def _is_masked(self, value: str, original: str) -> bool:
        if value == "[REDACTED]":
            return True
        # Accept partial mask format like 'ab…yz'
        return (
            len(original) > 10
            and len(value) >= 5
            and value[:2] == original[:2]
            and value[-2:] == original[-2:]
            and "…" in value
        )

    def test_masks_only_secrets(self):
        """Test that redact_processor masks only secret keys, not other fields."""
        original = {
            "API_KEY": "abcdef123456",
            "password": "supersecret",
            "event": "keep this",
            "level": "keep this",
            "timestamp": "keep this",
            "note": "keep this",
        }
        red = redact_processor(None, "info", dict(original))
        assert self._is_masked(red["API_KEY"], original["API_KEY"])  # masked
        assert self._is_masked(red["password"], original["password"])  # masked
        assert red["event"] == "keep this"  # untouched
        assert red["level"] == "keep this"  # untouched
        assert red["timestamp"] == "keep this"  # untouched
        assert red["note"] == "keep this"  # untouched

    def test_normalizes_key_names(self):
        """Test that redact_processor normalizes key names before checking."""
        original = {
            "proxy-authorization": "abcdef123456",
            "judge-api-key": "abcdef123456",
            "NVIDIA_API_KEY": "abcdef123456",
            "Access-Key": "abcdef123456",
            "private key": "abcdef123456",
            "gitlab_token": "abcdef123456",
            "openai_client_secret": "abcdef",
            "password_to_encrypt": "abcdef",
            "my_pwd": "abcdef",
            "exceldb_pass_wd": "abcdef",
        }
        red = redact_processor(None, "info", dict(original))
        for k in original:
            assert self._is_masked(red[k], original[k])

    def test_redacts_nested_headers(self):
        """Test that redact_processor redacts nested headers structure."""
        original = {
            "headers": {
                "Authorization": "Bearer xyz123",
                "X-API-Key": "abcdef123456",
                "Content-Type": "application/json",
            }
        }
        red = redact_processor(None, "info", dict(original))
        assert self._is_masked(
            red["headers"]["Authorization"], original["headers"]["Authorization"]
        )  # type: ignore[index]
        assert self._is_masked(
            red["headers"]["X-API-Key"], original["headers"]["X-API-Key"]
        )  # type: ignore[index]
        assert red["headers"]["Content-Type"] == "application/json"  # type: ignore[index]

    def test_disable_env(self, monkeypatch: pytest.MonkeyPatch):
        """Test that LOG_DISABLE_REDACTION environment variable disables redaction."""
        monkeypatch.setenv("LOG_DISABLE_REDACTION", "1")
        original = {"API_KEY": "abcdef123456", "note": "ok"}
        red = redact_processor(None, "info", dict(original))
        assert red == original
        monkeypatch.delenv("LOG_DISABLE_REDACTION", raising=False)

    def test_allowlisted_keys_not_redacted(self):
        """Test that keys containing allowlisted substrings are not redacted."""
        original = {
            "max_new_tokens": "should_not_be_redacted",
            "limit_tokens": "should_not_be_redacted",
            "api_token": "booooo",
        }
        red = redact_processor(None, "info", dict(original))

        # Allowlisted keys should not be redacted
        assert red["max_new_tokens"] == "should_not_be_redacted"
        assert red["limit_tokens"] == "should_not_be_redacted"

        # Normal sensitive keys should still be redacted
        assert self._is_masked(red["api_token"], original["api_token"])
