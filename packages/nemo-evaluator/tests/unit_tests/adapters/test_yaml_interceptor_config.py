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

"""Tests for YAML-based interceptor configuration."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    RequestInterceptor,
)

from .testing_utils import get_request_interceptors, reset_registry


def test_yaml_config_validation_success():
    reset_registry()

    @register_for_adapter(
        name="test_yaml_interceptor",
        description="Test interceptor for YAML configuration",
    )
    class TestYamlInterceptor(RequestInterceptor):
        class Params(BaseModel):
            api_key_name: str = Field(
                ..., description="API key name for authentication"
            )
            timeout: int = Field(
                default=30, ge=1, le=300, description="Request timeout"
            )
            retry_count: int = Field(
                default=3, ge=0, le=5, description="Number of retries"
            )
            endpoints: list[str] = Field(
                default_factory=list, description="List of endpoints"
            )

        def __init__(self, params: Params):
            self.api_key_name = params.api_key_name
            self.timeout = params.timeout
            self.retry_count = params.retry_count
            self.endpoints = params.endpoints

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    yaml_config = """
    api_key_name: "test-api-key-12345"
    timeout: 60
    retry_count: 2
    endpoints:
      - "https://api1.example.com"
      - "https://api2.example.com"
    """
    config_dict = yaml.safe_load(yaml_config)
    interceptors = get_request_interceptors({"test_yaml_interceptor": config_dict})
    assert len(interceptors) == 1
    interceptor = interceptors[0]
    assert interceptor.api_key_name == "test-api-key-12345"
    assert interceptor.timeout == 60
    assert interceptor.retry_count == 2
    assert len(interceptor.endpoints) == 2


def test_yaml_config_validation_failure():
    reset_registry()

    @register_for_adapter(
        name="test_yaml_validation",
        description="Test interceptor for YAML validation",
    )
    class TestYamlValidationInterceptor(RequestInterceptor):
        class Params(BaseModel):
            api_key_name: str = Field(
                ..., description="API key name for authentication"
            )
            timeout: int = Field(
                default=30, ge=1, le=300, description="Request timeout"
            )

        def __init__(self, params: Params):
            self.api_key_name = params.api_key_name
            self.timeout = params.timeout

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    yaml_config = """
    timeout: 60
    # Missing api_key_name
    """
    config_dict = yaml.safe_load(yaml_config)
    with pytest.raises(ValidationError, match="api_key_name"):
        get_request_interceptors({"test_yaml_validation": config_dict})

    yaml_config_wrong_type = """
    api_key_name: "test-key"
    timeout: "not-an-int"
    """
    config_dict_wrong_type = yaml.safe_load(yaml_config_wrong_type)
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        get_request_interceptors({"test_yaml_validation": config_dict_wrong_type})


def test_yaml_file_loading():
    reset_registry()

    @register_for_adapter(
        name="test_file_interceptor",
        description="Test interceptor for file loading",
    )
    class TestFileInterceptor(RequestInterceptor):
        class Params(BaseModel):
            cache_dir: str = Field(..., description="Cache directory path")
            max_size: int = Field(default=1000, description="Maximum cache size")
            enable_compression: bool = Field(
                default=False, description="Enable compression"
            )

        def __init__(self, params: Params):
            self.cache_dir = params.cache_dir
            self.max_size = params.max_size
            self.enable_compression = params.enable_compression

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    yaml_content = """
    test_file_interceptor:
      cache_dir: "/tmp/test_cache"
      max_size: 2000
      enable_compression: true
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_file_path = f.name
    try:
        with open(yaml_file_path, "r") as f:
            config = yaml.safe_load(f)
        interceptors = get_request_interceptors(config)
        assert len(interceptors) == 1
        interceptor = interceptors[0]
        assert interceptor.cache_dir == "/tmp/test_cache"
        assert interceptor.max_size == 2000
        assert interceptor.enable_compression is True
    finally:
        Path(yaml_file_path).unlink()


def test_complex_yaml_configuration():
    reset_registry()

    class RetryConfig(BaseModel):
        max_retries: int = Field(default=3, ge=0, le=10)
        backoff_factor: float = Field(default=1.0, ge=0.1, le=10.0)
        timeout: int = Field(default=30, ge=1, le=300)

    @register_for_adapter(
        name="test_complex_interceptor",
        description="Test interceptor with complex configuration",
    )
    class TestComplexInterceptor(RequestInterceptor):
        class Params(BaseModel):
            api_key_name: str = Field(
                ..., description="API key name for authentication"
            )
            endpoints: list[str] = Field(
                default_factory=list, description="List of endpoints"
            )
            retry_config: RetryConfig = Field(
                default_factory=RetryConfig, description="Retry configuration"
            )
            headers: dict[str, str] = Field(
                default_factory=dict, description="Custom headers"
            )

        def __init__(self, params: Params):
            self.api_key_name = params.api_key_name
            self.endpoints = params.endpoints
            self.retry_config = params.retry_config
            self.headers = params.headers

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    yaml_config = """
    api_key_name: "complex-api-key-12345"
    endpoints:
      - "https://primary.example.com"
      - "https://secondary.example.com"
      - "https://fallback.example.com"
    retry_config:
      max_retries: 5
      backoff_factor: 2.0
      timeout: 60
    headers:
      X-Custom-Header: "custom-value"
      User-Agent: "test-agent/1.0"
      Accept: "application/json"
    """
    config_dict = yaml.safe_load(yaml_config)
    interceptors = get_request_interceptors({"test_complex_interceptor": config_dict})
    assert len(interceptors) == 1
    interceptor = interceptors[0]
    assert interceptor.api_key_name == "complex-api-key-12345"
    assert len(interceptor.endpoints) == 3
    assert interceptor.retry_config.max_retries == 5
    assert interceptor.retry_config.backoff_factor == 2.0
    assert interceptor.retry_config.timeout == 60
    assert len(interceptor.headers) == 3
    assert interceptor.headers["X-Custom-Header"] == "custom-value"


def test_multiple_interceptors_yaml_config():
    reset_registry()

    @register_for_adapter(
        name="auth_interceptor",
        description="Authentication interceptor",
    )
    class AuthInterceptor(RequestInterceptor):
        class Params(BaseModel):
            api_key_name: str = Field(..., description="API key name")

        def __init__(self, params: Params):
            self.api_key_name = params.api_key_name

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    @register_for_adapter(
        name="logging_interceptor",
        description="Logging interceptor",
    )
    class LoggingInterceptor(RequestInterceptor):
        class Params(BaseModel):
            log_level: str = Field(default="INFO", description="Log level")
            output_file: str = Field(default="/tmp/logs.txt", description="Output file")

        def __init__(self, params: Params):
            self.log_level = params.log_level
            self.output_file = params.output_file

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    yaml_config = """
    auth_interceptor:
      api_key_name: "multi-auth-key"
    logging_interceptor:
      log_level: "DEBUG"
      output_file: "/tmp/debug.log"
    """
    config_dict = yaml.safe_load(yaml_config)
    interceptors = get_request_interceptors(config_dict)
    assert len(interceptors) == 2
    auth_interceptor = next(i for i in interceptors if hasattr(i, "api_key_name"))
    assert auth_interceptor.api_key_name == "multi-auth-key"
    logging_interceptor = next(i for i in interceptors if hasattr(i, "log_level"))
    assert logging_interceptor.log_level == "DEBUG"
    assert logging_interceptor.output_file == "/tmp/debug.log"


def test_yaml_config_with_defaults():
    reset_registry()

    @register_for_adapter(
        name="test_defaults_interceptor",
        description="Test interceptor with defaults",
    )
    class TestDefaultsInterceptor(RequestInterceptor):
        class Params(BaseModel):
            required_field: str = Field(..., description="Required field")
            optional_field: int = Field(
                default=42, description="Optional field with default"
            )
            optional_list: list[str] = Field(
                default_factory=list, description="Optional list"
            )

        def __init__(self, params: Params):
            self.required_field = params.required_field
            self.optional_field = params.optional_field
            self.optional_list = params.optional_list

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    yaml_config = """
    required_field: "only-required-field"
    # optional_field and optional_list will use defaults
    """
    config_dict = yaml.safe_load(yaml_config)
    interceptors = get_request_interceptors({"test_defaults_interceptor": config_dict})
    assert len(interceptors) == 1
    interceptor = interceptors[0]
    assert interceptor.required_field == "only-required-field"
    assert interceptor.optional_field == 42
    assert interceptor.optional_list == []


def test_yaml_config_error_handling():
    reset_registry()

    @register_for_adapter(
        name="test_error_interceptor",
        description="Test interceptor for error handling",
    )
    class TestErrorInterceptor(RequestInterceptor):
        class Params(BaseModel):
            field: str = Field(..., description="Required field")

        def __init__(self, params: Params):
            self.field = params.field

        def intercept_request(
            self, ar: AdapterRequest, context: AdapterGlobalContext
        ) -> AdapterRequest:
            return ar

    malformed_yaml = """
    test_error_interceptor:
      field: "valid-value"
      invalid_field: "this-should-be-ignored"
    """
    config_dict = yaml.safe_load(malformed_yaml)
    # This should not raise an error - extra fields should be ignored
    interceptors = get_request_interceptors(config_dict)
    assert len(interceptors) == 1
    interceptor = interceptors[0]
    assert interceptor.field == "valid-value"
