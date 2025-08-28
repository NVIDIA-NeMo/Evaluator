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

"""Tests for PostEvalReportHook functionality."""

import json

from nemo_evaluator.adapters.caching.diskcaching import Cache
from nemo_evaluator.adapters.reports.post_eval_report_hook import (
    PostEvalReportHook,
    ReportType,
)
from nemo_evaluator.adapters.types import AdapterGlobalContext


def test_post_eval_report_hook_creation():
    """Test that PostEvalReportHook can be created with minimal configuration."""
    params = PostEvalReportHook.Params(report_types=[ReportType.HTML, ReportType.JSON])

    hook = PostEvalReportHook(params)
    assert ReportType.HTML in hook.report_types
    assert ReportType.JSON in hook.report_types


def test_post_eval_report_hook_post_eval_hook(tmpdir):
    """Test that PostEvalReportHook generates reports correctly."""
    # Create test cache data
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"
    headers_dir = cache_dir / "headers"

    responses_dir.mkdir()
    requests_dir.mkdir()
    headers_dir.mkdir()

    # Create cache instances and add test data
    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))
    headers_cache = Cache(directory=str(headers_dir))

    test_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    test_response = {"choices": [{"message": {"content": "test response"}}]}
    test_headers = {"content-type": "application/json"}

    # Store test data
    cache_key = "test_key_123"
    requests_cache[cache_key] = test_request
    responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")
    headers_cache[cache_key] = test_headers

    # Create hook
    params = PostEvalReportHook.Params(report_types=[ReportType.HTML, ReportType.JSON])
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook
    hook.post_eval_hook(context)

    # Verify HTML report was created
    html_file = tmpdir / "report.html"
    assert html_file.exists()

    # Verify JSON report was created
    json_file = tmpdir / "report.json"
    assert json_file.exists()

    # Check HTML content
    html_content = html_file.read()
    assert "test-model" in html_content
    assert "test response" in html_content
    assert "test_key_123" in html_content
    assert "http://test.example.com/api" in html_content

    # Check JSON content
    json_content = json.loads(json_file.read())
    assert len(json_content) == 1
    assert json_content[0]["cache_key"] == "test_key_123"
    assert json_content[0]["endpoint"] == "http://test.example.com/api"


def test_post_eval_report_hook_html_only(tmpdir):
    """Test that PostEvalReportHook generates only HTML when configured."""
    # Create test cache data
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"

    responses_dir.mkdir()
    requests_dir.mkdir()

    # Create cache instances and add test data
    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))

    test_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    test_response = {"choices": [{"message": {"content": "test response"}}]}

    # Store test data
    cache_key = "test_key_456"
    requests_cache[cache_key] = test_request
    responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")

    # Create hook with HTML only
    params = PostEvalReportHook.Params(report_types=[ReportType.HTML])
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook
    hook.post_eval_hook(context)

    # Verify HTML report was created
    html_file = tmpdir / "report.html"
    assert html_file.exists()

    # Verify JSON report was NOT created
    json_file = tmpdir / "report.json"
    assert not json_file.exists()


def test_post_eval_report_hook_json_only(tmpdir):
    """Test that PostEvalReportHook generates only JSON when configured."""
    # Create test cache data
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"

    responses_dir.mkdir()
    requests_dir.mkdir()

    # Create cache instances and add test data
    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))

    test_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    test_response = {"choices": [{"message": {"content": "test response"}}]}

    # Store test data
    cache_key = "test_key_789"
    requests_cache[cache_key] = test_request
    responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")

    # Create hook with JSON only
    params = PostEvalReportHook.Params(report_types=[ReportType.JSON])
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook
    hook.post_eval_hook(context)

    # Verify HTML report was NOT created
    html_file = tmpdir / "report.html"
    assert not html_file.exists()

    # Verify JSON report was created
    json_file = tmpdir / "report.json"
    assert json_file.exists()

    # Check JSON content
    json_content = json.loads(json_file.read())
    assert len(json_content) == 1
    assert json_content[0]["cache_key"] == "test_key_789"


def test_post_eval_report_hook_no_cache_data(tmpdir):
    """Test that PostEvalReportHook handles empty cache gracefully."""
    # Create hook
    params = PostEvalReportHook.Params(report_types=[ReportType.HTML, ReportType.JSON])
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook (should not create any files since no cache data)
    hook.post_eval_hook(context)

    # Verify no files were created
    html_file = tmpdir / "report.html"
    json_file = tmpdir / "report.json"
    assert not html_file.exists()
    assert not json_file.exists()


def test_post_eval_report_hook_default_configuration(tmpdir):
    """Test that PostEvalReportHook uses default configuration (HTML only)."""
    # Create test cache data
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()
    responses_dir = cache_dir / "responses"
    requests_dir = cache_dir / "requests"

    responses_dir.mkdir()
    requests_dir.mkdir()

    # Create cache instances and add test data
    responses_cache = Cache(directory=str(responses_dir))
    requests_cache = Cache(directory=str(requests_dir))

    test_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
    }
    test_response = {"choices": [{"message": {"content": "test response"}}]}

    # Store test data
    cache_key = "test_key_default"
    requests_cache[cache_key] = test_request
    responses_cache[cache_key] = json.dumps(test_response).encode("utf-8")

    # Create hook with default configuration (no report_types specified)
    params = PostEvalReportHook.Params()
    hook = PostEvalReportHook(params)

    # Create context
    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com/api"
    )

    # Run the hook
    hook.post_eval_hook(context)

    # Verify HTML report was created (default)
    html_file = tmpdir / "report.html"
    assert html_file.exists()

    # Verify JSON report was NOT created (not in default)
    json_file = tmpdir / "report.json"
    assert not json_file.exists()
