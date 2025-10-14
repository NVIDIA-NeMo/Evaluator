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

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)


def test_render_command_double_call_logs_error(caplog):
    """Test that calling render_command() twice logs an error message."""
    # Create an evaluation instance
    evaluation = Evaluation(
        command="echo {{framework_name}} {{pkg_name}}",
        framework_name="test_framework",
        pkg_name="test_package",
        config=EvaluationConfig(output_dir="/tmp/test"),
        target=EvaluationTarget(api_endpoint=ApiEndpoint(url="http://test.com")),
    )

    # First call should work normally
    result1 = evaluation.render_command()
    assert result1 == "echo test_framework test_package"

    # Second call should log an error and add warning prefix
    result2 = evaluation.render_command()
    expected_warning = 'echo("Warning, this eval command is potentially different from the actually used one, see logs") && echo test_framework test_package'
    assert result2 == expected_warning

    # Check that error was logged
    expected_error_msg = (
        "It is error to call `render_command` more than once. Expect command "
        "to be logged with discrepancies or refactor the code."
    )
    assert any(expected_error_msg in record.message for record in caplog.records)


def test_render_command_basic_functionality():
    """Test that render_command() works correctly with jinja2 templates."""
    evaluation = Evaluation(
        command="echo {{framework_name}} {{pkg_name}} --output {{config.output_dir}}",
        framework_name="my_framework",
        pkg_name="my_package",
        config=EvaluationConfig(output_dir="/path/to/output"),
        target=EvaluationTarget(api_endpoint=ApiEndpoint(url="http://api.com")),
    )

    result = evaluation.render_command()
    expected = "echo my_framework my_package --output /path/to/output"
    assert result == expected


def test_render_command_recursive_rendering():
    """Test that render_command() handles recursive jinja2 template rendering."""
    evaluation = Evaluation(
        command="{{framework_name}}_{{pkg_name}}",
        framework_name="test",
        pkg_name="processed",
        config=EvaluationConfig(output_dir="/tmp"),
        target=EvaluationTarget(api_endpoint=ApiEndpoint(url="http://test.com")),
    )

    result = evaluation.render_command()
    assert result == "test_processed"


def test_render_command_missing_variable_error():
    """Test that render_command() raises ValueError for missing template variables."""
    evaluation = Evaluation(
        command="echo {{missing_variable}}",
        framework_name="test",
        pkg_name="test",
        config=EvaluationConfig(output_dir="/tmp"),
        target=EvaluationTarget(api_endpoint=ApiEndpoint(url="http://test.com")),
    )

    with pytest.raises(ValueError, match="Missing required configuration field"):
        evaluation.render_command()
