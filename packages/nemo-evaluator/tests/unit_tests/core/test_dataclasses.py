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

"""Tests for API dataclasses."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)


def test_api_key_field_removed():
    """Test that the deprecated api_key field has been fully removed."""
    with pytest.raises(ValidationError, match="extra"):
        ApiEndpoint(api_key="SOME_KEY")


@pytest.mark.parametrize(
    "endpoint_type,expected_type,expect_warning",
    [
        ("vlm", "chat", True),
        ("chat", "chat", False),
        ("completions", "completions", False),
        (None, None, False),
    ],
)
def test_endpoint_vlm_type_deprecation(endpoint_type, expected_type, expect_warning):
    """Test that 'vlm' endpoint type is deprecated and converted to 'chat'."""
    kwargs = {"model_id": "my-model", "url": "http://localhost:8000"}
    if endpoint_type is not None:
        kwargs["type"] = endpoint_type

    if expect_warning:
        with pytest.warns(DeprecationWarning, match="Endpoint type 'vlm' was removed"):
            config = ApiEndpoint(**kwargs)
    else:
        config = ApiEndpoint(**kwargs)

    assert config.type == expected_type


def test_render_command_api_key_alias_for_framework_templates():
    """Test that render_command injects api_key as an alias of api_key_name.

    TODO: Remove once nvidia-simple-evals framework.yml templates are updated
    to use api_key_name. Their Jinja2 command templates still reference
    target.api_endpoint.api_key, which was removed from ApiEndpoint.
    """
    evaluation = Evaluation(
        command=(
            "{% if target.api_endpoint.api_key is not none %}"
            "export API_KEY=${{target.api_endpoint.api_key}} && "
            "{% endif %}echo done"
        ),
        framework_name="test",
        pkg_name="test_pkg",
        config=EvaluationConfig(
            type="test_eval",
            output_dir="/tmp/test",
            params=ConfigParams(),
        ),
        target=EvaluationTarget(
            api_endpoint=ApiEndpoint(
                api_key_name="MY_API_KEY",
                model_id="test-model",
                url="http://localhost:8000",
            )
        ),
    )
    rendered = evaluation.render_command()
    assert "export API_KEY=$MY_API_KEY" in rendered


def test_vlm_endpoint_type_deprecation_removal_reminder():
    """Test that fails after 3 months to remind us to remove the deprecated 'vlm' endpoint type.

    This test will start failing on June 6, 2026 (3 months after March 6, 2026).
    When this test fails, it's time to:
    1. Remove the 'handle_vlm_type_deprecation' validator from EndpointModelConfig
    2. Remove the call to _handle_vlm_type_deprecation from ApiEndpoint.handle_api_key_deprecation
    3. Remove the _handle_vlm_type_deprecation helper function
    4. Remove this test
    """
    # Deprecation date: March 6, 2026
    deprecation_date = datetime(2026, 3, 6)
    removal_date = datetime(2026, 6, 6)  # 3 months later
    current_date = datetime.now()

    assert current_date < removal_date, (
        f"Time to remove the deprecated 'vlm' endpoint type! "
        f"The 3-month deprecation period has ended (started {deprecation_date.strftime('%Y-%m-%d')}, "
        f"removal due {removal_date.strftime('%Y-%m-%d')}). "
        f"Please remove:\n"
        f"1. The 'vlm' handling from _handle_vlm_type_deprecation in api_dataclasses.py\n"
        f"2. The 'handle_vlm_type_deprecation' validator from EndpointModelConfig\n"
        f"3. The call to _handle_vlm_type_deprecation from ApiEndpoint.handle_api_key_deprecation\n"
        f"4. The _handle_vlm_type_deprecation helper function\n"
        f"5. This test function (test_vlm_endpoint_type_deprecation_removal_reminder)\n"
    )
