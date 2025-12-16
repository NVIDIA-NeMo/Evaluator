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

from nemo_evaluator.api.api_dataclasses import ApiEndpoint


@pytest.mark.parametrize(
    "api_key,api_key_name",
    [
        (None, None),
        (None, "MY_API_KEY_ENV"),
        ("MY_API_KEY_ENV", None),
        ("MY_API_KEY_ENV1", "MY_API_KEY_ENV2"),
    ],
)
def test_api_key_deprecation(api_key, api_key_name):
    """Test that api_key deprecation works correctly."""
    endpoint = None
    if api_key is not None and api_key_name is None:
        with pytest.warns(DeprecationWarning, match="'api_key' is deprecated"):
            endpoint = ApiEndpoint(api_key=api_key)
    elif api_key is None and api_key_name is not None:
        endpoint = ApiEndpoint(api_key_name=api_key_name)
    elif api_key is not None and api_key_name is not None:
        with pytest.raises(
            ValueError, match="Both 'api_key' and 'api_key_name' are set"
        ):
            ApiEndpoint(api_key=api_key, api_key_name=api_key_name)
    else:
        endpoint = ApiEndpoint(api_key=api_key, api_key_name=api_key_name)
    if endpoint is not None:
        assert (endpoint.api_key_name == api_key_name or api_key) or (
            endpoint.api_key is None
        )
        assert (endpoint.api_key == api_key) or (endpoint.api_key is None)


def test_api_key_deprecation_removal_reminder():
    """Test that fails after 3 months to remind us to remove the deprecated api_key parameter.

    This test will start failing on March 15, 2026 (3 months after December 15, 2025).
    When this test fails, it's time to:
    1. Remove the 'api_key' field from ApiEndpoint
    2. Remove the 'handle_api_key_deprecation' validator
    3. Remove all related deprecation handling code
    4. Remove this test
    """
    # Deprecation date: December 15, 2025
    deprecation_date = datetime(2025, 12, 15)
    removal_date = datetime(2026, 3, 15)  # 3 months later
    current_date = datetime.now()

    assert current_date < removal_date, (
        f"Time to remove the deprecated 'api_key' parameter! "
        f"The 3-month deprecation period has ended (started {deprecation_date.strftime('%Y-%m-%d')}, "
        f"removal due {removal_date.strftime('%Y-%m-%d')}). "
        f"Please remove:\n"
        f"1. The 'api_key' field from ApiEndpoint class in api_dataclasses.py\n"
        f"2. The 'handle_api_key_deprecation' model_validator method\n"
        f"3. The 'warnings' import if no longer needed\n"
        f"4. This test function (test_api_key_deprecation_removal_reminder)\n"
        f"5. Update any documentation that mentions 'api_key'\n"
    )
