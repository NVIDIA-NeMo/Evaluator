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
"""Tests for mapping URL validation."""

import os
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from nemo_evaluator_launcher.common.mapping import MAPPING_URL, _download_latest_mapping


def test_mapping_url_contains_main():
    """Test that MAPPING_URL contains 'main' substring in the branch path."""
    # Check that the URL contains 'main' in the branch part
    assert "NVIDIA-NeMo/Eval/main" in MAPPING_URL, (
        f"MAPPING_URL '{MAPPING_URL}' must contain 'NVIDIA-NeMo/Eval/main'"
    )


def test_mapping_url_is_reachable(enable_network):
    """Test that MAPPING_URL is reachable and returns valid TOML."""
    # Get GitLab token from environment
    gitlab_token = os.environ.get("GITLAB_TOKEN", "")
    if not gitlab_token:
        pytest.skip("GITLAB_TOKEN not set, skipping network test")

    # Test the actual download function
    result = _download_latest_mapping()
    assert result is not None, f"Failed to download mapping from '{MAPPING_URL}'"
    assert len(result) > 0, "Downloaded mapping should not be empty"
    # Test the content
    mapping_str = result.decode("utf-8")
    _ = tomllib.loads(mapping_str)
