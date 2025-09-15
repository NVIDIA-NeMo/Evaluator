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

from nemo_evaluator_launcher.common.mapping import MAPPING_URL


def test_mapping_url_contains_main():
    """Test that MAPPING_URL contains 'main' substring in the branch path."""
    # Check that the URL contains 'main' in the branch part (after /raw/)
    assert "ref=main" in MAPPING_URL, (
        f"MAPPING_URL '{MAPPING_URL}' must contain '/raw/main/'"
    )
