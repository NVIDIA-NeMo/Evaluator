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

"""Test module for logging utils."""

from nemo_evaluator.logging.utils import MainConsoleRenderer


def test_colors_disabled():
    """Test that colors are disabled when colors=False."""
    renderer = MainConsoleRenderer(colors=False)

    event_dict = {
        "timestamp": "2025-01-01T12:00:00.000",
        "event": "test message",
        "level": "info",
    }
    output = renderer(None, "info", event_dict)

    # Should not contain ANSI color codes
    assert "\033[" not in output


def test_colors_enabled():
    """Test that colors are enabled when colors=True."""
    renderer = MainConsoleRenderer(colors=True)

    event_dict = {
        "timestamp": "2025-01-01T12:00:00.000",
        "event": "test message",
        "level": "info",
    }
    output = renderer(None, "info", event_dict)

    # Should contain ANSI color codes
    assert "\033[" in output
