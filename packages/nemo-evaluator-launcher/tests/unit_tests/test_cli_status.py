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
"""Tests for the status CLI command, including progress formatting."""

import pytest

from nemo_evaluator_launcher.cli.status import Cmd


class TestFormatProgress:
    """Tests for Cmd._format_progress static method."""

    def test_float_zero(self):
        assert Cmd._format_progress(0.0) == "0.0%"

    def test_float_half(self):
        assert Cmd._format_progress(0.5) == "50.0%"

    def test_float_full(self):
        assert Cmd._format_progress(1.0) == "100.0%"

    def test_float_fractional(self):
        assert Cmd._format_progress(0.753) == "75.3%"

    def test_float_rounds_correctly(self):
        assert Cmd._format_progress(0.9999) == "100.0%"

    def test_int_raw_count(self):
        assert Cmd._format_progress(1234) == "1234 samples"

    def test_int_zero(self):
        assert Cmd._format_progress(0) == "0 samples"

    def test_unknown_string(self):
        assert Cmd._format_progress("unknown") == "-"

    def test_none(self):
        assert Cmd._format_progress(None) == "-"

    def test_missing_key_defaults_to_dash(self):
        """Simulates job dict without progress key — .get() returns None."""
        job = {"status": "running"}
        assert Cmd._format_progress(job.get("progress")) == "-"
