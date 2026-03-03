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

from nemo_evaluator_launcher.cli.status import Cmd


class TestFormatRequestsProcessed:
    """Tests for Cmd._format_requests_processed static method."""

    def test_int_request_count(self):
        assert Cmd._format_requests_processed(1234) == "1234"

    def test_int_zero(self):
        assert Cmd._format_requests_processed(0) == "0"

    def test_none_returns_dash(self):
        assert Cmd._format_requests_processed(None) == "-"

    def test_dict_wrapped_int(self):
        """Local executor wraps progress in dict(progress=N)."""
        job = {"progress": {"progress": 42}}
        progress_data = job.get("progress")
        if isinstance(progress_data, dict):
            progress_data = progress_data.get("progress")
        assert Cmd._format_requests_processed(progress_data) == "42"
