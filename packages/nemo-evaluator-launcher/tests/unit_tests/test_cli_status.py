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

    def test_unknown_string(self):
        assert Cmd._format_requests_processed("unknown") == "-"

    def test_none(self):
        assert Cmd._format_requests_processed(None) == "-"

    def test_float_falls_through_to_dash(self):
        """Floats are no longer expected — they should display as dash."""
        assert Cmd._format_requests_processed(0.75) == "-"

    def test_missing_key_defaults_to_dash(self):
        """Simulates job dict without progress key — .get() returns None."""
        job = {"status": "running"}
        assert Cmd._format_requests_processed(job.get("progress")) == "-"


class TestExtractProgressFromJob:
    """Tests that progress is correctly extracted from job dicts (dict wrapper)."""

    @staticmethod
    def _extract_progress(job: dict) -> object:
        """Reproduce the extraction logic used in Cmd.execute."""
        progress_data = job.get("progress")
        if isinstance(progress_data, dict):
            progress_data = progress_data.get("progress")
        return progress_data

    def test_dict_wrapped_int(self):
        job = {"progress": {"progress": 42}}
        assert Cmd._format_requests_processed(self._extract_progress(job)) == "42"

    def test_dict_wrapped_none(self):
        job = {"progress": {"progress": None}}
        assert Cmd._format_requests_processed(self._extract_progress(job)) == "-"

    def test_no_progress_key(self):
        job = {"status": "running"}
        assert Cmd._format_requests_processed(self._extract_progress(job)) == "-"
