# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for nemo_evaluator.scoring.code_execution."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

from nemo_evaluator.scoring.code_execution import (
    _extract_code,
    code_sandbox,
    code_sandbox_async,
)
from nemo_evaluator.scoring.types import ScorerInput


class TestExtractCode:
    def test_python_fence(self):
        sample = ScorerInput(response="```python\nprint(1)\n```", target="")
        assert _extract_code(sample) == "print(1)\n"

    def test_plain_fence(self):
        sample = ScorerInput(response="```\nx = 1\n```", target="")
        assert _extract_code(sample) == "x = 1\n"

    def test_no_fence_returns_raw(self):
        sample = ScorerInput(response="print('hi')", target="")
        assert _extract_code(sample) == "print('hi')"

    def test_multiple_fences_picks_first(self):
        resp = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
        sample = ScorerInput(response=resp, target="")
        assert "first" in _extract_code(sample)

    def test_empty_response(self):
        sample = ScorerInput(response="", target="")
        assert _extract_code(sample) == ""

    def test_multiline_code(self):
        code = "def f():\n    return 1\n\nprint(f())\n"
        sample = ScorerInput(response=f"```python\n{code}```", target="")
        assert _extract_code(sample) == code


class TestCodeSandbox:
    @patch("subprocess.run")
    def test_passing_code(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        sample = ScorerInput(
            response="```python\ndef solution():\n    return 1\n```",
            target="",
            metadata={"_prompt": "", "_test": "def check(fn): assert fn() == 1", "_entry_point": "solution"},
        )
        result = code_sandbox(sample)
        assert result["correct"] is True
        assert "extracted" in result
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_failing_code(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        sample = ScorerInput(response="pass", target="", metadata={"_prompt": "", "_test": "", "_entry_point": "f"})
        result = code_sandbox(sample)
        assert result["correct"] is False

    @patch("subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
        sample = ScorerInput(response="while True: pass", target="", metadata={})
        result = code_sandbox(sample)
        assert result["correct"] is False

    @patch("subprocess.run")
    def test_docker_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("docker not found")
        sample = ScorerInput(response="print(1)", target="", metadata={})
        result = code_sandbox(sample)
        assert result["correct"] is False

    @patch("subprocess.run")
    def test_extracted_truncated_to_500(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        long_code = "x" * 1000
        sample = ScorerInput(response=long_code, target="", metadata={})
        result = code_sandbox(sample)
        assert len(result["extracted"]) == 500


class TestCodeSandboxAsync:
    async def test_passing(self):
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0)
        sample = ScorerInput(
            response="def solution(): return 1",
            target="",
            metadata={"_prompt": "", "_test": "def check(fn): assert fn() == 1", "_entry_point": "solution"},
        )
        result = await code_sandbox_async(sample, sandbox)
        assert result["correct"] is True
        sandbox.exec.assert_called_once()

    async def test_failing(self):
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=1)
        sample = ScorerInput(response="bad code", target="", metadata={})
        result = await code_sandbox_async(sample, sandbox)
        assert result["correct"] is False

    async def test_custom_timeout(self):
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0)
        sample = ScorerInput(response="pass", target="", metadata={})
        await code_sandbox_async(sample, sandbox, timeout=60.0)
        call_kwargs = sandbox.exec.call_args
        assert call_kwargs.kwargs.get("timeout_sec") == 60.0 or 60.0 in call_kwargs.args

    async def test_entry_point_fallback(self):
        sandbox = AsyncMock()
        sandbox.exec.return_value = MagicMock(return_code=0)
        sample = ScorerInput(response="x=1", target="", metadata={"entry_point": "my_func"})
        result = await code_sandbox_async(sample, sandbox)
        assert result["correct"] is True
        cmd = sandbox.exec.call_args[0][0]
        assert "my_func" in cmd
