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
"""Tests for LLM-as-judge scoring primitives."""

from nemo_evaluator.scoring.judge import build_judge_prompt, parse_judge_response


class TestParseJudgeResponse:
    def test_valid_json(self):
        result = parse_judge_response('{"score": 4, "reasoning": "Good answer"}')
        assert result["score"] == 4
        assert result["normalized"] == 0.8
        assert result["reasoning"] == "Good answer"

    def test_score_regex_fallback(self):
        result = parse_judge_response("I'd give this a score of \"score\": 3 because it's ok")
        assert result["score"] == 3
        assert result["normalized"] == 0.6

    def test_fraction_fallback(self):
        result = parse_judge_response("Rating: 4/5")
        assert result["score"] == 4
        assert result["normalized"] == 0.8

    def test_unparseable_returns_zero(self):
        result = parse_judge_response("This is great!")
        assert result["score"] == 0
        assert result.get("parse_error") is True

    def test_score_clamped_to_max(self):
        result = parse_judge_response('{"score": 10, "reasoning": "perfect"}', max_score=5)
        assert result["score"] == 5
        assert result["normalized"] == 1.0


class TestBuildJudgePrompt:
    def test_includes_instruction_and_response(self):
        prompt = build_judge_prompt("Summarize this", "Here is my summary")
        assert "Summarize this" in prompt
        assert "Here is my summary" in prompt

    def test_includes_reference_when_provided(self):
        prompt = build_judge_prompt("Q", "A", expected="Reference answer text")
        assert "Reference answer text" in prompt

    def test_no_reference_section_without_expected(self):
        prompt = build_judge_prompt("Q", "A")
        assert "Reference answer" not in prompt
