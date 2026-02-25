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

"""Unit tests for BYOB LLM-as-Judge support (aligned with nemo-skills extra.judge)."""

import json

import pytest
from unittest.mock import MagicMock, patch

from nemo_evaluator.contrib.byob.decorators import ScorerInput
from nemo_evaluator.contrib.byob.judge import (
    JudgeConfig,
    judge_call,
    judge_score,
    parse_grade,
    render_judge_prompt,
)
from nemo_evaluator.contrib.byob.judge_templates import (
    DEFAULT_PATTERNS,
    DEFAULT_SCORE_MAPPINGS,
    TEMPLATES,
)

# Standard judge dict matching the nemo-skills convention
JUDGE_DICT = {"url": "http://judge:8000/v1", "model_id": "nemotron"}


# ---------------------------------------------------------------------------
# TestJudgeConfig
# ---------------------------------------------------------------------------


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass (aligned with nemo-skills schema)."""

    def test_construction_defaults(self):
        config = JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")
        assert config.url == "http://judge:8000/v1"
        assert config.model_id == "nemotron"
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.max_new_tokens == 4096
        assert config.api_key is None
        assert config.max_retries == 16
        assert config.request_timeout == 600

    def test_custom_params(self):
        config = JudgeConfig(
            url="http://judge:8000/v1",
            model_id="nemotron",
            temperature=0.3,
            max_new_tokens=512,
            api_key="JUDGE_KEY",
            top_p=0.9,
            parallelism=8,
        )
        assert config.temperature == 0.3
        assert config.max_new_tokens == 512
        assert config.api_key == "JUDGE_KEY"
        assert config.top_p == 0.9
        assert config.parallelism == 8

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("JUDGE_API_KEY", "secret-key-123")
        config = JudgeConfig(
            url="http://judge:8000/v1",
            model_id="nemotron",
            api_key="JUDGE_API_KEY",
        )
        assert config.resolve_api_key() == "secret-key-123"

    def test_resolve_api_key_missing_env(self):
        config = JudgeConfig(
            url="http://judge:8000/v1",
            model_id="nemotron",
            api_key="NONEXISTENT_KEY_XYZ",
        )
        assert config.resolve_api_key() is None

    def test_resolve_api_key_no_name(self):
        config = JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")
        assert config.resolve_api_key() is None

    def test_from_dict_nemo_skills_format(self):
        """Test from_dict with nemo-skills field names."""
        config = JudgeConfig.from_dict({
            "url": "https://inference-api.nvidia.com/v1",
            "model_id": "openai/gpt-4.1",
            "api_key": "JUDGE_API_KEY",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 4096,
            "parallelism": 16,
        })
        assert config.url == "https://inference-api.nvidia.com/v1"
        assert config.model_id == "openai/gpt-4.1"
        assert config.api_key == "JUDGE_API_KEY"
        assert config.max_new_tokens == 4096
        assert config.parallelism == 16

    def test_from_dict_defaults(self):
        config = JudgeConfig.from_dict({
            "url": "http://judge:8000/v1",
            "model_id": "nemotron",
        })
        assert config.temperature == 0.0
        assert config.max_new_tokens == 4096
        assert config.max_retries == 16

    def test_from_dict_legacy_field_names(self):
        """Test backward compat: api_key_name -> api_key, max_tokens -> max_new_tokens."""
        config = JudgeConfig.from_dict({
            "url": "http://judge:8000/v1",
            "model_id": "nemotron",
            "api_key_name": "MY_KEY",
            "max_tokens": 1024,
        })
        assert config.api_key == "MY_KEY"
        assert config.max_new_tokens == 1024

    def test_from_dict_extra_keys_ignored(self):
        config = JudgeConfig.from_dict({
            "url": "http://judge:8000/v1",
            "model_id": "nemotron",
            "extra_field": "should be ignored",
        })
        assert config.url == "http://judge:8000/v1"


# ---------------------------------------------------------------------------
# TestJudgeCall
# ---------------------------------------------------------------------------


class TestJudgeCall:
    """Tests for judge_call function."""

    @pytest.fixture
    def judge_config(self):
        return JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")

    def test_successful_call(self, judge_config):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "GRADE: C"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        result = judge_call(judge_config, "Test prompt", session=mock_session)
        assert result == "GRADE: C"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://judge:8000/v1/chat/completions"

    def test_call_payload_structure(self, judge_config):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "GRADE: C"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        judge_call(judge_config, "Test prompt", session=mock_session)

        call_kwargs = mock_session.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "nemotron"
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["max_tokens"] == 4096
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Test prompt"

    def test_call_with_api_key(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        config = JudgeConfig(
            url="http://judge:8000/v1",
            model_id="nemotron",
            api_key="MY_KEY",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "GRADE: C"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        judge_call(config, "Test prompt", session=mock_session)

        headers = mock_session.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer secret"

    def test_call_uses_request_timeout(self):
        config = JudgeConfig(
            url="http://judge:8000/v1", model_id="nemotron",
            request_timeout=30,
        )
        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("timeout")

        with pytest.raises(Exception):
            judge_call(config, "Test prompt", session=mock_session)

        # Verify timeout was passed
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["timeout"] == 30

    def test_call_http_error(self):
        import requests as req

        config = JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("500 Server Error")

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with pytest.raises(req.HTTPError):
            judge_call(config, "Test prompt", session=mock_session)


# ---------------------------------------------------------------------------
# TestParseGrade
# ---------------------------------------------------------------------------


class TestParseGrade:
    """Tests for parse_grade function."""

    def test_binary_correct(self):
        assert parse_grade("The answer is correct.\nGRADE: C", r"GRADE:\s*([CI])") == "C"

    def test_binary_incorrect(self):
        assert parse_grade("Wrong answer.\nGRADE: I", r"GRADE:\s*([CI])") == "I"

    def test_likert_scale(self):
        assert parse_grade("Good quality.\nGRADE: 4", r"GRADE:\s*([1-5])") == "4"

    def test_safety_safe(self):
        assert parse_grade("This is fine.\nGRADE: SAFE", r"GRADE:\s*(SAFE|UNSAFE)") == "SAFE"

    def test_safety_unsafe(self):
        assert parse_grade("Harmful content.\nGRADE: UNSAFE", r"GRADE:\s*(SAFE|UNSAFE)") == "UNSAFE"

    def test_yes_no(self):
        assert parse_grade("GRADE: YES", r"GRADE:\s*(YES|NO)") == "YES"

    def test_no_match_returns_none(self):
        assert parse_grade("No grade here at all", r"GRADE:\s*([CI])") is None

    def test_json_fallback(self):
        response = 'Some reasoning. {"grade": "C", "explanation": "correct"}'
        result = parse_grade(response, r"NONEXISTENT_PATTERN:\s*(\S+)")
        assert result == "C"

    def test_json_fallback_score_key(self):
        response = '{"score": 4}'
        result = parse_grade(response, r"NONEXISTENT:\s*(\S+)")
        assert result == "4"

    def test_no_match_no_json(self):
        result = parse_grade("Just some text without any grade", r"GRADE:\s*([CI])")
        assert result is None

    def test_grade_in_middle_of_text(self):
        response = "Let me think...\nGRADE: C\nThat's my verdict."
        assert parse_grade(response, r"GRADE:\s*([CI])") == "C"

    def test_partial_credit_pattern(self):
        assert parse_grade("GRADE: P", r"GRADE:\s*([CPI])") == "P"


# ---------------------------------------------------------------------------
# TestRenderJudgePrompt
# ---------------------------------------------------------------------------


class TestRenderJudgePrompt:
    """Tests for render_judge_prompt function."""

    def test_basic_substitution(self):
        template = "Q: {question}\nA: {response}\nRef: {reference}"
        result = render_judge_prompt(
            template,
            question="What is 2+2?",
            response="4",
            reference="4",
        )
        assert "What is 2+2?" in result
        assert "A: 4" in result

    def test_criteria_substitution(self):
        template = "Evaluate: {response}\n{criteria}"
        result = render_judge_prompt(
            template,
            response="Some answer",
            criteria="Be strict",
        )
        assert "Be strict" in result

    def test_missing_variable_raises(self):
        template = "Q: {question}\nA: {response}"
        with pytest.raises(KeyError):
            render_judge_prompt(template, question="Q1")

    def test_empty_criteria(self):
        template = "Q: {question}\n{criteria}"
        result = render_judge_prompt(template, question="Q1", criteria="")
        assert "Q: Q1" in result


# ---------------------------------------------------------------------------
# TestJudgeScore
# ---------------------------------------------------------------------------


class TestJudgeScore:
    """Tests for judge_score high-level entry point."""

    def _make_sample(self, judge=None, extra_config=None):
        config = extra_config or {}
        if judge:
            config["judge"] = judge
        return ScorerInput(
            response="The answer is 42",
            target="42",
            metadata={"question": "What is the answer?"},
            config=config,
        )

    def test_missing_judge_raises(self):
        sample = self._make_sample()
        with pytest.raises(ValueError, match="judge"):
            judge_score(sample)

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_binary_qa_correct(self, mock_call):
        mock_call.return_value = "The answer matches.\nGRADE: C"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "C"
        assert result["judge_score"] == 1.0

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_binary_qa_incorrect(self, mock_call):
        mock_call.return_value = "Wrong answer.\nGRADE: I"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "I"
        assert result["judge_score"] == 0.0

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_likert_5(self, mock_call):
        mock_call.return_value = "Good quality.\nGRADE: 4"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="likert_5")
        assert result["judge_grade"] == "4"
        assert result["judge_score"] == 0.8

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_partial_credit(self, mock_call):
        mock_call.return_value = "Partially correct.\nGRADE: P"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="binary_qa_partial")
        assert result["judge_grade"] == "P"
        assert result["judge_score"] == 0.5

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_safety_template(self, mock_call):
        mock_call.return_value = "The response is safe.\nGRADE: SAFE"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="safety")
        assert result["judge_grade"] == "SAFE"
        assert result["judge_score"] == 1.0

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_custom_template(self, mock_call):
        mock_call.return_value = "GRADE: YES"
        custom_template = "Is {response} correct for {question}?\n{reference}\n{criteria}\nGRADE: YES/NO"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(
            sample,
            template=custom_template,
            grade_pattern=r"GRADE:\s*(YES|NO)",
            score_mapping={"YES": 1.0, "NO": 0.0},
        )
        assert result["judge_grade"] == "YES"
        assert result["judge_score"] == 1.0

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_parse_error(self, mock_call):
        mock_call.return_value = "I cannot determine the grade"
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "PARSE_ERROR"
        assert result["judge_score"] == 0.0

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_call_error(self, mock_call):
        mock_call.side_effect = Exception("Connection refused")
        sample = self._make_sample(judge=JUDGE_DICT)
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "CALL_ERROR"
        assert result["judge_score"] == 0.0

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_criteria_passed_to_template(self, mock_call):
        mock_call.return_value = "GRADE: C"
        sample = self._make_sample(judge=JUDGE_DICT)
        judge_score(sample, template="binary_qa", criteria="Be strict about facts")
        call_args = mock_call.call_args
        prompt = call_args[0][1]
        assert "Be strict about facts" in prompt

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_judge_config_object_accepted(self, mock_call):
        mock_call.return_value = "GRADE: C"
        config = JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")
        sample = self._make_sample(extra_config={"judge": config})
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "C"

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_uses_internal_session(self, mock_call):
        """Test that _judge_session from config is used when available."""
        mock_call.return_value = "GRADE: C"
        mock_session = MagicMock()
        sample = self._make_sample(extra_config={
            "judge": JUDGE_DICT,
            "_judge_session": mock_session,
        })
        judge_score(sample, template="binary_qa")
        call_kwargs = mock_call.call_args
        assert call_kwargs[1].get("session") is mock_session

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_multi_judge_key(self, mock_call):
        """Test that judge_key parameter routes to the right config."""
        mock_call.return_value = "GRADE: C"
        sample = ScorerInput(
            response="answer",
            target="42",
            metadata={"question": "Q?"},
            config={
                "judge": {"url": "http://judge-a:8000/v1", "model_id": "judge-a"},
                "judge_1": {"url": "http://judge-b:8000/v1", "model_id": "judge-b"},
            },
        )
        # Default key="judge"
        judge_score(sample, template="binary_qa")
        first_config = mock_call.call_args[0][0]
        assert first_config.model_id == "judge-a"

        mock_call.reset_mock()
        # Explicit key="judge_1"
        judge_score(sample, template="binary_qa", judge_key="judge_1")
        second_config = mock_call.call_args[0][0]
        assert second_config.model_id == "judge-b"

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_nemo_skills_config_format(self, mock_call):
        """Test with exact nemo-skills framework.yml config format."""
        mock_call.return_value = "GRADE: C"
        sample = ScorerInput(
            response="answer",
            target="42",
            metadata={"question": "Q?"},
            config={
                "judge_support": True,
                "judge": {
                    "model_id": "openai/gpt-4.1",
                    "url": "https://inference-api.nvidia.com/v1",
                    "api_key": "JUDGE_API_KEY",
                    "max_new_tokens": 4096,
                    "parallelism": 16,
                },
            },
        )
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "C"

        called_config = mock_call.call_args[0][0]
        assert called_config.model_id == "openai/gpt-4.1"
        assert called_config.max_new_tokens == 4096


# ---------------------------------------------------------------------------
# TestJudgeTemplates
# ---------------------------------------------------------------------------


class TestJudgeTemplates:
    """Tests for built-in judge templates."""

    def test_all_templates_have_required_placeholders(self):
        for name, template in TEMPLATES.items():
            assert "{response}" in template, \
                f"Template '{name}' missing placeholder '{{response}}'"
        for name, template in TEMPLATES.items():
            if name != "safety":
                assert "{reference}" in template, \
                    f"Template '{name}' missing placeholder '{{reference}}'"

    def test_all_templates_have_criteria_placeholder(self):
        for name, template in TEMPLATES.items():
            assert "{criteria}" in template, \
                f"Template '{name}' missing '{{criteria}}' placeholder"

    def test_all_templates_have_default_pattern(self):
        for name in TEMPLATES:
            assert name in DEFAULT_PATTERNS, \
                f"Template '{name}' missing default pattern"

    def test_all_templates_have_default_score_mapping(self):
        for name in TEMPLATES:
            assert name in DEFAULT_SCORE_MAPPINGS, \
                f"Template '{name}' missing default score mapping"

    def test_binary_qa_pattern_matches(self):
        import re
        pattern = DEFAULT_PATTERNS["binary_qa"]
        assert re.search(pattern, "GRADE: C")
        assert re.search(pattern, "GRADE: I")
        assert not re.search(pattern, "GRADE: X")

    def test_likert_5_pattern_matches(self):
        import re
        pattern = DEFAULT_PATTERNS["likert_5"]
        for i in range(1, 6):
            assert re.search(pattern, f"GRADE: {i}")
        assert not re.search(pattern, "GRADE: 0")
        assert not re.search(pattern, "GRADE: 6")

    def test_safety_pattern_matches(self):
        import re
        pattern = DEFAULT_PATTERNS["safety"]
        assert re.search(pattern, "GRADE: SAFE")
        assert re.search(pattern, "GRADE: UNSAFE")
        assert not re.search(pattern, "GRADE: MAYBE")

    def test_score_mappings_cover_all_grades(self):
        assert set(DEFAULT_SCORE_MAPPINGS["binary_qa"].keys()) == {"C", "I"}
        assert set(DEFAULT_SCORE_MAPPINGS["binary_qa_partial"].keys()) == {"C", "P", "I"}
        assert set(DEFAULT_SCORE_MAPPINGS["likert_5"].keys()) == {"1", "2", "3", "4", "5"}
        assert set(DEFAULT_SCORE_MAPPINGS["safety"].keys()) == {"SAFE", "UNSAFE"}

    def test_templates_renderable(self):
        for name, template in TEMPLATES.items():
            rendered = render_judge_prompt(
                template,
                question="What is 2+2?",
                response="4",
                reference="4",
                criteria="Be accurate",
            )
            assert len(rendered) > 0, f"Template '{name}' rendered empty"


# ---------------------------------------------------------------------------
# TestJudgeIntegration — compile benchmark with extra.judge
# ---------------------------------------------------------------------------


class TestJudgeIntegration:
    """Integration tests: benchmark with judge config through compile pipeline."""

    def test_benchmark_with_judge_extra_config(self):
        """Verify judge={...} flows through @benchmark **kwargs."""
        from nemo_evaluator.contrib.byob.decorators import (
            benchmark,
            scorer,
            get_registered_benchmarks,
        )

        @benchmark(
            name="judge-test",
            dataset="test.jsonl",
            prompt="Q: {q}",
            judge={"url": "http://judge:8000/v1", "model_id": "nemotron"},
        )
        @scorer
        def my_scorer(sample):
            return {"score": 1.0}

        benchmarks = get_registered_benchmarks()
        bench = benchmarks["judge_test"]
        assert "judge" in bench.extra_config
        assert bench.extra_config["judge"]["url"] == "http://judge:8000/v1"

    def test_judge_in_scorer_input(self):
        """Verify judge config reaches ScorerInput.config via StandardStrategy."""
        from nemo_evaluator.contrib.byob.decorators import benchmark, scorer
        from nemo_evaluator.contrib.byob.eval_logic import StandardStrategy

        received_configs = []

        @benchmark(
            name="judge-input-test",
            dataset="test.jsonl",
            prompt="Q: {question}",
            judge={"url": "http://judge:8000/v1", "model_id": "nemotron"},
        )
        @scorer
        def capture_scorer(sample):
            received_configs.append(sample.config)
            return {"score": 1.0}

        from nemo_evaluator.contrib.byob.decorators import get_registered_benchmarks
        bench = get_registered_benchmarks()["judge_input_test"]

        strategy = StandardStrategy()
        mock_model = MagicMock(return_value="response")
        strategy.evaluate_sample(
            idx=0,
            row={"question": "test"},
            bench=bench,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert len(received_configs) == 1
        assert "judge" in received_configs[0]
        assert received_configs[0]["judge"]["model_id"] == "nemotron"

    def test_judge_support_flag_in_fdf(self):
        """Verify compiler sets judge_support: true in FDF when judge is present."""
        from nemo_evaluator.contrib.byob.compiler import compile_benchmark
        import tempfile, os

        code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer
@benchmark(
    name="fdf-judge",
    dataset="test.jsonl",
    prompt="Q: {q}",
    judge={"url": "http://judge:8000/v1", "model_id": "nemotron"},
)
@scorer
def s(sample):
    return {"score": 1.0}
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            # Create dummy dataset
            ds_path = os.path.join(os.path.dirname(f.name), "test.jsonl")
            with open(ds_path, "w") as ds:
                ds.write('{"q": "hello"}\n')
            try:
                compiled = compile_benchmark(f.name)
                fdf = compiled["fdf_judge"]
                extra = fdf["defaults"]["config"]["params"]["extra"]
                assert extra.get("judge_support") is True
                assert "judge" in extra
                assert extra["judge"]["model_id"] == "nemotron"
            finally:
                os.unlink(f.name)
                os.unlink(ds_path)

    def test_multi_judge_in_fdf(self):
        """Verify multiple judge configs flow to FDF."""
        from nemo_evaluator.contrib.byob.compiler import compile_benchmark
        import tempfile, os

        code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer
@benchmark(
    name="multi-judge",
    dataset="test.jsonl",
    prompt="Q: {q}",
    judge={"url": "http://judge-a:8000/v1", "model_id": "judge-a"},
    judge_1={"url": "http://judge-b:8000/v1", "model_id": "judge-b"},
)
@scorer
def s(sample):
    return {"score": 1.0}
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            ds_path = os.path.join(os.path.dirname(f.name), "test.jsonl")
            with open(ds_path, "w") as ds:
                ds.write('{"q": "hello"}\n')
            try:
                compiled = compile_benchmark(f.name)
                fdf = compiled["multi_judge"]
                extra = fdf["defaults"]["config"]["params"]["extra"]
                assert extra.get("judge_support") is True
                assert "judge" in extra
                assert "judge_1" in extra
                assert extra["judge"]["model_id"] == "judge-a"
                assert extra["judge_1"]["model_id"] == "judge-b"
            finally:
                os.unlink(f.name)
                os.unlink(ds_path)


# ---------------------------------------------------------------------------
# TestStructuredJudgeOutput — response_format and structured parsing
# ---------------------------------------------------------------------------


class TestStructuredJudgeOutput:
    """Tests for structured judge output (response_format + structured parse)."""

    def test_response_format_in_payload_when_set(self):
        """Test that response_format is included in HTTP payload when provided."""
        config = JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"grade": "C"}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        response_format = {"type": "json_object"}
        judge_call(config, "Test prompt", session=mock_session, response_format=response_format)

        payload = mock_session.post.call_args[1]["json"]
        assert "response_format" in payload
        assert payload["response_format"] == {"type": "json_object"}

    def test_response_format_absent_when_none(self):
        """Test that response_format is NOT in HTTP payload when None."""
        config = JudgeConfig(url="http://judge:8000/v1", model_id="nemotron")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "GRADE: C"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        judge_call(config, "Test prompt", session=mock_session)

        payload = mock_session.post.call_args[1]["json"]
        assert "response_format" not in payload

    def test_structured_parse_extracts_grade_from_json(self):
        response = '{"grade": "C", "reasoning": "correct answer"}'
        result = parse_grade(response, r"GRADE:\s*([CI])", structured=True)
        assert result == "C"

    def test_structured_parse_extracts_score_key(self):
        response = '{"score": 4, "explanation": "good"}'
        result = parse_grade(response, r"NONEXISTENT", structured=True)
        assert result == "4"

    def test_structured_parse_extracts_verdict_key(self):
        response = '{"verdict": "SAFE"}'
        result = parse_grade(response, r"NONEXISTENT", structured=True)
        assert result == "SAFE"

    def test_structured_parse_falls_back_to_regex(self):
        response = "Not valid JSON. GRADE: C"
        result = parse_grade(response, r"GRADE:\s*([CI])", structured=True)
        assert result == "C"

    def test_unstructured_parse_unchanged(self):
        response = "The answer is correct.\nGRADE: C"
        result = parse_grade(response, r"GRADE:\s*([CI])")
        assert result == "C"

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_judge_score_with_response_format(self, mock_call):
        mock_call.return_value = '{"grade": "C"}'
        sample = ScorerInput(
            response="The answer is 42",
            target="42",
            metadata={"question": "What is the answer?"},
            config={"judge": JUDGE_DICT},
        )
        judge_score(sample, template="binary_qa", response_format={"type": "json_object"})
        assert mock_call.call_args[1].get("response_format") == {"type": "json_object"}

    @patch("nemo_evaluator.contrib.byob.judge.judge_call")
    def test_judge_score_without_response_format(self, mock_call):
        mock_call.return_value = "The answer matches.\nGRADE: C"
        sample = ScorerInput(
            response="The answer is 42",
            target="42",
            metadata={"question": "What is the answer?"},
            config={"judge": JUDGE_DICT},
        )
        result = judge_score(sample, template="binary_qa")
        assert result["judge_grade"] == "C"
        assert result["judge_score"] == 1.0
        assert mock_call.call_args[1].get("response_format") is None
