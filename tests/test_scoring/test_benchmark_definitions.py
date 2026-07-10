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
"""Golden tests: verify benchmark scorers produce correct results on known data."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from nemo_evaluator.scoring import (
    ScorerInput,
    answer_line,
    exact_match,
    fuzzy_match,
    multichoice_regex,
    numeric_match,
)


class TestMultichoiceRegex:
    def test_correct_answer(self):
        s = ScorerInput(response="I think the answer is B.\n\nAnswer: B", target="B")
        assert multichoice_regex(s)["correct"] is True

    def test_wrong_answer(self):
        s = ScorerInput(response="Answer: C", target="A")
        assert multichoice_regex(s)["correct"] is False

    def test_case_insensitive(self):
        s = ScorerInput(response="answer: d", target="D")
        assert multichoice_regex(s)["correct"] is True

    def test_no_answer_found(self):
        s = ScorerInput(response="I'm not sure about this one", target="A")
        assert multichoice_regex(s)["correct"] is False

    def test_10_choice(self):
        s = ScorerInput(response="Answer: H", target="H")
        assert multichoice_regex(s, pattern=r"(?i)Answer\s*:\s*([A-J])")["correct"] is True


class TestAnswerLine:
    def test_correct_math_answer(self):
        s = ScorerInput(response="Let me solve...\n\nAnswer: 42", target="42")
        assert answer_line(s)["correct"] is True

    def test_whitespace_tolerance(self):
        s = ScorerInput(response="Answer:  42 ", target="42")
        assert answer_line(s)["correct"] is True

    def test_wrong_answer(self):
        s = ScorerInput(response="Answer: 43", target="42")
        assert answer_line(s)["correct"] is False

    def test_extracts_from_last_line_if_no_pattern(self):
        s = ScorerInput(response="The result is\n42", target="42")
        result = answer_line(s)
        assert result["extracted"] == "42"


class TestNumericMatch:
    def test_integer(self):
        s = ScorerInput(response="The answer is 42", target="42")
        assert numeric_match(s)["correct"] is True

    def test_strips_trailing_zeros(self):
        s = ScorerInput(response="Answer: 5.00", target="5")
        assert numeric_match(s)["correct"] is True

    def test_commas_in_number(self):
        s = ScorerInput(response="1,234 people", target="1234")
        assert numeric_match(s)["correct"] is True

    def test_picks_last_number(self):
        s = ScorerInput(response="Step 1: 10, Step 2: 20, Final: 30", target="30")
        assert numeric_match(s)["correct"] is True


class TestFuzzyMatch:
    def test_substring_match(self):
        s = ScorerInput(
            response="The Battle of Gettysburg", target="Gettysburg", metadata={"correct_answers": ["Gettysburg"]}
        )
        assert fuzzy_match(s)["correct"] is True

    def test_no_match(self):
        s = ScorerInput(response="Paris", target="London", metadata={"correct_answers": ["London"]})
        assert fuzzy_match(s)["correct"] is False


class TestExactMatch:
    def test_exact(self):
        s = ScorerInput(response="  Paris  ", target="paris")
        assert exact_match(s)["correct"] is True

    def test_article_removal(self):
        s = ScorerInput(response="The Eiffel Tower", target="Eiffel Tower")
        assert exact_match(s)["correct"] is True


class TestBenchmarkScorerImport:
    """Verify each built-in benchmark's scorer is importable and callable."""

    def test_mmlu_scorer(self):
        from nemo_evaluator.benchmarks.mmlu import mmlu_scorer

        s = ScorerInput(response="Answer: C", target="C")
        assert mmlu_scorer(s)["correct"] is True

    def test_mmlu_pro_scorer(self):
        from nemo_evaluator.benchmarks.mmlu_pro import mmlu_pro_scorer

        s = ScorerInput(response="Answer: F", target="F")
        assert mmlu_pro_scorer(s)["correct"] is True

    def test_math500_scorer(self):
        from nemo_evaluator.benchmarks.math500 import math500_scorer

        s = ScorerInput(response="Answer: 42", target="42")
        assert math500_scorer(s)["correct"] is True

    def test_gpqa_scorer(self):
        from nemo_evaluator.benchmarks.gpqa import gpqa_scorer

        s = ScorerInput(response="Answer: A", target="A")
        assert gpqa_scorer(s)["correct"] is True

    def test_gsm8k_scorer(self):
        from nemo_evaluator.benchmarks.gsm8k import gsm8k_scorer

        s = ScorerInput(response="The answer is 42", target="42")
        assert gsm8k_scorer(s)["correct"] is True

    def test_mgsm_scorer(self):
        from nemo_evaluator.benchmarks.mgsm import mgsm_scorer

        s = ScorerInput(response="The result is 15", target="15")
        assert mgsm_scorer(s)["correct"] is True


def test_healthbench_fallback_uses_available_evaluation_file(monkeypatch):
    benchmark_package = ModuleType("nemo_evaluator.benchmarks")
    benchmark_package.__path__ = [str(Path(__file__).parents[2] / "src" / "nemo_evaluator" / "benchmarks")]
    monkeypatch.setitem(sys.modules, "nemo_evaluator.benchmarks", benchmark_package)
    monkeypatch.delitem(sys.modules, "nemo_evaluator.benchmarks.healthbench", raising=False)
    healthbench = importlib.import_module("nemo_evaluator.benchmarks.healthbench")

    def fail_dataset_load(*args, **kwargs):
        raise FileNotFoundError("test split unavailable")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"prompt": [{"role": "user", "content": "Question"}]}\n'

    requested = {}

    def fake_urlopen(url, timeout):
        requested.update(url=url, timeout=timeout)
        return Response()

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fail_dataset_load))
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    rows = healthbench._load_healthbench()

    assert requested == {
        "url": ("https://huggingface.co/datasets/openai/HealthBench/resolve/main/2025-05-07-06-14-12_oss_eval.jsonl"),
        "timeout": 60,
    }
    assert rows == [{"prompt": [{"role": "user", "content": "Question"}]}]
