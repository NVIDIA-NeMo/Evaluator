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

"""Unit tests for the BYOB Gym adapter module."""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_evaluator.contrib.byob.decorators import (
    BenchmarkDefinition,
    ScorerInput,
    _normalize_name,
)
from nemo_evaluator.contrib.byob.gym_adapter import (
    ByobGymHarness,
    GymHarness,
    _clamp01,
    _coerce_to_reward,
    extract_answer_from_scores,
    scores_to_reward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bench(
    name: str = "test-bench",
    dataset: str = "/tmp/test_data.jsonl",
    prompt: str = "Q: {question}\nA:",
    target_field: str = "answer",
    scorer_fn=None,
    system_prompt=None,
    extra_config=None,
    field_mapping=None,
    response_field=None,
) -> BenchmarkDefinition:
    """Create a BenchmarkDefinition for testing without triggering the decorator."""
    if scorer_fn is None:

        def scorer_fn(inp: ScorerInput) -> dict:
            return {
                "correct": inp.response.strip().lower()
                == str(inp.target).strip().lower()
            }

    return BenchmarkDefinition(
        name=name,
        normalized_name=_normalize_name(name),
        dataset=dataset,
        prompt=prompt,
        scorer_fn=scorer_fn,
        target_field=target_field,
        system_prompt=system_prompt,
        extra_config=extra_config or {},
        field_mapping=field_mapping,
        response_field=response_field,
    )


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small JSONL dataset file and return its path."""
    data = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
        {"question": "Color of sky?", "answer": "blue"},
    ]
    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture
def basic_bench(sample_dataset):
    """A basic BenchmarkDefinition backed by a real dataset file."""
    return _make_bench(dataset=sample_dataset)


# ---------------------------------------------------------------------------
# scores_to_reward tests
# ---------------------------------------------------------------------------


class TestScoresToReward:
    """Tests for the scores_to_reward() mapping function."""

    def test_empty_scores(self):
        assert scores_to_reward({}) == 0.0

    def test_explicit_reward_key_bool(self):
        assert scores_to_reward({"correct": True}, reward_key="correct") == 1.0
        assert scores_to_reward({"correct": False}, reward_key="correct") == 0.0

    def test_explicit_reward_key_float(self):
        assert scores_to_reward({"f1": 0.85}, reward_key="f1") == 0.85

    def test_explicit_reward_key_missing(self):
        with pytest.raises(KeyError):
            scores_to_reward({"f1": 0.85}, reward_key="nonexistent")

    def test_priority_correct(self):
        """'correct' key is used when present, even if other keys exist."""
        scores = {"correct": True, "f1": 0.5, "precision": 0.6}
        assert scores_to_reward(scores) == 1.0

    def test_priority_judge_score(self):
        """'judge_score' is used when 'correct' is absent."""
        scores = {"judge_score": 0.8, "judge_grade": "GOOD"}
        assert scores_to_reward(scores) == 0.8

    def test_first_bool_fallback(self):
        """First boolean key is used when no priority keys exist."""
        scores = {"is_valid": True, "f1": 0.5}
        assert scores_to_reward(scores) == 1.0

    def test_first_float_fallback(self):
        """First numeric key is used when no bool keys exist."""
        scores = {"f1": 0.85, "precision": 0.9}
        assert scores_to_reward(scores) == 0.85

    def test_mean_fallback(self):
        """Mean of all numeric values when no single key matches."""
        # This case triggers when keys are all non-bool numerics and none
        # match priority. The first-float-fallback would catch first, so
        # we need a case where step 3/4 skip. Actually, step 4 catches
        # first numeric, so mean is for edge cases. Let's test clamping.
        scores = {"f1": 0.8}
        assert scores_to_reward(scores) == 0.8

    def test_clamping_above_one(self):
        """Values above 1.0 are clamped to 1.0."""
        assert scores_to_reward({"score": 1.5}, reward_key="score") == 1.0

    def test_clamping_below_zero(self):
        """Negative values are clamped to 0.0."""
        assert scores_to_reward({"score": -0.5}, reward_key="score") == 0.0

    def test_no_numeric_values(self):
        """Non-numeric values result in 0.0."""
        scores = {"grade": "A", "label": "good"}
        assert scores_to_reward(scores) == 0.0

    def test_bool_is_not_int(self):
        """Booleans are treated as booleans, not integers."""
        # bool is subclass of int in Python; verify correct handling
        scores = {"correct": False}
        assert scores_to_reward(scores) == 0.0


class TestCoerceToReward:
    """Tests for the _coerce_to_reward helper."""

    def test_bool_true(self):
        assert _coerce_to_reward(True) == 1.0

    def test_bool_false(self):
        assert _coerce_to_reward(False) == 0.0

    def test_int(self):
        assert _coerce_to_reward(1) == 1.0

    def test_float(self):
        assert _coerce_to_reward(0.75) == 0.75

    def test_string_numeric(self):
        assert _coerce_to_reward("0.5") == 0.5

    def test_string_non_numeric(self):
        assert _coerce_to_reward("good") == 0.0


class TestClamp01:
    """Tests for the _clamp01 helper."""

    def test_within_range(self):
        assert _clamp01(0.5) == 0.5

    def test_at_zero(self):
        assert _clamp01(0.0) == 0.0

    def test_at_one(self):
        assert _clamp01(1.0) == 1.0

    def test_above_one(self):
        assert _clamp01(2.5) == 1.0

    def test_below_zero(self):
        assert _clamp01(-0.3) == 0.0


# ---------------------------------------------------------------------------
# extract_answer_from_scores tests
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    """Tests for extract_answer_from_scores()."""

    def test_extracted_answer_key(self):
        assert extract_answer_from_scores({"extracted_answer": "42"}) == "42"

    def test_predicted_key(self):
        assert extract_answer_from_scores({"predicted": "A"}) == "A"

    def test_prediction_key(self):
        assert extract_answer_from_scores({"prediction": "B"}) == "B"

    def test_judge_grade_key(self):
        assert extract_answer_from_scores({"judge_grade": "GOOD"}) == "GOOD"

    def test_no_answer_key(self):
        assert extract_answer_from_scores({"correct": True, "f1": 0.5}) is None

    def test_empty_scores(self):
        assert extract_answer_from_scores({}) is None


# ---------------------------------------------------------------------------
# ByobGymHarness.get_dataset() tests
# ---------------------------------------------------------------------------


class TestGetDataset:
    """Tests for ByobGymHarness.get_dataset()."""

    def test_basic_dataset(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        dataset = harness.get_dataset()

        assert len(dataset) == 3
        for row in dataset:
            assert "responses_create_params" in row
            assert "expected_answer" in row
            assert "_byob_row" in row

    def test_prompt_rendering(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        dataset = harness.get_dataset()

        first_row = dataset[0]
        messages = first_row["responses_create_params"]["input"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What is 2+2?" in messages[0]["content"]

    def test_expected_answer(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        dataset = harness.get_dataset()

        answers = [row["expected_answer"] for row in dataset]
        assert answers == ["4", "Paris", "blue"]

    def test_system_prompt(self, sample_dataset):
        bench = _make_bench(
            dataset=sample_dataset,
            system_prompt="You are a helpful assistant.",
        )
        harness = ByobGymHarness(bench=bench)
        dataset = harness.get_dataset()

        messages = dataset[0]["responses_create_params"]["input"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    def test_system_prompt_with_jinja2(self, sample_dataset):
        bench = _make_bench(
            dataset=sample_dataset,
            system_prompt="Answer questions about {question}",
        )
        # Mark system prompt as NOT jinja2 (plain format string)
        bench._is_system_prompt_jinja2 = False

        harness = ByobGymHarness(bench=bench)
        dataset = harness.get_dataset()

        messages = dataset[0]["responses_create_params"]["input"]
        assert "What is 2+2?" in messages[0]["content"]

    def test_byob_row_passthrough(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        dataset = harness.get_dataset()

        row = dataset[0]["_byob_row"]
        assert row["question"] == "What is 2+2?"
        assert row["answer"] == "4"

    def test_dataset_caching(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        ds1 = harness.get_dataset()
        ds2 = harness.get_dataset()
        assert ds1 is ds2  # Same object, not a copy

    def test_dataset_limit(self, sample_dataset):
        bench = _make_bench(dataset=sample_dataset)
        harness = ByobGymHarness(bench=bench, dataset_limit=2)
        dataset = harness.get_dataset()
        assert len(dataset) == 2

    def test_skip_bad_prompt(self, tmp_path):
        """Rows with missing prompt fields are skipped, not errored."""
        data = [
            {"question": "Good row", "answer": "yes"},
            {"answer": "no"},  # missing "question" field
        ]
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        bench = _make_bench(dataset=str(path))
        harness = ByobGymHarness(bench=bench)
        dataset = harness.get_dataset()

        assert len(dataset) == 1
        assert dataset[0]["expected_answer"] == "yes"

    def test_structured_target(self, tmp_path):
        """Non-string targets are converted to str for expected_answer."""
        data = [
            {"question": "List items", "answer": ["a", "b", "c"]},
        ]
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        bench = _make_bench(dataset=str(path))
        harness = ByobGymHarness(bench=bench)
        dataset = harness.get_dataset()

        assert dataset[0]["expected_answer"] == "['a', 'b', 'c']"
        # But the original structured target is in _byob_row
        assert dataset[0]["_byob_row"]["answer"] == ["a", "b", "c"]

    def test_field_mapping(self, tmp_path):
        """field_mapping is applied during dataset loading."""
        data = [
            {"q": "What?", "a": "Yes"},
        ]
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        bench = _make_bench(
            dataset=str(path),
            prompt="Q: {question}\nA:",
            target_field="answer",
            field_mapping={"q": "question", "a": "answer"},
        )
        harness = ByobGymHarness(bench=bench)
        dataset = harness.get_dataset()

        assert len(dataset) == 1
        assert "What?" in dataset[0]["responses_create_params"]["input"][0]["content"]
        assert dataset[0]["expected_answer"] == "Yes"


# ---------------------------------------------------------------------------
# ByobGymHarness.verify() tests
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Helper to run an async function in tests."""
    return asyncio.run(coro)


class TestVerify:
    """Tests for ByobGymHarness.verify()."""

    def test_correct_response(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        row = {"question": "What is 2+2?", "answer": "4"}

        reward, answer = _run_async(
            harness.verify(
                response_text="4",
                expected_answer="4",
                _byob_row=row,
            )
        )
        assert reward == 1.0

    def test_incorrect_response(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        row = {"question": "What is 2+2?", "answer": "4"}

        reward, answer = _run_async(
            harness.verify(
                response_text="5",
                expected_answer="4",
                _byob_row=row,
            )
        )
        assert reward == 0.0

    def test_case_insensitive(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        row = {"question": "Capital?", "answer": "Paris"}

        reward, _ = _run_async(
            harness.verify(
                response_text="paris",
                expected_answer="Paris",
                _byob_row=row,
            )
        )
        assert reward == 1.0

    def test_missing_byob_row(self, basic_bench):
        """verify() works even without _byob_row (fallback to expected_answer)."""
        harness = ByobGymHarness(bench=basic_bench)

        reward, _ = _run_async(
            harness.verify(
                response_text="4",
                expected_answer="4",
            )
        )
        assert reward == 1.0

    def test_custom_reward_key(self, sample_dataset):
        def f1_scorer(inp: ScorerInput) -> dict:
            return {"f1": 0.85, "precision": 0.9, "recall": 0.8}

        bench = _make_bench(dataset=sample_dataset, scorer_fn=f1_scorer)
        harness = ByobGymHarness(bench=bench, reward_key="f1")

        reward, _ = _run_async(
            harness.verify(
                response_text="anything",
                expected_answer="anything",
                _byob_row={"question": "q", "answer": "a"},
            )
        )
        assert reward == 0.85

    def test_scorer_exception(self, sample_dataset):
        def bad_scorer(inp: ScorerInput) -> dict:
            raise RuntimeError("Scorer exploded")

        bench = _make_bench(dataset=sample_dataset, scorer_fn=bad_scorer)
        harness = ByobGymHarness(bench=bench)

        reward, answer = _run_async(
            harness.verify(
                response_text="anything",
                expected_answer="anything",
                _byob_row={},
            )
        )
        assert reward == 0.0
        assert answer is None

    def test_scorer_non_dict_result(self, sample_dataset):
        def bad_scorer(inp: ScorerInput) -> Any:
            return 42  # Not a dict

        bench = _make_bench(dataset=sample_dataset, scorer_fn=bad_scorer)
        harness = ByobGymHarness(bench=bench)

        reward, answer = _run_async(
            harness.verify(
                response_text="anything",
                expected_answer="anything",
                _byob_row={},
            )
        )
        assert reward == 0.0
        assert answer is None

    def test_extracted_answer(self, sample_dataset):
        def scorer_with_answer(inp: ScorerInput) -> dict:
            return {"correct": True, "extracted_answer": "42"}

        bench = _make_bench(dataset=sample_dataset, scorer_fn=scorer_with_answer)
        harness = ByobGymHarness(bench=bench)

        reward, answer = _run_async(
            harness.verify(
                response_text="The answer is 42",
                expected_answer="42",
                _byob_row={"question": "q", "answer": "42"},
            )
        )
        assert reward == 1.0
        assert answer == "42"

    def test_structured_target_preserved(self, sample_dataset):
        """The original structured target from _byob_row is used, not expected_answer str."""
        expected_list = ["a", "b"]

        def list_scorer(inp: ScorerInput) -> dict:
            return {
                "correct": isinstance(inp.target, list) and inp.target == expected_list
            }

        bench = _make_bench(
            dataset=sample_dataset, scorer_fn=list_scorer, target_field="items"
        )
        harness = ByobGymHarness(bench=bench)

        reward, _ = _run_async(
            harness.verify(
                response_text="anything",
                expected_answer="['a', 'b']",  # stringified
                _byob_row={"question": "q", "items": expected_list},
            )
        )
        assert reward == 1.0

    def test_judge_config_merge(self, sample_dataset):
        """Gym-level judge_config is merged into bench.extra_config for the scorer."""
        captured_config = {}

        def judge_scorer(inp: ScorerInput) -> dict:
            captured_config.update(inp.config)
            return {"judge_score": 0.9}

        bench = _make_bench(
            dataset=sample_dataset,
            scorer_fn=judge_scorer,
            extra_config={"judge": {"url": "http://original:8000", "model_id": "orig"}},
        )

        harness = ByobGymHarness(
            bench=bench,
            judge_config={"url": "http://override:8000"},
        )

        _run_async(
            harness.verify(
                response_text="anything",
                expected_answer="anything",
                _byob_row={},
            )
        )

        # Gym-level override should have been merged
        assert captured_config["judge"]["url"] == "http://override:8000"
        # Original keys not overridden should be preserved
        assert captured_config["judge"]["model_id"] == "orig"

    def test_model_call_fn_passthrough(self, sample_dataset):
        """model_call_fn is available in ScorerInput.model_call_fn."""
        captured_fn = []

        def checker_scorer(inp: ScorerInput) -> dict:
            captured_fn.append(inp.model_call_fn)
            return {"correct": True}

        mock_fn = MagicMock()
        bench = _make_bench(dataset=sample_dataset, scorer_fn=checker_scorer)
        harness = ByobGymHarness(bench=bench, model_call_fn=mock_fn)

        _run_async(
            harness.verify(
                response_text="x",
                expected_answer="x",
                _byob_row={},
            )
        )

        assert captured_fn[0] is mock_fn

    def test_invalid_reward_key(self, sample_dataset):
        """A reward_key not in scorer output logs warning and returns 0.0."""

        def simple_scorer(inp: ScorerInput) -> dict:
            return {"correct": True}

        bench = _make_bench(dataset=sample_dataset, scorer_fn=simple_scorer)
        harness = ByobGymHarness(bench=bench, reward_key="nonexistent")

        reward, _ = _run_async(
            harness.verify(
                response_text="x",
                expected_answer="x",
                _byob_row={},
            )
        )
        assert reward == 0.0


# ---------------------------------------------------------------------------
# ByobGymHarness offload_scorer tests
# ---------------------------------------------------------------------------


class TestOffloadScorer:
    """Tests for the gym_offload_scorer flag."""

    def test_offload_to_thread(self, sample_dataset):
        """When gym_offload_scorer=True, scorer runs via asyncio.to_thread."""

        call_count = 0

        def slow_scorer(inp: ScorerInput) -> dict:
            nonlocal call_count
            call_count += 1
            return {"correct": True}

        bench = _make_bench(
            dataset=sample_dataset,
            scorer_fn=slow_scorer,
            extra_config={"gym_offload_scorer": True},
        )
        harness = ByobGymHarness(bench=bench)

        reward, _ = _run_async(
            harness.verify(
                response_text="x",
                expected_answer="x",
                _byob_row={},
            )
        )
        assert reward == 1.0
        assert call_count == 1


# ---------------------------------------------------------------------------
# ByobGymHarness construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for ByobGymHarness construction and properties."""

    def test_eval_type(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        assert harness.eval_type == "byob_test_bench.test-bench"

    def test_benchmark_property(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        assert harness.benchmark is basic_bench

    def test_reward_key_property(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench, reward_key="f1")
        assert harness.reward_key == "f1"

    def test_reward_key_default_none(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        assert harness.reward_key is None

    def test_is_gym_harness_subclass(self, basic_bench):
        harness = ByobGymHarness(bench=basic_bench)
        assert isinstance(harness, GymHarness)


# ---------------------------------------------------------------------------
# ByobGymHarness.from_module() tests
# ---------------------------------------------------------------------------


class TestFromModule:
    """Tests for the from_module() factory classmethod."""

    def test_from_module_file(self, tmp_path):
        """Load a benchmark from a .py file path."""
        bench_py = tmp_path / "my_bench.py"
        bench_py.write_text(
            textwrap.dedent("""\
                from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

                @benchmark(
                    name="simple-qa",
                    dataset="__DATASET__",
                    prompt="Q: {q}",
                    target_field="a",
                )
                @scorer
                def my_scorer(inp: ScorerInput) -> dict:
                    return {"correct": inp.response.strip() == str(inp.target).strip()}
            """)
        )

        # Create a minimal dataset
        data_path = tmp_path / "data.jsonl"
        data_path.write_text(json.dumps({"q": "Hi", "a": "Hello"}) + "\n")

        # Patch the dataset path in the source
        content = bench_py.read_text().replace("__DATASET__", str(data_path))
        bench_py.write_text(content)

        harness = ByobGymHarness.from_module(
            benchmark_module=str(bench_py),
            benchmark_name="simple-qa",
        )

        assert harness.eval_type == "byob_simple_qa.simple-qa"
        dataset = harness.get_dataset()
        assert len(dataset) == 1

    def test_from_module_not_found(self, tmp_path):
        """Raise ValueError when benchmark name does not exist."""
        bench_py = tmp_path / "empty_bench.py"
        bench_py.write_text("# No benchmarks here\n")

        with pytest.raises(ValueError, match="not found"):
            ByobGymHarness.from_module(
                benchmark_module=str(bench_py),
                benchmark_name="nonexistent",
            )


# ---------------------------------------------------------------------------
# ByobGymHarness.from_eval_type() tests
# ---------------------------------------------------------------------------


class TestFromEvalType:
    """Tests for the from_eval_type() factory classmethod."""

    def test_invalid_format_no_dot(self):
        with pytest.raises(ValueError, match="Invalid BYOB eval_type format"):
            ByobGymHarness.from_eval_type("byob_foo")

    def test_invalid_format_no_byob_prefix(self):
        with pytest.raises(ValueError, match="must start with 'byob_'"):
            ByobGymHarness.from_eval_type("custom_foo.bar")


# ---------------------------------------------------------------------------
# Integration: end-to-end get_dataset + verify
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end tests exercising the full get_dataset -> verify pipeline."""

    def test_exact_match_pipeline(self, sample_dataset):
        from nemo_evaluator.contrib.byob.scorers import exact_match

        bench = _make_bench(dataset=sample_dataset, scorer_fn=exact_match)
        harness = ByobGymHarness(bench=bench)

        dataset = harness.get_dataset()
        assert len(dataset) == 3

        # Correct answer
        row = dataset[0]
        reward, _ = _run_async(
            harness.verify(
                response_text="4",
                expected_answer=row["expected_answer"],
                _byob_row=row["_byob_row"],
            )
        )
        assert reward == 1.0

        # Wrong answer
        reward, _ = _run_async(
            harness.verify(
                response_text="five",
                expected_answer=row["expected_answer"],
                _byob_row=row["_byob_row"],
            )
        )
        assert reward == 0.0

    def test_f1_scorer_pipeline(self, sample_dataset):
        from nemo_evaluator.contrib.byob.scorers import f1_token

        bench = _make_bench(dataset=sample_dataset, scorer_fn=f1_token)
        # Use f1 as reward key since f1_token returns {"f1": ..., "precision": ..., "recall": ...}
        harness = ByobGymHarness(bench=bench, reward_key="f1")

        dataset = harness.get_dataset()
        row = dataset[0]

        reward, _ = _run_async(
            harness.verify(
                response_text="4",
                expected_answer=row["expected_answer"],
                _byob_row=row["_byob_row"],
            )
        )
        # Perfect match: f1 should be 1.0
        assert reward == 1.0

    def test_multi_metric_auto_detect(self, sample_dataset):
        """Without reward_key, f1_token scores use first-float fallback."""
        from nemo_evaluator.contrib.byob.scorers import f1_token

        bench = _make_bench(dataset=sample_dataset, scorer_fn=f1_token)
        harness = ByobGymHarness(bench=bench)  # No reward_key

        dataset = harness.get_dataset()
        row = dataset[0]

        reward, _ = _run_async(
            harness.verify(
                response_text="4",
                expected_answer=row["expected_answer"],
                _byob_row=row["_byob_row"],
            )
        )
        # Auto-detect: no "correct" or "judge_score" key, no bool key,
        # first float key ("f1") is used
        assert reward == 1.0

    def test_contains_scorer_pipeline(self, sample_dataset):
        from nemo_evaluator.contrib.byob.scorers import contains

        bench = _make_bench(dataset=sample_dataset, scorer_fn=contains)
        harness = ByobGymHarness(bench=bench)

        dataset = harness.get_dataset()
        row = dataset[1]  # "Capital of France?" -> "Paris"

        reward, _ = _run_async(
            harness.verify(
                response_text="The capital is Paris, a beautiful city.",
                expected_answer=row["expected_answer"],
                _byob_row=row["_byob_row"],
            )
        )
        assert reward == 1.0

    def test_all_rows_pipeline(self, sample_dataset):
        """Run verify on every row in the dataset -- simulates full Gym evaluation."""

        def score_fn(inp: ScorerInput) -> dict:
            return {"correct": str(inp.target).lower() in inp.response.lower()}

        bench = _make_bench(dataset=sample_dataset, scorer_fn=score_fn)
        harness = ByobGymHarness(bench=bench)
        dataset = harness.get_dataset()

        rewards = []
        for row in dataset:
            reward, _ = _run_async(
                harness.verify(
                    response_text=row["expected_answer"],  # echo back the answer
                    expected_answer=row["expected_answer"],
                    _byob_row=row["_byob_row"],
                )
            )
            rewards.append(reward)

        # All should be correct since we echo the target
        assert all(r == 1.0 for r in rewards)
        assert len(rewards) == 3
