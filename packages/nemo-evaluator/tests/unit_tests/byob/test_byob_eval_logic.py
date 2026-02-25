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

"""Unit tests for BYOB eval_logic shared between subprocess and native modes."""

import pytest
from unittest.mock import MagicMock

from nemo_evaluator.contrib.byob.decorators import (
    BenchmarkDefinition,
    ScorerInput,
    get_registered_benchmarks,
)
from nemo_evaluator.contrib.byob.eval_logic import (
    EvalOnlyStrategy,
    SampleResult,
    StandardStrategy,
    import_benchmark,
    render_prompt,
    run_eval_loop,
)


class TestImportBenchmark:
    """Tests for import_benchmark function."""

    @pytest.fixture
    def temp_benchmark_file(self, tmp_path):
        """Create a temporary benchmark .py file."""
        code = """
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="test-import", dataset="unused", prompt="Q: {q}\\nA:", target_field="a")
@scorer
def test_scorer(sample):
    return {"match": sample.target in sample.response}
"""
        benchmark_file = tmp_path / "import_test.py"
        benchmark_file.write_text(code)
        return benchmark_file

    def test_import_benchmark_by_filepath(self, temp_benchmark_file):
        """Test importing benchmark from file path.

        Validates:
        - File path is resolved correctly
        - Benchmark is registered during import
        - Correct benchmark is returned
        """
        bench = import_benchmark(str(temp_benchmark_file), "test_import")

        assert bench is not None, "Expected benchmark to be returned"
        assert bench.name == "test-import", (
            f"Expected name='test-import', got '{bench.name}'"
        )
        assert bench.prompt == "Q: {q}\nA:", (
            f"Expected specific prompt, got '{bench.prompt}'"
        )
        assert bench.target_field == "a", (
            f"Expected target_field='a', got '{bench.target_field}'"
        )

    def test_import_benchmark_clears_registry_first(self, temp_benchmark_file):
        """Test that import_benchmark clears registry before importing.

        This prevents cross-evaluation pollution. Registry should be
        empty at start, populated after first import, then cleared
        and repopulated on second import.
        """
        # First import
        import_benchmark(str(temp_benchmark_file), "test_import")
        benchmarks_after_first = get_registered_benchmarks()
        assert "test_import" in benchmarks_after_first, (
            "Benchmark should be registered after import"
        )

        # Manually register a different benchmark to simulate pollution
        from nemo_evaluator.contrib.byob import benchmark, scorer

        @benchmark(name="polluted-bench", dataset="x", prompt="x", target_field="x")
        @scorer
        def polluted(sample):
            return {}

        benchmarks_after_pollution = get_registered_benchmarks()
        assert len(benchmarks_after_pollution) == 2, (
            "Should have 2 benchmarks after manual registration"
        )

        # Second import - should clear first
        import_benchmark(str(temp_benchmark_file), "test_import")
        benchmarks_after_second = get_registered_benchmarks()

        # Only test-import should remain (polluted-bench cleared)
        assert "test_import" in benchmarks_after_second, (
            "test-import should still be registered"
        )
        assert "polluted-bench" not in benchmarks_after_second, (
            "import_benchmark should have cleared previous registry state"
        )

    def test_import_benchmark_not_found(self, temp_benchmark_file):
        """Test clear error when requested benchmark doesn't exist."""
        with pytest.raises(ValueError, match="nonexistent.*not found"):
            import_benchmark(str(temp_benchmark_file), "nonexistent")

    def test_import_benchmark_module_not_found(self):
        """Test clear error when benchmark module file doesn't exist."""
        # This should raise during import, not during lookup
        with pytest.raises((FileNotFoundError, ImportError, ModuleNotFoundError)):
            import_benchmark("/tmp/does_not_exist_12345.py", "whatever")

    def test_import_benchmark_available_list_in_error(self, temp_benchmark_file):
        """Test that error message lists available benchmarks.

        This makes debugging much easier when wrong name is used.
        """
        try:
            import_benchmark(str(temp_benchmark_file), "wrong-name")
            pytest.fail("Expected ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "test_import" in error_msg, (
                f"Error message should list available benchmarks, got: {error_msg}"
            )


class TestRunEvalLoop:
    """Tests for run_eval_loop function."""

    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock BenchmarkDefinition with ScorerInput-based scorer."""

        def simple_scorer(sample: ScorerInput):
            return {"correct": sample.target.lower() in sample.response.lower()}

        return BenchmarkDefinition(
            name="test-loop",
            normalized_name="test_loop",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=simple_scorer,
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for eval loop tests."""
        return [
            {"question": "Is the sky blue?", "answer": "yes"},
            {"question": "Is water dry?", "answer": "no"},
            {"question": "Do cats meow?", "answer": "yes"},
        ]

    @pytest.fixture
    def mock_model_call_fn(self):
        """Mock model call function."""
        return MagicMock(return_value="Yes, that is correct.")

    def test_run_eval_loop_basic(
        self, mock_benchmark, sample_dataset, mock_model_call_fn
    ):
        """Test basic eval loop execution.

        Validates:
        - All samples are processed
        - Prompts are rendered with sample fields
        - Model call function is invoked correctly
        - Scorer is called with ScorerInput
        - Returns list of score dicts
        """
        scores, _predictions = run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert len(scores) == 3, (
            f"Expected 3 scores (one per sample), got {len(scores)}"
        )

        # Verify all scores have expected keys
        for score in scores:
            assert "correct" in score, (
                f"Expected 'correct' key in score, got {list(score.keys())}"
            )

        # Verify model was called 3 times with correct signature
        assert mock_model_call_fn.call_count == 3, (
            f"Expected 3 model calls, got {mock_model_call_fn.call_count}"
        )

        for call in mock_model_call_fn.call_args_list:
            args, kwargs = call
            assert len(args) == 2, (
                f"Expected (prompt, endpoint_type) args, got {len(args)}"
            )
            prompt, endpoint = args
            assert "Q:" in prompt, f"Expected rendered prompt, got: {prompt}"
            assert endpoint == "chat", (
                f"Expected endpoint_type='chat', got '{endpoint}'"
            )

    def test_run_eval_loop_missing_prompt_field(
        self, mock_benchmark, mock_model_call_fn
    ):
        """Test that samples missing prompt fields are skipped with warning.

        Middle sample missing 'question' -> should be skipped, not crashed.
        """
        dataset = [
            {"question": "q1", "answer": "yes"},
            {"WRONG_KEY": "q2", "answer": "no"},  # Missing 'question'
            {"question": "q3", "answer": "yes"},
        ]

        scores, _preds = run_eval_loop(
            bench=mock_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert len(scores) == 2, (
            f"Expected 2 scores (1 skipped for missing field), got {len(scores)}"
        )

        # Model should only be called twice
        assert mock_model_call_fn.call_count == 2, (
            f"Expected 2 model calls (1 sample skipped), got {mock_model_call_fn.call_count}"
        )

    def test_run_eval_loop_model_error_skips_sample(
        self, mock_benchmark, sample_dataset
    ):
        """Test that model errors skip the sample, not crash the loop.

        Model fails on sample 1 -> that sample skipped, others processed.
        """
        mock_model_call_fn = MagicMock(
            side_effect=[
                "Yes",  # sample 0 succeeds
                Exception("Model timeout"),  # sample 1 fails
                "No",  # sample 2 succeeds
            ]
        )

        scores, _preds = run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert len(scores) == 2, (
            f"Expected 2 scores (1 skipped for model error), got {len(scores)}"
        )

    def test_run_eval_loop_empty_dataset(self, mock_benchmark, mock_model_call_fn):
        """Test that empty dataset returns empty score list."""
        scores, _preds = run_eval_loop(
            bench=mock_benchmark,
            dataset=[],
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert scores == [], f"Expected empty list for empty dataset, got {scores}"
        assert mock_model_call_fn.call_count == 0, (
            "Model should not be called for empty dataset"
        )

    def test_run_eval_loop_scorer_receives_metadata(
        self, sample_dataset, mock_model_call_fn
    ):
        """Test that scorer receives ScorerInput with full sample dict as metadata.

        This allows scorers to access additional fields beyond target.
        """
        scorer_calls = []

        def capture_scorer(sample: ScorerInput):
            scorer_calls.append(sample)
            return {"match": True}

        mock_benchmark = BenchmarkDefinition(
            name="test-metadata",
            normalized_name="test_metadata",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=capture_scorer,
        )

        scores, _preds = run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        # Verify scorer was called with ScorerInput containing full sample dict
        assert len(scorer_calls) == 3, (
            f"Expected 3 scorer calls, got {len(scorer_calls)}"
        )

        for i, sample in enumerate(scorer_calls):
            assert isinstance(sample, ScorerInput), (
                f"Sample {i}: expected ScorerInput, got {type(sample)}"
            )
            expected_row = sample_dataset[i]
            assert sample.response == mock_model_call_fn.return_value, (
                f"Sample {i}: expected response from model call fn"
            )
            assert sample.target == expected_row["answer"], (
                f"Sample {i}: expected target to match answer field"
            )
            assert sample.metadata == expected_row, (
                f"Sample {i}: expected metadata to be full sample dict"
            )

    def test_run_eval_loop_endpoint_type_passed_through(
        self, mock_benchmark, sample_dataset
    ):
        """Test that endpoint_type is passed to model_call_fn correctly.

        Tests both chat and completions endpoint types.
        """
        mock_chat = MagicMock(return_value="Response")
        run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_chat,
            endpoint_type="chat",
        )

        for call in mock_chat.call_args_list:
            args, _ = call
            assert args[1] == "chat", f"Expected endpoint_type='chat', got '{args[1]}'"

        mock_completions = MagicMock(return_value="Response")
        run_eval_loop(
            bench=mock_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_completions,
            endpoint_type="completions",
        )

        for call in mock_completions.call_args_list:
            args, _ = call
            assert args[1] == "completions", (
                f"Expected endpoint_type='completions', got '{args[1]}'"
            )


class TestStandardStrategy:
    """Tests for StandardStrategy.evaluate_sample."""

    @pytest.fixture
    def strategy(self):
        """Create a StandardStrategy instance."""
        return StandardStrategy()

    @pytest.fixture
    def sample_row(self):
        """A single sample row from a dataset."""
        return {"question": "Is the sky blue?", "answer": "yes", "category": "science"}

    @pytest.fixture
    def mock_model_call_fn(self):
        """Mock model call returning a fixed response."""
        return MagicMock(return_value="Yes, the sky is blue.")

    def test_standard_strategy_constructs_scorer_input(
        self, strategy, sample_row, mock_model_call_fn
    ):
        """Test that the scorer receives a ScorerInput with correct fields.

        StandardStrategy should construct ScorerInput from the model response,
        target field, full row as metadata, model_call_fn, and extra_config.
        """
        received_inputs = []

        def capturing_scorer(inp: ScorerInput):
            received_inputs.append(inp)
            return {"correct": True}

        bench = BenchmarkDefinition(
            name="test-strategy",
            normalized_name="test_strategy",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=capturing_scorer,
        )

        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=sample_row,
            bench=bench,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert scores == {"correct": True}, f"Expected scores dict, got {scores}"
        assert len(received_inputs) == 1, (
            f"Expected 1 scorer call, got {len(received_inputs)}"
        )

        inp = received_inputs[0]
        assert isinstance(inp, ScorerInput), f"Expected ScorerInput, got {type(inp)}"
        assert inp.response == "Yes, the sky is blue.", (
            f"Expected model response, got '{inp.response}'"
        )
        assert inp.target == "yes", f"Expected target 'yes', got '{inp.target}'"
        assert inp.metadata == sample_row, (
            f"Expected full row as metadata, got {inp.metadata}"
        )

    def test_standard_strategy_missing_field(self, strategy, mock_model_call_fn):
        """Test that a missing prompt field returns skipped_missing_field status.

        When the sample lacks a field required by the prompt template,
        the strategy should return (None, SampleResult) without calling
        the model or scorer.
        """
        bench = BenchmarkDefinition(
            name="test-missing",
            normalized_name="test_missing",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=lambda inp: {"correct": True},
        )

        row_missing_field = {"WRONG_KEY": "value", "answer": "yes"}

        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row_missing_field,
            bench=bench,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert scores is None, "Expected None scores for missing field"
        assert prediction is not None, "Expected SampleResult for missing field"
        assert isinstance(prediction, SampleResult), (
            f"Expected SampleResult, got {type(prediction)}"
        )
        assert prediction.status == "skipped_missing_field", (
            f"Expected status 'skipped_missing_field', got '{prediction.status}'"
        )
        assert prediction.error is not None, "Expected error message for missing field"
        assert "question" in prediction.error, (
            f"Expected error to mention missing field 'question', got '{prediction.error}'"
        )
        # Model should not have been called
        assert mock_model_call_fn.call_count == 0, (
            "Model should not be called when prompt field is missing"
        )

    def test_standard_strategy_model_error(self, strategy):
        """Test that a model exception returns skipped_model_error status.

        When the model call raises an exception, the strategy should
        return (None, SampleResult) without calling the scorer.
        """
        failing_model = MagicMock(side_effect=RuntimeError("Connection timeout"))
        scorer_mock = MagicMock(return_value={"correct": True})

        bench = BenchmarkDefinition(
            name="test-model-error",
            normalized_name="test_model_error",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=scorer_mock,
        )

        row = {"question": "Is the sky blue?", "answer": "yes"}

        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=failing_model,
            endpoint_type="chat",
        )

        assert scores is None, "Expected None scores for model error"
        assert prediction is not None, "Expected SampleResult for model error"
        assert prediction.status == "skipped_model_error", (
            f"Expected status 'skipped_model_error', got '{prediction.status}'"
        )
        assert "Connection timeout" in prediction.error, (
            f"Expected error message to contain exception text, got '{prediction.error}'"
        )
        assert prediction.response is None, "Expected None response when model fails"
        # Scorer should not have been called
        assert scorer_mock.call_count == 0, (
            "Scorer should not be called when model fails"
        )

    def test_standard_strategy_scorer_error(self, strategy, mock_model_call_fn):
        """Test that a scorer exception returns skipped_scorer_error status.

        This is the P0 bug fix test: previously scorer exceptions would crash
        the entire eval loop. Now they are caught and the sample is skipped.
        """

        def exploding_scorer(inp: ScorerInput):
            raise ValueError("Cannot parse model output")

        bench = BenchmarkDefinition(
            name="test-scorer-error",
            normalized_name="test_scorer_error",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=exploding_scorer,
        )

        row = {"question": "Is the sky blue?", "answer": "yes"}

        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert scores is None, "Expected None scores for scorer error"
        assert prediction is not None, "Expected SampleResult for scorer error"
        assert prediction.status == "skipped_scorer_error", (
            f"Expected status 'skipped_scorer_error', got '{prediction.status}'"
        )
        assert "Cannot parse model output" in prediction.error, (
            f"Expected scorer error message, got '{prediction.error}'"
        )
        # The model was called successfully before scorer failed
        assert prediction.response == mock_model_call_fn.return_value, (
            "Expected response to be populated even though scorer failed"
        )
        assert prediction.prompt == "Q: Is the sky blue?\nA:", (
            f"Expected rendered prompt, got '{prediction.prompt}'"
        )

    def test_standard_strategy_scorer_receives_model_call_fn(
        self, strategy, mock_model_call_fn
    ):
        """Test that ScorerInput.model_call_fn is set to the model call function.

        This enables LLM-as-Judge scorers to make additional model calls.
        """
        received_model_fn = []

        def inspector_scorer(inp: ScorerInput):
            received_model_fn.append(inp.model_call_fn)
            return {"score": 1.0}

        bench = BenchmarkDefinition(
            name="test-model-fn",
            normalized_name="test_model_fn",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=inspector_scorer,
        )

        row = {"question": "Is the sky blue?", "answer": "yes"}

        strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert len(received_model_fn) == 1, "Expected scorer to be called once"
        assert received_model_fn[0] is mock_model_call_fn, (
            "Expected ScorerInput.model_call_fn to be the same model_call_fn passed to evaluate_sample"
        )

    def test_standard_strategy_scorer_receives_extra_config(
        self, strategy, mock_model_call_fn
    ):
        """Test that ScorerInput.config contains bench.extra_config.

        Users pass extra kwargs to @benchmark which are forwarded through
        extra_config to the scorer via ScorerInput.config.
        """
        received_configs = []

        def config_scorer(inp: ScorerInput):
            received_configs.append(inp.config)
            return {"score": 1.0}

        bench = BenchmarkDefinition(
            name="test-config",
            normalized_name="test_config",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=config_scorer,
            extra_config={"judge": {"model_id": "gpt-4"}, "temperature": 0.0},
        )

        row = {"question": "Is the sky blue?", "answer": "yes"}

        strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
        )

        assert len(received_configs) == 1, "Expected scorer to be called once"
        assert received_configs[0] == {
            "judge": {"model_id": "gpt-4"},
            "temperature": 0.0,
        }, f"Expected extra_config in ScorerInput.config, got {received_configs[0]}"


class TestRunEvalLoopWithStrategy:
    """Tests for run_eval_loop strategy integration and loop control parameters."""

    @pytest.fixture
    def simple_benchmark(self):
        """A minimal benchmark for strategy integration tests."""

        def simple_scorer(inp: ScorerInput):
            return {"correct": inp.target.lower() in inp.response.lower()}

        return BenchmarkDefinition(
            name="test-strategy-loop",
            normalized_name="test_strategy_loop",
            dataset="unused",
            prompt="Q: {question}\nA:",
            target_field="answer",
            scorer_fn=simple_scorer,
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for strategy tests."""
        return [
            {"question": "Is the sky blue?", "answer": "yes"},
            {"question": "Is water dry?", "answer": "no"},
            {"question": "Do cats meow?", "answer": "yes"},
        ]

    @pytest.fixture
    def mock_model_call_fn(self):
        """Mock model call function."""
        return MagicMock(return_value="Yes, that is correct.")

    def test_custom_strategy_used(
        self, simple_benchmark, sample_dataset, mock_model_call_fn
    ):
        """Test that a custom strategy is used when provided.

        When a strategy is passed to run_eval_loop, it should be called
        for each sample instead of the default StandardStrategy.
        """
        mock_strategy = MagicMock()
        mock_strategy.evaluate_sample.return_value = (
            {"custom_score": 1.0},
            SampleResult(
                sample_id=0,
                prompt="p",
                response="r",
                target="t",
                scores={"custom_score": 1.0},
                status="scored",
            ),
        )

        scores, _preds = run_eval_loop(
            bench=simple_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            strategy=mock_strategy,
        )

        assert mock_strategy.evaluate_sample.call_count == 3, (
            f"Expected custom strategy called 3 times, got {mock_strategy.evaluate_sample.call_count}"
        )
        assert len(scores) == 3, (
            f"Expected 3 scores from custom strategy, got {len(scores)}"
        )
        for s in scores:
            assert "custom_score" in s, (
                f"Expected custom_score key, got {list(s.keys())}"
            )

    def test_default_strategy_is_standard(
        self, simple_benchmark, sample_dataset, mock_model_call_fn
    ):
        """Test that StandardStrategy is used when no strategy is provided.

        Verify the default behavior produces standard evaluation results,
        confirming StandardStrategy is wired in as the default.
        """
        scores, _preds = run_eval_loop(
            bench=simple_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            # strategy=None (default)
        )

        # StandardStrategy processes all samples and uses the benchmark scorer
        assert len(scores) == 3, (
            f"Expected 3 scores from default strategy, got {len(scores)}"
        )
        for s in scores:
            assert "correct" in s, (
                f"Expected 'correct' key from simple_scorer, got {list(s.keys())}"
            )

    def test_max_consecutive_errors(self, simple_benchmark, mock_model_call_fn):
        """Test that eval loop aborts after N consecutive failures.

        With max_consecutive_errors=2 and a dataset where samples 1 and 2
        both fail, the loop should raise RuntimeError after the second failure.
        """
        # Strategy that fails on samples 1 and 2
        call_count = 0

        class FailingStrategy:
            def evaluate_sample(self, idx, row, bench, model_call_fn, endpoint_type):
                nonlocal call_count
                call_count += 1
                if idx == 0:
                    return (
                        {"score": 1.0},
                        SampleResult(
                            sample_id=idx,
                            prompt="p",
                            response="r",
                            target="t",
                            scores={"score": 1.0},
                            status="scored",
                        ),
                    )
                # Samples 1 and 2 fail
                return (
                    None,
                    SampleResult(
                        sample_id=idx,
                        prompt="p",
                        response=None,
                        target="t",
                        scores=None,
                        status="skipped_model_error",
                        error="Fake error",
                    ),
                )

        dataset = [
            {"question": "q0", "answer": "a0"},
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
            {"question": "q3", "answer": "a3"},
        ]

        with pytest.raises(RuntimeError, match="consecutive errors"):
            run_eval_loop(
                bench=simple_benchmark,
                dataset=dataset,
                model_call_fn=mock_model_call_fn,
                endpoint_type="chat",
                strategy=FailingStrategy(),
                max_consecutive_errors=2,
            )

        # Sample 0 succeeds, sample 1 fails, sample 2 fails -> abort
        assert call_count == 3, f"Expected 3 calls before abort, got {call_count}"

    def test_fail_on_skip(self, simple_benchmark, mock_model_call_fn):
        """Test that RuntimeError is raised when fail_on_skip=True and a sample is skipped.

        Even a single skipped sample should cause immediate failure.
        """

        class SkipFirstStrategy:
            def evaluate_sample(self, idx, row, bench, model_call_fn, endpoint_type):
                if idx == 0:
                    return (
                        None,
                        SampleResult(
                            sample_id=idx,
                            prompt="p",
                            response=None,
                            target="t",
                            scores=None,
                            status="skipped_missing_field",
                            error="Missing field",
                        ),
                    )
                return (
                    {"score": 1.0},
                    SampleResult(
                        sample_id=idx,
                        prompt="p",
                        response="r",
                        target="t",
                        scores={"score": 1.0},
                        status="scored",
                    ),
                )

        dataset = [
            {"question": "q0", "answer": "a0"},
            {"question": "q1", "answer": "a1"},
        ]

        with pytest.raises(RuntimeError, match="Sample 0 failed"):
            run_eval_loop(
                bench=simple_benchmark,
                dataset=dataset,
                model_call_fn=mock_model_call_fn,
                endpoint_type="chat",
                strategy=SkipFirstStrategy(),
                fail_on_skip=True,
            )

    def test_save_predictions(
        self, simple_benchmark, sample_dataset, mock_model_call_fn
    ):
        """Test that predictions list is populated when save_predictions=True.

        Each sample should have a corresponding SampleResult in the
        predictions list with status, prompt, response, and scores.
        """
        scores, predictions = run_eval_loop(
            bench=simple_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=True,
        )

        assert len(predictions) == 3, f"Expected 3 predictions, got {len(predictions)}"
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"

        for i, pred in enumerate(predictions):
            assert isinstance(pred, SampleResult), (
                f"Prediction {i}: expected SampleResult, got {type(pred)}"
            )
            assert pred.status == "scored", (
                f"Prediction {i}: expected status 'scored', got '{pred.status}'"
            )
            assert pred.sample_id == i, (
                f"Prediction {i}: expected sample_id={i}, got {pred.sample_id}"
            )
            assert pred.prompt is not None, f"Prediction {i}: expected non-None prompt"
            assert pred.response is not None, (
                f"Prediction {i}: expected non-None response"
            )
            assert pred.scores is not None, f"Prediction {i}: expected non-None scores"
            assert "correct" in pred.scores, (
                f"Prediction {i}: expected 'correct' in scores"
            )

    def test_save_predictions_includes_skipped(
        self, simple_benchmark, mock_model_call_fn
    ):
        """Test that skipped samples are included in predictions when save_predictions=True."""
        dataset = [
            {"question": "q1", "answer": "yes"},
            {"WRONG_KEY": "q2", "answer": "no"},  # Will be skipped
            {"question": "q3", "answer": "yes"},
        ]

        scores, predictions = run_eval_loop(
            bench=simple_benchmark,
            dataset=dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=True,
        )

        assert len(scores) == 2, f"Expected 2 scores (1 skipped), got {len(scores)}"
        assert len(predictions) == 3, (
            f"Expected 3 predictions (including skipped), got {len(predictions)}"
        )

        # Find the skipped prediction
        skipped = [p for p in predictions if p.status != "scored"]
        assert len(skipped) == 1, f"Expected 1 skipped prediction, got {len(skipped)}"
        assert skipped[0].status == "skipped_missing_field", (
            f"Expected skipped_missing_field, got '{skipped[0].status}'"
        )

    def test_save_predictions_false_returns_empty(
        self, simple_benchmark, sample_dataset, mock_model_call_fn
    ):
        """Test that predictions list is empty when save_predictions=False (default)."""
        scores, predictions = run_eval_loop(
            bench=simple_benchmark,
            dataset=sample_dataset,
            model_call_fn=mock_model_call_fn,
            endpoint_type="chat",
            save_predictions=False,
        )

        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        assert predictions == [], (
            f"Expected empty predictions when save_predictions=False, got {len(predictions)}"
        )


# ---------------------------------------------------------------------------
# Gap 3: Structured Ground Truth tests
# ---------------------------------------------------------------------------


class TestStructuredTarget:
    """Tests for structured (non-string) target support."""

    def test_dict_target_passed_through(self):
        """Validate that a dict target is passed through to the scorer without str() coercion."""
        received_targets = []

        def capture_scorer(inp: ScorerInput):
            received_targets.append(inp.target)
            return {"score": 1.0}

        bench = BenchmarkDefinition(
            name="dict-target",
            normalized_name="dict_target",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=capture_scorer,
        )

        row = {"question": "test", "answer": {"key": "value", "nested": [1, 2]}}
        mock_model = MagicMock(return_value="response")

        strategy = StandardStrategy()
        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert scores is not None
        assert isinstance(received_targets[0], dict), (
            f"Expected dict target, got {type(received_targets[0])}"
        )
        assert received_targets[0] == {"key": "value", "nested": [1, 2]}
        assert isinstance(prediction.target, dict), (
            f"Expected dict in SampleResult.target, got {type(prediction.target)}"
        )

    def test_list_target_passed_through(self):
        """Validate that a list target is passed through to the scorer without str() coercion."""
        received_targets = []

        def capture_scorer(inp: ScorerInput):
            received_targets.append(inp.target)
            return {"score": 1.0}

        bench = BenchmarkDefinition(
            name="list-target",
            normalized_name="list_target",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=capture_scorer,
        )

        row = {"question": "test", "answer": ["a", "b", "c"]}
        mock_model = MagicMock(return_value="response")

        strategy = StandardStrategy()
        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert scores is not None
        assert isinstance(received_targets[0], list), (
            f"Expected list target, got {type(received_targets[0])}"
        )
        assert received_targets[0] == ["a", "b", "c"]

    def test_scorer_input_accepts_any_target(self):
        """Validate ScorerInput.target accepts dict, list, int, etc."""
        inp_dict = ScorerInput(response="r", target={"a": 1}, metadata={})
        assert isinstance(inp_dict.target, dict)

        inp_list = ScorerInput(response="r", target=[1, 2, 3], metadata={})
        assert isinstance(inp_list.target, list)

        inp_int = ScorerInput(response="r", target=42, metadata={})
        assert inp_int.target == 42


# ---------------------------------------------------------------------------
# Gap 4: Eval-Only Mode tests
# ---------------------------------------------------------------------------


class TestEvalOnlyStrategy:
    """Tests for EvalOnlyStrategy."""

    def test_uses_dataset_response(self):
        """Validate EvalOnlyStrategy reads response from dataset instead of calling model."""

        def simple_scorer(inp: ScorerInput):
            return {"match": inp.target in inp.response}

        bench = BenchmarkDefinition(
            name="eval-only",
            normalized_name="eval_only",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=simple_scorer,
        )

        row = {"question": "Is sky blue?", "answer": "yes", "model_output": "Yes it is"}
        mock_model = MagicMock(return_value="SHOULD NOT BE CALLED")

        strategy = EvalOnlyStrategy("model_output")
        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert scores is not None
        assert prediction.response == "Yes it is", (
            f"Expected response from dataset, got '{prediction.response}'"
        )
        mock_model.assert_not_called()

    def test_model_never_called(self):
        """Validate that model_call_fn is never invoked in eval-only mode."""
        dataset = [
            {"question": "q1", "answer": "a1", "resp": "r1"},
            {"question": "q2", "answer": "a2", "resp": "r2"},
        ]
        mock_model = MagicMock()

        scores, _ = run_eval_loop(
            bench=BenchmarkDefinition(
                name="eval-only-no-model",
                normalized_name="eval_only_no_model",
                dataset="unused",
                prompt="Q: {question}",
                target_field="answer",
                scorer_fn=lambda inp: {"ok": True},
                response_field="resp",
            ),
            dataset=dataset,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert len(scores) == 2
        mock_model.assert_not_called()

    def test_missing_response_field_skips(self):
        """Validate that a missing response field skips the sample gracefully."""
        bench = BenchmarkDefinition(
            name="eval-only-missing",
            normalized_name="eval_only_missing",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=lambda inp: {"ok": True},
        )

        row = {"question": "q", "answer": "a"}  # No "resp" field
        mock_model = MagicMock()

        strategy = EvalOnlyStrategy("resp")
        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert scores is None
        assert prediction.status == "skipped_missing_field"
        assert "resp" in prediction.error

    def test_auto_selection_eval_only(self):
        """Validate that run_eval_loop auto-selects EvalOnlyStrategy when response_field is set."""
        bench = BenchmarkDefinition(
            name="auto-eval-only",
            normalized_name="auto_eval_only",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=lambda inp: {"match": True},
            response_field="output",
        )

        dataset = [{"question": "q", "answer": "a", "output": "model said this"}]
        mock_model = MagicMock()

        scores, _ = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert len(scores) == 1
        mock_model.assert_not_called()

    def test_auto_selection_standard_when_no_response_field(self):
        """Validate that run_eval_loop uses StandardStrategy when response_field is None."""
        bench = BenchmarkDefinition(
            name="auto-standard",
            normalized_name="auto_standard",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=lambda inp: {"match": True},
        )

        dataset = [{"question": "q", "answer": "a"}]
        mock_model = MagicMock(return_value="response")

        scores, _ = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert len(scores) == 1
        mock_model.assert_called_once()


# ---------------------------------------------------------------------------
# Gap 2: Jinja2 Template Engine tests
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    """Tests for render_prompt helper function."""

    def test_format_string_unchanged(self):
        """Validate that a Python format string renders correctly with is_jinja2=False."""
        result = render_prompt("Q: {question}\nA:", {"question": "Is sky blue?"})
        assert result == "Q: Is sky blue?\nA:"

    def test_jinja2_for_loop(self):
        """Validate Jinja2 for-loop over a list of documents."""
        template = "{% for d in docs %}{{ d.title }} {% endfor %}"
        row = {"docs": [{"title": "A"}, {"title": "B"}, {"title": "C"}]}

        result = render_prompt(template, row, is_jinja2=True)
        assert "A" in result and "B" in result and "C" in result

    def test_jinja2_conditional(self):
        """Validate Jinja2 conditional rendering."""
        template = "{% if show_hint %}Hint: {{ hint }}{% endif %}\nQ: {{ question }}"
        row = {"show_hint": True, "hint": "Think carefully", "question": "2+2?"}

        result = render_prompt(template, row, is_jinja2=True)
        assert "Hint: Think carefully" in result
        assert "Q: 2+2?" in result

    def test_jinja2_no_conditional(self):
        """Validate Jinja2 conditional is skipped when False."""
        template = "{% if show_hint %}Hint: {{ hint }}{% endif %}Q: {{ question }}"
        row = {"show_hint": False, "hint": "unused", "question": "2+2?"}

        result = render_prompt(template, row, is_jinja2=True)
        assert "Hint" not in result
        assert "Q: 2+2?" in result


class TestJinja2InStandardStrategy:
    """Tests for Jinja2 rendering in StandardStrategy."""

    def test_jinja2_prompt_in_strategy(self):
        """Validate that StandardStrategy uses Jinja2 when bench._is_jinja2 is True."""
        bench = BenchmarkDefinition(
            name="jinja2-strategy",
            normalized_name="jinja2_strategy",
            dataset="unused",
            prompt="{% for d in docs %}{{ d.title }} {% endfor %}",
            target_field="answer",
            scorer_fn=lambda inp: {"ok": True},
            _is_jinja2=True,
        )

        row = {"docs": [{"title": "X"}, {"title": "Y"}], "answer": "a"}
        mock_model = MagicMock(return_value="response")

        strategy = StandardStrategy()
        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=mock_model,
            endpoint_type="chat",
        )

        assert scores is not None
        assert "X" in prediction.prompt and "Y" in prediction.prompt


# ---------------------------------------------------------------------------
# Gap 1: System Prompt tests
# ---------------------------------------------------------------------------


class TestSystemPromptInStrategy:
    """Tests for system prompt support in StandardStrategy."""

    def test_system_prompt_passed_to_model_call(self):
        """Validate that system_prompt is rendered and passed to model_call_fn."""
        received_kwargs = []

        def capturing_model(prompt, endpoint_type, **kwargs):
            received_kwargs.append(kwargs)
            return "response"

        bench = BenchmarkDefinition(
            name="sys-prompt",
            normalized_name="sys_prompt",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=lambda inp: {"ok": True},
            system_prompt="You are a {role} assistant.",
        )

        row = {"question": "test", "answer": "a", "role": "helpful"}

        strategy = StandardStrategy()
        scores, prediction = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=capturing_model,
            endpoint_type="chat",
        )

        assert scores is not None
        assert len(received_kwargs) == 1
        assert received_kwargs[0]["system_prompt"] == "You are a helpful assistant."

    def test_no_system_prompt_backward_compat(self):
        """Validate backward compatibility: no system_prompt means None is passed."""
        received_kwargs = []

        def capturing_model(prompt, endpoint_type, **kwargs):
            received_kwargs.append(kwargs)
            return "response"

        bench = BenchmarkDefinition(
            name="no-sys-prompt",
            normalized_name="no_sys_prompt",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=lambda inp: {"ok": True},
        )

        row = {"question": "test", "answer": "a"}

        strategy = StandardStrategy()
        scores, _ = strategy.evaluate_sample(
            idx=0,
            row=row,
            bench=bench,
            model_call_fn=capturing_model,
            endpoint_type="chat",
        )

        assert scores is not None
        assert received_kwargs[0]["system_prompt"] is None

    def test_system_prompt_per_sample_rendering(self):
        """Validate that system prompt is rendered per-sample with row data."""
        prompts_seen = []

        def capturing_model(prompt, endpoint_type, **kwargs):
            prompts_seen.append(kwargs.get("system_prompt"))
            return "response"

        bench = BenchmarkDefinition(
            name="per-sample-sys",
            normalized_name="per_sample_sys",
            dataset="unused",
            prompt="Q: {question}",
            target_field="answer",
            scorer_fn=lambda inp: {"ok": True},
            system_prompt="Domain: {domain}",
        )

        dataset = [
            {"question": "q1", "answer": "a1", "domain": "math"},
            {"question": "q2", "answer": "a2", "domain": "science"},
        ]

        scores, _ = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=capturing_model,
            endpoint_type="chat",
        )

        assert len(scores) == 2
        assert prompts_seen[0] == "Domain: math"
        assert prompts_seen[1] == "Domain: science"
