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

"""Unit tests for BYOB decorators and benchmark registration."""

from pathlib import Path

import pytest
from nemo_evaluator.contrib.byob.decorators import (
    ScorerInput,
    _normalize_name,
    _resolve_prompt,
    _resolve_requirements,
    benchmark,
    clear_registry,
    get_registered_benchmarks,
    scorer,
)


class TestNormalizeName:
    """Tests for the _normalize_name function."""

    @pytest.mark.parametrize(
        "input_name,expected_output",
        [
            ("My QA!", "my_qa"),
            ("Toxicity-Check v2", "toxicity_check_v2"),
            ("SIMPLE", "simple"),
            ("a" * 100, "a" * 50),
            ("hello world", "hello_world"),
            ("test---bench", "test_bench"),
            ("__leading__", "leading"),
        ],
    )
    def test_normalize_name(self, input_name, expected_output):
        """Validate name normalization follows spec rules."""
        actual = _normalize_name(input_name)
        assert actual == expected_output, (
            f"_normalize_name({input_name!r}) returned {actual!r}, "
            f"expected {expected_output!r}"
        )

    @pytest.mark.parametrize(
        "input_name,expected_output",
        [
            ("  spaces  ", "spaces"),
            ("dash-to-under", "dash_to_under"),
            ("123numeric", "123numeric"),
            ("!@#$%", ""),
            ("", ""),
            ("a_b_c", "a_b_c"),
            ("MiXeD CaSe!", "mixed_case"),
            ("___", ""),
        ],
    )
    def test_normalize_name_special_chars(self, input_name, expected_output):
        """Validate normalization handles edge cases correctly."""
        actual = _normalize_name(input_name)
        assert actual == expected_output, (
            f"_normalize_name({input_name!r}) returned {actual!r}, "
            f"expected {expected_output!r}"
        )


class TestBenchmarkDecorator:
    """Tests for the @benchmark decorator."""

    def test_register_benchmark(self):
        """Validate @benchmark registers correctly with all fields accessible."""

        @benchmark(name="test-qa", dataset="data.jsonl", prompt="Q: {q}\nA:")
        @scorer
        def my_scorer(sample):
            return {"correct": True}

        benchmarks = get_registered_benchmarks()
        assert "test_qa" in benchmarks, (
            f"Expected 'test_qa' in registry, got {list(benchmarks.keys())}"
        )

        defn = benchmarks["test_qa"]
        assert defn.name == "test-qa"
        assert defn.normalized_name == "test_qa"
        assert defn.dataset == "data.jsonl"
        assert defn.prompt == "Q: {q}\nA:"
        assert defn.scorer_fn is my_scorer
        assert callable(defn.scorer_fn)

    def test_register_benchmark_default_fields(self):
        """Validate default field values for target_field, endpoint_type, extra_config."""

        @benchmark(name="defaults-test", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn(sample):
            return {"ok": True}

        defn = get_registered_benchmarks()["defaults_test"]
        assert defn.target_field == "target", (
            f"Expected default target_field='target', got {defn.target_field!r}"
        )
        assert defn.endpoint_type == "chat", (
            f"Expected default endpoint_type='chat', got {defn.endpoint_type!r}"
        )
        assert defn.extra_config == {}, (
            f"Expected default extra_config={{}}, got {defn.extra_config!r}"
        )

    def test_register_benchmark_custom_fields(self):
        """Validate custom target_field, endpoint_type, and **kwargs passthrough."""

        @benchmark(
            name="custom",
            dataset="d.jsonl",
            prompt="{x}",
            target_field="answer",
            endpoint_type="completions",
            custom_param="value",
            another=42,
        )
        @scorer
        def fn(sample):
            return {"ok": True}

        defn = get_registered_benchmarks()["custom"]
        assert defn.target_field == "answer"
        assert defn.endpoint_type == "completions"
        assert defn.extra_config == {"custom_param": "value", "another": 42}

    def test_duplicate_name_raises(self):
        """Validate duplicate normalized names raise ValueError."""

        @benchmark(name="duplicate", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn1(sample):
            return {"ok": True}

        with pytest.raises(ValueError, match="already registered"):

            @benchmark(name="Duplicate!", dataset="d2.jsonl", prompt="{y}")
            @scorer
            def fn2(sample):
                return {"ok": True}

    def test_multiple_benchmarks(self):
        """Validate two benchmarks with different names coexist in registry."""

        @benchmark(name="bench-a", dataset="d1.jsonl", prompt="{x}")
        @scorer
        def fn_a(sample):
            return {"ok": True}

        @benchmark(name="bench-b", dataset="d2.jsonl", prompt="{y}")
        @scorer
        def fn_b(sample):
            return {"ok": True}

        benchmarks = get_registered_benchmarks()
        assert len(benchmarks) == 2
        assert "bench_a" in benchmarks
        assert "bench_b" in benchmarks
        assert benchmarks["bench_a"].name == "bench-a"
        assert benchmarks["bench_b"].name == "bench-b"

    def test_benchmark_definition_has_fn_attribute(self):
        """Validate decorated function has _benchmark_definition attribute."""

        @benchmark(name="attr-test", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn(sample):
            return {"ok": True}

        assert hasattr(fn, "_benchmark_definition"), (
            "Expected function to have _benchmark_definition attribute"
        )
        defn = fn._benchmark_definition
        assert defn.name == "attr-test"
        assert defn.normalized_name == "attr_test"
        assert defn.dataset == "d.jsonl"


class TestScorerDecorator:
    """Tests for the @scorer decorator."""

    def test_scorer_decorator(self):
        """Validate @scorer sets _is_scorer=True and preserves functionality."""

        @scorer
        def my_fn(sample):
            return {"score": 1.0}

        assert hasattr(my_fn, "_is_scorer"), (
            "Expected function to have _is_scorer attribute"
        )
        assert my_fn._is_scorer is True

        # Function is still callable and works correctly
        result = my_fn(ScorerInput(response="resp", target="tgt", metadata={}))
        assert result == {"score": 1.0}


class TestRegistryManagement:
    """Tests for registry management functions."""

    def test_clear_registry(self):
        """Validate clear_registry empties a non-empty registry."""

        @benchmark(name="temp", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn(sample):
            return {}

        assert len(get_registered_benchmarks()) == 1, (
            "Registry should have 1 benchmark after registration"
        )

        clear_registry()

        assert len(get_registered_benchmarks()) == 0, (
            "Registry should be empty after clear_registry()"
        )


# ---------------------------------------------------------------------------
# New test classes for ScorerInput, prompt resolution, and requirements
# ---------------------------------------------------------------------------


class TestScorerInput:
    """Tests for the ScorerInput dataclass."""

    def test_construction_required_fields_only(self):
        """Validate ScorerInput can be constructed with only required fields."""
        inp = ScorerInput(response="hello", target="world", metadata={"key": "val"})

        assert inp.response == "hello", (
            f"Expected response='hello', got {inp.response!r}"
        )
        assert inp.target == "world", (
            f"Expected target='world', got {inp.target!r}"
        )
        assert inp.metadata == {"key": "val"}, (
            f"Expected metadata={{'key': 'val'}}, got {inp.metadata!r}"
        )

    def test_construction_all_fields(self):
        """Validate ScorerInput can be constructed with all fields."""
        call_fn = lambda prompt: "mocked response"
        conversation = [{"role": "user", "content": "hi"}]

        inp = ScorerInput(
            response="resp",
            target="tgt",
            metadata={"m": 1},
            model_call_fn=call_fn,
            config={"temperature": 0.7},
            conversation=conversation,
            turn_index=2,
        )

        assert inp.response == "resp", (
            f"Expected response='resp', got {inp.response!r}"
        )
        assert inp.target == "tgt", (
            f"Expected target='tgt', got {inp.target!r}"
        )
        assert inp.metadata == {"m": 1}, (
            f"Expected metadata={{'m': 1}}, got {inp.metadata!r}"
        )
        assert inp.model_call_fn is call_fn, (
            "Expected model_call_fn to be the provided callable"
        )
        assert inp.config == {"temperature": 0.7}, (
            f"Expected config={{'temperature': 0.7}}, got {inp.config!r}"
        )
        assert inp.conversation == conversation, (
            f"Expected conversation={conversation!r}, got {inp.conversation!r}"
        )
        assert inp.turn_index == 2, (
            f"Expected turn_index=2, got {inp.turn_index!r}"
        )

    def test_default_values_for_optional_fields(self):
        """Validate optional fields have correct default values."""
        inp = ScorerInput(response="r", target="t", metadata={})

        assert inp.model_call_fn is None, (
            f"Expected model_call_fn=None, got {inp.model_call_fn!r}"
        )
        assert inp.config == {}, (
            f"Expected config={{}}, got {inp.config!r}"
        )
        assert inp.conversation is None, (
            f"Expected conversation=None, got {inp.conversation!r}"
        )
        assert inp.turn_index is None, (
            f"Expected turn_index=None, got {inp.turn_index!r}"
        )

    def test_import_from_decorators_module(self):
        """Validate ScorerInput is importable from nemo_evaluator.contrib.byob.decorators."""
        from nemo_evaluator.contrib.byob.decorators import ScorerInput as DecScorerInput

        assert DecScorerInput is ScorerInput, (
            "ScorerInput imported from decorators module should be the same class"
        )

    def test_import_from_public_api(self):
        """Validate ScorerInput is importable from nemo_evaluator.contrib.byob (public API)."""
        from nemo_evaluator.contrib.byob import ScorerInput as PubScorerInput

        assert PubScorerInput is ScorerInput, (
            "ScorerInput imported from public API should be the same class"
        )


class TestPromptFileResolution:
    """Tests for the _resolve_prompt helper function."""

    def test_literal_string_returns_as_is(self):
        """Validate a plain string without file extension is returned unchanged."""
        result = _resolve_prompt("Question: {q}\nAnswer:", Path("/nonexistent"))

        assert result == "Question: {q}\nAnswer:", (
            f"Expected literal string back, got {result!r}"
        )

    def test_txt_file_is_read(self, tmp_path):
        """Validate a .txt file path is read and its content returned."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Hello from txt: {name}", encoding="utf-8")

        result = _resolve_prompt("prompt.txt", tmp_path)

        assert result == "Hello from txt: {name}", (
            f"Expected file content, got {result!r}"
        )

    def test_md_file_is_read(self, tmp_path):
        """Validate a .md file path is read and its content returned."""
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("# Markdown prompt\n{question}", encoding="utf-8")

        result = _resolve_prompt("prompt.md", tmp_path)

        assert result == "# Markdown prompt\n{question}", (
            f"Expected file content, got {result!r}"
        )

    def test_jinja_file_is_read(self, tmp_path):
        """Validate a .jinja file path is read and its content returned."""
        prompt_file = tmp_path / "template.jinja"
        prompt_file.write_text("{{ user_input }}", encoding="utf-8")

        result = _resolve_prompt("template.jinja", tmp_path)

        assert result == "{{ user_input }}", (
            f"Expected file content, got {result!r}"
        )

    def test_jinja2_file_is_read(self, tmp_path):
        """Validate a .jinja2 file path is read and its content returned."""
        prompt_file = tmp_path / "template.jinja2"
        prompt_file.write_text("{% for item in items %}{{ item }}{% endfor %}", encoding="utf-8")

        result = _resolve_prompt("template.jinja2", tmp_path)

        assert result == "{% for item in items %}{{ item }}{% endfor %}", (
            f"Expected file content, got {result!r}"
        )

    def test_nonexistent_file_returns_string_as_is(self):
        """Validate non-existent file path with extension returns string as-is (no error)."""
        result = _resolve_prompt("missing.txt", Path("/nonexistent/dir"))

        assert result == "missing.txt", (
            f"Expected non-existent file path to be returned as-is, got {result!r}"
        )

    def test_absolute_path_to_existing_file(self, tmp_path):
        """Validate absolute path to an existing file is read correctly."""
        prompt_file = tmp_path / "abs_prompt.txt"
        prompt_file.write_text("Absolute content: {x}", encoding="utf-8")

        result = _resolve_prompt(str(prompt_file), tmp_path)

        assert result == "Absolute content: {x}", (
            f"Expected file content from absolute path, got {result!r}"
        )

    def test_relative_path_resolved_relative_to_base_dir(self, tmp_path):
        """Validate relative path is resolved relative to base_dir."""
        subdir = tmp_path / "prompts"
        subdir.mkdir()
        prompt_file = subdir / "my_prompt.txt"
        prompt_file.write_text("Relative: {val}", encoding="utf-8")

        result = _resolve_prompt("prompts/my_prompt.txt", tmp_path)

        assert result == "Relative: {val}", (
            f"Expected file content from relative path, got {result!r}"
        )


class TestRequirementsResolution:
    """Tests for the _resolve_requirements helper function."""

    def test_none_returns_empty_list(self):
        """Validate None input returns an empty list."""
        result = _resolve_requirements(None, Path("/any"))

        assert result == [], (
            f"Expected empty list for None requirements, got {result!r}"
        )

    def test_list_returns_list_copy(self):
        """Validate list input returns a copy of the list."""
        original = ["numpy>=1.0", "pandas"]
        result = _resolve_requirements(original, Path("/any"))

        assert result == ["numpy>=1.0", "pandas"], (
            f"Expected list copy, got {result!r}"
        )
        assert result is not original, (
            "Expected a copy, not the same list object"
        )

    def test_string_path_reads_requirements_file(self, tmp_path):
        """Validate string path to requirements.txt is read and parsed."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy>=1.0\npandas\nrequests==2.28.0\n", encoding="utf-8")

        result = _resolve_requirements("requirements.txt", tmp_path)

        assert result == ["numpy>=1.0", "pandas", "requests==2.28.0"], (
            f"Expected parsed requirements, got {result!r}"
        )

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        """Validate comments and blank lines are filtered out."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            "# This is a comment\n"
            "numpy>=1.0\n"
            "\n"
            "  # Another comment\n"
            "pandas\n"
            "\n"
            "  \n"
            "scikit-learn\n",
            encoding="utf-8",
        )

        result = _resolve_requirements("requirements.txt", tmp_path)

        assert result == ["numpy>=1.0", "pandas", "scikit-learn"], (
            f"Expected only non-comment, non-blank lines, got {result!r}"
        )

    def test_nonexistent_file_raises_file_not_found(self):
        """Validate non-existent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Requirements file not found"):
            _resolve_requirements("missing_requirements.txt", Path("/nonexistent/dir"))

    def test_absolute_path_works(self, tmp_path):
        """Validate absolute path to requirements file is read correctly."""
        req_file = tmp_path / "abs_reqs.txt"
        req_file.write_text("flask>=2.0\n", encoding="utf-8")

        result = _resolve_requirements(str(req_file), tmp_path)

        assert result == ["flask>=2.0"], (
            f"Expected parsed requirements from absolute path, got {result!r}"
        )

    def test_relative_path_resolved_relative_to_base_dir(self, tmp_path):
        """Validate relative path is resolved relative to base_dir."""
        subdir = tmp_path / "deps"
        subdir.mkdir()
        req_file = subdir / "reqs.txt"
        req_file.write_text("torch\ntransformers\n", encoding="utf-8")

        result = _resolve_requirements("deps/reqs.txt", tmp_path)

        assert result == ["torch", "transformers"], (
            f"Expected parsed requirements from relative path, got {result!r}"
        )


class TestBenchmarkWithRequirements:
    """Tests for the @benchmark decorator with requirements parameter."""

    def test_requirements_list_stored_in_definition(self):
        """Validate requirements list is stored directly in BenchmarkDefinition."""

        @benchmark(
            name="req-list",
            dataset="d.jsonl",
            prompt="{x}",
            requirements=["numpy>=1.0", "pandas"],
        )
        @scorer
        def fn(sample):
            return {"ok": True}

        defn = get_registered_benchmarks()["req_list"]
        assert defn.requirements == ["numpy>=1.0", "pandas"], (
            f"Expected requirements=['numpy>=1.0', 'pandas'], got {defn.requirements!r}"
        )

    def test_requirements_none_stores_empty_list(self):
        """Validate requirements=None (default) stores empty list."""

        @benchmark(name="req-none", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn(sample):
            return {"ok": True}

        defn = get_registered_benchmarks()["req_none"]
        assert defn.requirements == [], (
            f"Expected empty requirements list, got {defn.requirements!r}"
        )

    def test_requirements_file_path_reads_and_stores_parsed(self, tmp_path):
        """Validate requirements='file.txt' reads, parses, and stores the list."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("# deps\nrogue-score>=0.1.2\nnltk\n", encoding="utf-8")

        # We need to invoke _resolve_requirements through the decorator.
        # The decorator resolves base_dir from the function source file,
        # so we call the helpers directly to test the file-based path.
        from nemo_evaluator.contrib.byob.decorators import BenchmarkDefinition

        resolved_reqs = _resolve_requirements("requirements.txt", tmp_path)
        assert resolved_reqs == ["rogue-score>=0.1.2", "nltk"], (
            f"Expected parsed requirements from file, got {resolved_reqs!r}"
        )

        # Also verify the BenchmarkDefinition can store the result
        defn = BenchmarkDefinition(
            name="req-file",
            normalized_name="req_file",
            dataset="d.jsonl",
            prompt="{x}",
            scorer_fn=lambda sample: {},
            requirements=resolved_reqs,
        )
        assert defn.requirements == ["rogue-score>=0.1.2", "nltk"], (
            f"Expected requirements in definition, got {defn.requirements!r}"
        )


class TestBenchmarkWithPromptFile:
    """Tests for the @benchmark decorator with prompt file resolution."""

    def test_prompt_file_content_resolved_via_helper(self, tmp_path):
        """Validate prompt='prompt.txt' reads file content when file exists."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text(
            "You are a helpful assistant.\nQuestion: {question}\nAnswer:",
            encoding="utf-8",
        )

        resolved = _resolve_prompt("prompt.txt", tmp_path)

        assert resolved == "You are a helpful assistant.\nQuestion: {question}\nAnswer:", (
            f"Expected file content as resolved prompt, got {resolved!r}"
        )

    def test_prompt_literal_not_resolved_as_file(self):
        """Validate a literal prompt string (no file extension) is never file-resolved."""
        literal = "Answer the following question: {q}"
        resolved = _resolve_prompt(literal, Path("/nonexistent"))

        assert resolved == literal, (
            f"Expected literal prompt returned unchanged, got {resolved!r}"
        )

    def test_prompt_file_stored_in_definition_via_helper(self, tmp_path):
        """Validate resolved prompt content is stored in BenchmarkDefinition."""
        from nemo_evaluator.contrib.byob.decorators import BenchmarkDefinition

        prompt_file = tmp_path / "system.md"
        prompt_file.write_text("# System\nEvaluate: {input}", encoding="utf-8")

        resolved = _resolve_prompt("system.md", tmp_path)

        defn = BenchmarkDefinition(
            name="prompt-file",
            normalized_name="prompt_file",
            dataset="d.jsonl",
            prompt=resolved,
            scorer_fn=lambda sample: {},
        )
        assert defn.prompt == "# System\nEvaluate: {input}", (
            f"Expected resolved prompt in definition, got {defn.prompt!r}"
        )


class TestScorerSignatureValidation:
    """Tests for scorer signature validation via _validate_scorer_signature."""

    def test_one_arg_scorer_accepted(self):
        """Validate a scorer with one parameter (ScorerInput) is accepted."""

        @scorer
        def fn(sample):
            return {}

        assert hasattr(fn, "_is_scorer"), (
            "Expected @scorer to set _is_scorer attribute"
        )
        assert fn._is_scorer is True

    def test_two_arg_scorer_accepted(self):
        """Validate a scorer with two parameters (sample, config) is accepted."""

        @scorer
        def fn(sample, config):
            return {}

        assert hasattr(fn, "_is_scorer"), (
            "Expected @scorer to set _is_scorer attribute"
        )
        assert fn._is_scorer is True

    def test_zero_arg_scorer_rejected(self):
        """Validate a scorer with zero parameters is rejected with TypeError."""
        with pytest.raises(TypeError):

            @scorer
            def fn():
                return {}

    def test_three_arg_scorer_rejected(self):
        """Validate a scorer with three required parameters is rejected with TypeError."""
        with pytest.raises(TypeError):

            @scorer
            def fn(response, target, metadata):
                return {}

    def test_four_arg_scorer_rejected(self):
        """Validate a scorer with four required parameters is rejected with TypeError."""
        with pytest.raises(TypeError):

            @scorer
            def fn(response, target, metadata, extra):
                return {}

    def test_rejection_message_mentions_scorer_input(self):
        """Validate the TypeError message for rejected scorers mentions ScorerInput."""
        with pytest.raises(TypeError, match="ScorerInput"):

            @scorer
            def fn(response, target, metadata):
                return {}

    def test_rejection_message_provides_migration_hint(self):
        """Validate the TypeError message provides a migration hint with 'Migrate from'."""
        with pytest.raises(TypeError, match="Migrate from"):

            @scorer
            def fn(response, target, metadata):
                return {}
