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

import pytest
from nemo_evaluator.byob.decorators import (
    _normalize_name,
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
        def my_scorer(response, target, metadata):
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
        def fn(response, target, metadata):
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
        def fn(response, target, metadata):
            return {"ok": True}

        defn = get_registered_benchmarks()["custom"]
        assert defn.target_field == "answer"
        assert defn.endpoint_type == "completions"
        assert defn.extra_config == {"custom_param": "value", "another": 42}

    def test_duplicate_name_raises(self):
        """Validate duplicate normalized names raise ValueError."""

        @benchmark(name="duplicate", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn1(response, target, metadata):
            return {"ok": True}

        with pytest.raises(ValueError, match="already registered"):

            @benchmark(name="Duplicate!", dataset="d2.jsonl", prompt="{y}")
            @scorer
            def fn2(response, target, metadata):
                return {"ok": True}

    def test_multiple_benchmarks(self):
        """Validate two benchmarks with different names coexist in registry."""

        @benchmark(name="bench-a", dataset="d1.jsonl", prompt="{x}")
        @scorer
        def fn_a(response, target, metadata):
            return {"ok": True}

        @benchmark(name="bench-b", dataset="d2.jsonl", prompt="{y}")
        @scorer
        def fn_b(response, target, metadata):
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
        def fn(response, target, metadata):
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
        def my_fn(response, target, metadata):
            return {"score": 1.0}

        assert hasattr(my_fn, "_is_scorer"), (
            "Expected function to have _is_scorer attribute"
        )
        assert my_fn._is_scorer is True

        # Function is still callable and works correctly
        result = my_fn("resp", "tgt", {})
        assert result == {"score": 1.0}


class TestRegistryManagement:
    """Tests for registry management functions."""

    def test_clear_registry(self):
        """Validate clear_registry empties a non-empty registry."""

        @benchmark(name="temp", dataset="d.jsonl", prompt="{x}")
        @scorer
        def fn(response, target, metadata):
            return {}

        assert len(get_registered_benchmarks()) == 1, (
            "Registry should have 1 benchmark after registration"
        )

        clear_registry()

        assert len(get_registered_benchmarks()) == 0, (
            "Registry should be empty after clear_registry()"
        )
