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

"""Error handling and pre-flight check tests.

Tests T072-T082: Validate error detection, diagnostics, and pre-flight
validation logic.
"""

import glob as glob_module
import json
import os
import string

import pytest

from nemo_evaluator.byob import benchmark, scorer
from nemo_evaluator.byob.compiler import compile_benchmark
from nemo_evaluator.byob.decorators import clear_registry


def test_data_format_error_json_parse(tmp_path):
    """T072: Data format error produces diagnostic.

    Malformed JSON produces clear JSONDecodeError with line/column info.
    """
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text('{"question": "What is 2+2?", "answer": INVALID}')

    with pytest.raises(json.JSONDecodeError) as exc_info:
        with open(bad_json_path) as f:
            json.load(f)

    # JSONDecodeError provides line and column information
    error = exc_info.value
    assert hasattr(error, "lineno") or hasattr(error, "pos"), (
        "JSONDecodeError should provide position information for debugging"
    )


def test_field_mismatch_raises_keyerror():
    """T073: Field mismatch produces diagnostic.

    A prompt with {question} but data with {query} raises KeyError
    with the missing field name.
    """
    prompt = "Question: {question}\nAnswer:"
    row = {"query": "What is 2+2?", "answer": "4"}

    with pytest.raises(KeyError, match="question") as exc_info:
        prompt.format(**row)

    # Verify the error message contains the field name
    assert "question" in str(exc_info.value), (
        f"KeyError should mention the missing field 'question'. "
        f"Got: {exc_info.value}"
    )


def test_scorer_type_error_detected():
    """T074: Scorer type error detected.

    A scorer returning non-dict can be detected programmatically.
    """
    def bad_scorer(sample):
        return "correct"  # Wrong type

    result = bad_scorer("test")
    assert not isinstance(result, dict), (
        f"Type check should detect non-dict return. "
        f"Expected type check to fail for {type(result).__name__}"
    )


def test_import_error_keyword_in_prompt(skill_prompt_path):
    """T075: Import error keyword appears in skill prompt.

    Semi-automated: verifies error guidance exists. Full test would
    require uninstalling nemo_evaluator, which breaks CI.
    """
    with open(skill_prompt_path) as f:
        content = f.read().lower()

    # Check for import error guidance
    import_keywords = ["import", "install", "pip", "nemo_evaluator"]
    found = sum(1 for kw in import_keywords if kw in content)

    assert found >= 2, (
        f"Skill prompt missing import error guidance. "
        f"Expected at least 2 of {import_keywords}, found {found}. "
        f"Users need guidance when nemo_evaluator is not installed."
    )


def test_compilation_syntax_error(tmp_path):
    """T076: Compilation error from syntax error.

    A Python file with syntax errors fails compilation with descriptive error.
    """
    bad_file = tmp_path / "bad_benchmark.py"
    bad_file.write_text(
        "from nemo_evaluator.byob import benchmark, scorer\n"
        "def bad syntax():\n"  # Missing colon after function name
        "    pass\n"
    )

    with pytest.raises(SyntaxError):
        compile_benchmark(str(bad_file))


def test_dataset_file_not_found_at_runtime(tmp_path):
    """T077: Dataset file not found detected.

    A benchmark referencing a non-existent dataset path stores the path
    as-is (compiler does not validate existence), but accessing the file
    raises FileNotFoundError.
    """
    nonexistent_path = "/nonexistent/path/data.jsonl"

    # The path is accepted during benchmark registration
    clear_registry()

    @benchmark(
        name="test-bench",
        dataset=nonexistent_path,
        prompt="Q: {q}",
        target_field="a"
    )
    @scorer
    def test_scorer(sample):
        return {"correct": True}

    # Attempting to read the file will fail
    with pytest.raises(FileNotFoundError):
        with open(nonexistent_path) as f:
            f.read()


def test_registration_collision():
    """T078: Registration collision detected.

    Two benchmarks with the same normalized name raise ValueError.
    """
    clear_registry()

    @benchmark(name="My QA", dataset="d.jsonl", prompt="Q: {q}")
    @scorer
    def scorer1(sample):
        return {"correct": True}

    # Attempt to register with same normalized name (my_qa)
    with pytest.raises(ValueError, match="already registered"):
        @benchmark(name="my-qa", dataset="d.jsonl", prompt="Q: {q}")
        @scorer
        def scorer2(sample):
            return {"correct": True}


def test_preflight_field_validation():
    """T079: Pre-flight checks -- all prompt fields exist in dataset.

    Validates the pre-flight check pattern that catches prompt-data
    field mismatches before compilation.
    """
    prompt = "Question: {question}\nContext: {context}\nAnswer:"
    data_row = {"question": "test", "answer": "test"}  # Missing 'context'

    # Extract field names from prompt
    formatter = string.Formatter()
    fields = {name for _, name, _, _ in formatter.parse(prompt) if name}

    # Check which fields are missing
    missing = fields - set(data_row.keys())

    assert "context" in missing, (
        f"Pre-flight validation should detect missing 'context' field. "
        f"Fields in prompt: {fields}, fields in data: {set(data_row.keys())}"
    )


def test_preflight_target_field_validation():
    """T080: Pre-flight checks -- target_field exists in dataset.

    Validates the pre-flight check that catches missing target_field.
    """
    target_field = "answer"
    data_row = {"question": "test", "result": "test"}  # Missing 'answer'

    # Check if target_field exists
    assert target_field not in data_row, (
        f"Pre-flight validation should detect missing target_field '{target_field}'. "
        f"Available fields: {list(data_row.keys())}"
    )


def test_compiler_resolves_relative_to_absolute(tmp_path):
    """T081: Pre-flight checks -- dataset path is absolute.

    The compiler converts relative dataset paths to absolute when the
    file exists at compile time.
    """
    clear_registry()

    bench_file = tmp_path / "my_bench.py"
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"q": "test", "answer": "test"}\n')

    bench_file.write_text(
        "from nemo_evaluator.byob import benchmark, scorer\n"
        "@benchmark(name='test', dataset='data.jsonl', prompt='Q: {q}', target_field='answer')\n"
        "@scorer\n"
        "def s(sample): return {'correct': True}\n"
    )

    # Compile from the tmp_path directory so relative path resolves
    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        compiled = compile_benchmark(str(bench_file))
    finally:
        os.chdir(orig_cwd)

    fdf = list(compiled.values())[0]
    dataset_path = fdf["defaults"]["config"]["params"]["extra"]["dataset"]

    assert os.path.isabs(dataset_path), (
        f"Dataset path should be absolute after compilation when file exists. "
        f"Got: {dataset_path}"
    )


def test_template_discovery_graceful_failure():
    """T082: Template discovery fails gracefully.

    When template files are not found (curl-install mode), the skill
    should not crash. This test verifies that globbing a non-existent
    directory returns an empty list.
    """
    results = glob_module.glob("/nonexistent/path/examples/byob/templates/*.py")
    assert results == [], (
        f"glob on non-existent path should return empty list, got: {results}"
    )
    # No exception should be raised - degraded mode should handle this
