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

"""Template-data cross-validation tests.

Tests T013-T016: Verify template prompts match data fields, data files
are valid JSONL with correct row counts, and target fields exist.
"""

import importlib
import json
import os
import string
import sys

import pytest

from nemo_evaluator.contrib.byob.decorators import (
    clear_registry,
    get_registered_benchmarks,
)

from .conftest import TEMPLATE_DIR, TEMPLATES


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_template_prompt_fields_match_data(template_name):
    """T013: Each template's prompt placeholders match its data fields.

    Extracts {field} placeholders from the template's prompt using
    string.Formatter().parse() and verifies every placeholder exists
    as a key in every JSONL row.
    """
    clear_registry()

    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.py")
    data_path = os.path.join(TEMPLATE_DIR, f"{template_name}_data.jsonl")

    # Import template to register benchmark
    parent = os.path.dirname(template_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    mod_name = os.path.splitext(os.path.basename(template_path))[0]
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    else:
        importlib.import_module(mod_name)

    benchmarks = get_registered_benchmarks()
    assert len(benchmarks) > 0, (
        f"Template {template_name} registered no benchmarks after import. "
        f"Check that the template has @benchmark + @scorer decorators."
    )

    bench = list(benchmarks.values())[0]
    prompt_str = bench.prompt

    # Extract field names from prompt format string
    formatter = string.Formatter()
    field_names = {
        name for _, name, _, _ in formatter.parse(prompt_str) if name is not None
    }

    # Load data
    assert os.path.isfile(data_path), (
        f"Data file not found: {data_path}. "
        f"Expected {template_name}_data.jsonl in templates directory."
    )

    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) > 0, (
        f"Data file is empty: {data_path}. Expected at least one JSONL row."
    )

    # Validate every field in prompt exists in every data row
    for i, row in enumerate(rows):
        for field_name in field_names:
            assert field_name in row, (
                f"Template '{template_name}' row {i}: "
                f"prompt references '{{ {field_name} }}' but field not in data. "
                f"Data keys: {list(row.keys())}. "
                f"Prompt: {prompt_str[:100]}..."
            )


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_template_target_field_exists_in_data(template_name):
    """T014: Each template's target_field exists in its data.

    Validates that the target_field specified in @benchmark() exists
    as a key in every JSONL row.
    """
    clear_registry()

    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.py")
    data_path = os.path.join(TEMPLATE_DIR, f"{template_name}_data.jsonl")

    # Import template to register benchmark
    parent = os.path.dirname(template_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    mod_name = os.path.splitext(os.path.basename(template_path))[0]
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    else:
        importlib.import_module(mod_name)

    benchmarks = get_registered_benchmarks()
    bench = list(benchmarks.values())[0]
    target_field = bench.target_field

    # Load data
    with open(data_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Validate target_field exists in every row
    for i, row in enumerate(rows):
        assert target_field in row, (
            f"Template '{template_name}' row {i}: "
            f"target_field '{target_field}' not found in data. "
            f"Data keys: {list(row.keys())}"
        )


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_template_data_has_3_rows(template_name):
    """T015: Each data file has exactly 3 rows.

    Architecture doc requires exactly 3 sample rows per template data file.
    """
    data_path = os.path.join(TEMPLATE_DIR, f"{template_name}_data.jsonl")
    assert os.path.isfile(data_path), (
        f"Data file not found: {data_path}. "
        f"Expected {template_name}_data.jsonl in templates directory."
    )

    with open(data_path) as f:
        lines = [line for line in f if line.strip()]

    assert len(lines) == 3, (
        f"Template '{template_name}' data has {len(lines)} rows, expected exactly 3. "
        f"Contract requires 3 sample rows per template. "
        f"File: {data_path}"
    )


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_template_data_is_valid_jsonl(template_name):
    """T016: Each data file is valid JSONL.

    Validates that every line in every data file parses as valid JSON
    and is a dict (JSON object).
    """
    data_path = os.path.join(TEMPLATE_DIR, f"{template_name}_data.jsonl")
    assert os.path.isfile(data_path), f"Data file not found: {data_path}"

    with open(data_path) as f:
        lines = [line for line in f if line.strip()]

    for i, line in enumerate(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Template '{template_name}' data line {i} is not valid JSON. "
                f"Error: {e}. "
                f"Line: {line[:100]}..."
            )

        assert isinstance(parsed, dict), (
            f"Template '{template_name}' data line {i} is not a JSON object. "
            f"Expected dict, got {type(parsed).__name__}. "
            f"JSONL format requires each line to be a JSON object."
        )
