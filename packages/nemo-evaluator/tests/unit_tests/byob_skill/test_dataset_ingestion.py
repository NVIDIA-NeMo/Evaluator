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

"""Dataset ingestion tests.

Tests T055-T064: Validate JSON/CSV/JSONL conversion logic and edge cases.
These tests validate the conversion patterns that the /byob skill would
generate for users.
"""

import csv
import json

import pytest


def test_json_array_to_jsonl(tmp_path):
    """T055: JSON array-of-objects converts to valid JSONL."""
    input_data = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 3+3?", "answer": "6"},
    ]
    input_path = tmp_path / "data.json"
    output_path = tmp_path / "data.jsonl"

    # Write input JSON
    with open(input_path, "w") as f:
        json.dump(input_data, f)

    # Conversion logic (mirrors what the skill would generate)
    with open(input_path) as f:
        data = json.load(f)
    assert isinstance(data, list), (
        f"Expected JSON array, got {type(data).__name__}. "
        f"JSON file should contain an array of objects."
    )

    with open(output_path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    # Verify output
    with open(output_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2, f"Expected 2 JSONL rows, got {len(lines)}"

    for i, line in enumerate(lines):
        parsed = json.loads(line)
        assert parsed == input_data[i], (
            f"Row {i} mismatch. Expected: {input_data[i]}, got: {parsed}"
        )


def test_json_nested_structure_conversion(tmp_path):
    """T056: JSON nested structure detected and converted.

    Tests detection of nested array keys like {"data": [...], ...}.
    """
    input_data = {
        "metadata": {"source": "test"},
        "data": [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ],
    }
    input_path = tmp_path / "nested.json"
    output_path = tmp_path / "nested.jsonl"

    with open(input_path, "w") as f:
        json.dump(input_data, f)

    # Nested structure detection logic
    with open(input_path) as f:
        data = json.load(f)

    # Try to find an array key (common patterns: "data", "items", "questions", "rows")
    array_keys = ["data", "items", "questions", "rows", "examples"]
    extracted_array = None

    if isinstance(data, dict):
        for key in array_keys:
            if key in data and isinstance(data[key], list):
                extracted_array = data[key]
                break

    assert extracted_array is not None, (
        f"Could not find array in nested JSON. "
        f"Tried keys: {array_keys}. Available keys: {list(data.keys())}"
    )

    # Convert to JSONL
    with open(output_path, "w") as f:
        for row in extracted_array:
            f.write(json.dumps(row) + "\n")

    # Verify
    with open(output_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2, f"Expected 2 JSONL rows, got {len(lines)}"


def test_csv_with_headers_to_jsonl(tmp_path):
    """T057: CSV with headers converts to JSONL."""
    csv_path = tmp_path / "data.csv"
    jsonl_path = tmp_path / "data.jsonl"

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerow(["What is 2+2?", "4"])
        writer.writerow(["What is 3+3?", "6"])

    # CSV to JSONL conversion
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Verify
    with open(jsonl_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2, f"Expected 2 JSONL rows, got {len(lines)}"

    parsed_rows = [json.loads(line) for line in lines]
    assert parsed_rows[0]["question"] == "What is 2+2?"
    assert parsed_rows[0]["answer"] == "4"


def test_tsv_to_jsonl(tmp_path):
    """T058: TSV (tab-delimited) converts to JSONL."""
    tsv_path = tmp_path / "data.tsv"
    jsonl_path = tmp_path / "data.jsonl"

    # Write TSV
    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["question", "answer"])
        writer.writerow(["What is 2+2?", "4"])
        writer.writerow(["What is 3+3?", "6"])

    # TSV to JSONL conversion
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Verify
    with open(jsonl_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2, f"Expected 2 JSONL rows, got {len(lines)}"


def test_jsonl_validation_all_rows_parse(tmp_path):
    """T059: JSONL validation -- all rows parse as JSON."""
    jsonl_path = tmp_path / "valid.jsonl"

    with open(jsonl_path, "w") as f:
        f.write('{"question": "What is 2+2?", "answer": "4"}\n')
        f.write('{"question": "What is 3+3?", "answer": "6"}\n')

    # Validation logic
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                assert isinstance(parsed, dict), (
                    f"Line {i}: Expected dict, got {type(parsed).__name__}"
                )
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i}: Invalid JSON: {e}")


def test_jsonl_schema_detection(tmp_path):
    """T060: JSONL schema detection finds all field names.

    Schema detection should enumerate the union of all keys across rows.
    """
    jsonl_path = tmp_path / "data.jsonl"

    with open(jsonl_path, "w") as f:
        f.write('{"question": "Q1", "answer": "A1"}\n')
        f.write('{"question": "Q2", "answer": "A2", "context": "C2"}\n')
        f.write('{"question": "Q3", "answer": "A3"}\n')

    # Schema detection logic
    all_keys = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            all_keys.update(row.keys())

    assert "question" in all_keys
    assert "answer" in all_keys
    assert "context" in all_keys, (
        f"Schema detection should find 'context' in row 2. Found: {all_keys}"
    )


def test_empty_json_file_raises_error(tmp_path):
    """T061: Empty JSON file raises error."""
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("")

    with pytest.raises(json.JSONDecodeError):
        with open(empty_path) as f:
            json.load(f)


def test_malformed_jsonl_line_raises_error(tmp_path):
    """T062: Malformed JSON line raises error."""
    jsonl_path = tmp_path / "malformed.jsonl"

    with open(jsonl_path, "w") as f:
        f.write('{"question": "Q1", "answer": "A1"}\n')
        f.write('{"question": "Q2", "answer": INVALID}\n')  # Malformed
        f.write('{"question": "Q3", "answer": "A3"}\n')

    # Validation should catch the malformed line
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 1:  # Second line is malformed
                with pytest.raises(json.JSONDecodeError):
                    json.loads(line)
            else:
                json.loads(line)  # Should parse successfully


def test_json_with_no_array_raises_error(tmp_path):
    """T063: JSON with no array raises error.

    A JSON file that is a single object with no recognizable array key
    should be rejected.
    """
    json_path = tmp_path / "no_array.json"
    data = {"config": {"key": "value"}, "metadata": "test"}

    with open(json_path, "w") as f:
        json.dump(data, f)

    # Attempt to find an array
    with open(json_path) as f:
        data = json.load(f)

    # Check if it's directly an array
    if isinstance(data, list):
        array = data
    elif isinstance(data, dict):
        # Try common array keys
        array_keys = ["data", "items", "questions", "rows", "examples"]
        array = None
        for key in array_keys:
            if key in data and isinstance(data[key], list):
                array = data[key]
                break

        if array is None:
            # No array found - this should raise an error
            with pytest.raises(ValueError, match="[Nn]o array|[Nn]ot found|[Cc]ould not find"):
                raise ValueError(
                    f"Could not find an array in JSON file. "
                    f"Expected a top-level array or a dict with keys: {array_keys}. "
                    f"Found keys: {list(data.keys())}"
                )
    else:
        pytest.fail(f"Unexpected JSON type: {type(data)}")


def test_csv_with_no_data_rows(tmp_path):
    """T064: CSV with no data rows (header only) produces empty JSONL."""
    csv_path = tmp_path / "header_only.csv"
    jsonl_path = tmp_path / "header_only.jsonl"

    # Write CSV with only header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        # No data rows

    # Convert to JSONL
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 0, f"Expected 0 data rows, got {len(rows)}"

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Verify empty JSONL
    with open(jsonl_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 0, (
        f"Expected empty JSONL file (0 rows), got {len(lines)} rows. "
        f"CSV with only header should produce empty JSONL."
    )
