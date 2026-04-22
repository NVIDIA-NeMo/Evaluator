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
"""Tests for JSON schema validation scoring."""

from nemo_evaluator.scoring.json_schema import extract_json, validate_json_schema


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"key": "value"}') == {"key": "value"}

    def test_markdown_block(self):
        text = 'Here is the output:\n```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'The answer is {"result": 42} as expected.'
        assert extract_json(text) == {"result": 42}

    def test_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_no_json(self):
        assert extract_json("no json here") is None

    def test_invalid_json(self):
        assert extract_json("{broken json") is None


class TestValidateJsonSchema:
    def test_valid_object(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = validate_json_schema('{"name": "Alice", "age": 30}', schema)
        assert result["valid"]
        assert result["score"] == 1.0

    def test_missing_required(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        result = validate_json_schema('{"name": "Alice"}', schema)
        assert not result["valid"]
        assert result["score"] == 0.0
        assert any("age" in e for e in result["errors"])

    def test_wrong_type(self):
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        result = validate_json_schema('{"count": "five"}', schema)
        assert not result["valid"]

    def test_array_validation(self):
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 2}
        result = validate_json_schema("[1, 2, 3]", schema)
        assert result["valid"]

    def test_array_too_short(self):
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 5}
        result = validate_json_schema("[1, 2]", schema)
        assert not result["valid"]

    def test_enum(self):
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        assert validate_json_schema('"red"', schema)["valid"]
        assert not validate_json_schema('"yellow"', schema)["valid"]

    def test_unparseable_response(self):
        result = validate_json_schema("no json here", {"type": "object"})
        assert not result["valid"]
        assert result["extracted"] is None
