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

import pytest

from nemo_evaluator.core.input import parse_override_params
from nemo_evaluator.core.utils import dotlist_to_dict, get_api_key_from_env


def test_parse_override_params_basic():
    """Test basic functionality of parse_override_params"""
    input_str = "param1=true,param2=2,param3=3.14"
    result = parse_override_params(input_str)
    assert result == {"param1": True, "param2": 2, "param3": 3.14}


def test_parse_override_params_with_quoted_strings():
    """Test parse_override_params with strings containing commas"""
    input_str = "param1=true,param2=2,param3='some long, with commas, string'"
    result = parse_override_params(input_str)
    assert result == {
        "param1": True,
        "param2": 2,
        "param3": "some long, with commas, string",
    }


def test_parse_override_params_with_double_quotes():
    """Test parse_override_params with double quoted strings"""
    input_str = 'param1=true,param2=2,param3="some long, with commas, string"'
    result = parse_override_params(input_str)
    assert result == {
        "param1": True,
        "param2": 2,
        "param3": "some long, with commas, string",
    }


def test_parse_override_params_with_nested_quotes():
    """Test parse_override_params with nested quotes"""
    input_str = "param1=true,param2=2,param3=\"some 'nested' quotes\""
    result = parse_override_params(input_str)
    assert result == {"param1": True, "param2": 2, "param3": "some 'nested' quotes"}


def test_parse_override_params_empty():
    """Test parse_override_params with empty input"""
    assert parse_override_params(None) == {}
    assert parse_override_params("") == {}


def test_dotlist_to_dict_basic():
    """Test basic functionality of dotlist_to_dict"""
    input_list = ["param1=true", "param2=2", "param3=3.14"]
    result = dotlist_to_dict(input_list)
    assert result == {"param1": True, "param2": 2, "param3": 3.14}


def test_dotlist_to_dict_with_quoted_strings():
    """Test dotlist_to_dict with strings containing special characters"""
    input_list = ["param1=true", "param2=2", "param3='some long, with commas, string'"]
    result = dotlist_to_dict(input_list)
    assert result == {
        "param1": True,
        "param2": 2,
        "param3": "some long, with commas, string",
    }


def test_dotlist_to_dict_with_malformed_strings():
    """Test dotlist_to_dict with malformed strings"""
    input_list = ["param1=true", "param2=2", "param3='unclosed quote"]
    result = dotlist_to_dict(input_list)
    assert result == {"param1": True, "param2": 2, "param3": "'unclosed quote"}


def test_dotlist_to_dict_with_nested_keys():
    """Test dotlist_to_dict with nested keys"""
    input_list = ["parent.child1=value1", "parent.child2=value2"]
    result = dotlist_to_dict(input_list)
    assert result == {"parent": {"child1": "value1", "child2": "value2"}}


def test_dotlist_to_dict_with_complex_values():
    """Test dotlist_to_dict with complex YAML values"""
    input_list = ["param1=[1,2,3]", "param2={'key': 'value'}", "param3=null"]
    result = dotlist_to_dict(input_list)
    assert result == {"param1": [1, 2, 3], "param2": {"key": "value"}, "param3": None}


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "my-secret-key")
    assert get_api_key_from_env("TEST_API_KEY") == "my-secret-key"


def test_get_api_key_from_env_not_specified():
    """Test get_api_key_from_env when the variable name is not specified"""
    assert get_api_key_from_env(None) is None


def test_get_api_key_from_env_not_set(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISSING_API_KEY"):
        get_api_key_from_env("MISSING_API_KEY")
