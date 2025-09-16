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


from nemo_evaluator.core.input import merge_dicts


def test_distinct_keys():
    d1 = {"a": 1}
    d2 = {"b": 2}
    assert merge_dicts(d1, d2) == {"a": 1, "b": 2}


def test_common_key_non_lists():
    d1 = {"a": 1}
    d2 = {"a": 2}
    assert merge_dicts(d1, d2) == {"a": [1, 2]}


def test_value_is_list():
    d1 = {"a": [1]}
    d2 = {"a": 2}
    assert merge_dicts(d1, d2) == {"a": [1, 2]}


def test_both_values_are_lists():
    d1 = {"a": 1}
    d2 = {"a": [2, 3]}
    assert merge_dicts(d1, d2) == {"a": [1, 2, 3]}


def test_lists_and_nonlists_mixed():
    d1 = {"a": [1, 2]}
    d2 = {"a": 3}
    assert merge_dicts(d1, d2) == {"a": [1, 2, 3]}
    d1 = {"a": 1}
    d2 = {"a": [2, 3]}
    assert merge_dicts(d1, d2) == {"a": [1, 2, 3]}


def test_multiple_keys_various_types():
    d1 = {"a": 1, "b": [2, 3], "c": 4}
    d2 = {"b": 5, "c": [6], "d": 7}
    assert merge_dicts(d1, d2) == {"a": 1, "b": [2, 3, 5], "c": [4, 6], "d": 7}


def test_empty_dicts():
    d1 = {}
    d2 = {"a": 1}
    assert merge_dicts(d1, d2) == {"a": 1}
    d1 = {"b": 2}
    d2 = {}
    assert merge_dicts(d1, d2) == {"b": 2}
    assert merge_dicts({}, {}) == {}
