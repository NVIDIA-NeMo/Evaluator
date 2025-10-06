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

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    EndpointType,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.input import check_type_compatibility, merge_dicts
from nemo_evaluator.core.utils import MisconfigurationError


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


@pytest.mark.parametrize(
    "model_types,benchmark_types",
    [
        (EndpointType.CHAT, EndpointType.CHAT),
        ([EndpointType.CHAT], [EndpointType.CHAT]),
        (EndpointType.CHAT, [EndpointType.CHAT]),
        ([EndpointType.CHAT], EndpointType.CHAT),
        ("chat", "chat"),
        ("chat", None),
        ([EndpointType.CHAT, EndpointType.COMPLETIONS], [EndpointType.CHAT]),
        ([EndpointType.CHAT, EndpointType.COMPLETIONS], EndpointType.CHAT),
        (EndpointType.CHAT, [[EndpointType.CHAT], [EndpointType.COMPLETIONS]]),
        (
            [EndpointType.CHAT, EndpointType.COMPLETIONS],
            [EndpointType.CHAT, EndpointType.COMPLETIONS],
        ),
        (
            [EndpointType.CHAT, EndpointType.COMPLETIONS, EndpointType.VLM],
            [EndpointType.CHAT, EndpointType.COMPLETIONS],
        ),
        (
            [EndpointType.CHAT, EndpointType.VLM],
            [
                [EndpointType.COMPLETIONS, EndpointType.VLM],
                [EndpointType.CHAT, EndpointType.VLM],
            ],
        ),
    ],
)
def test_endpoint_type_single_compatible(model_types, benchmark_types):
    evaluation_config = EvaluationConfig(supported_endpoint_types=benchmark_types)
    target_config = EvaluationTarget(
        api_endpoint=ApiEndpoint(type=model_types, url="localhost", model_id="my_model")
    )
    evaluation = Evaluation(
        config=evaluation_config,
        target=target_config,
        command="",
        pkg_name="",
        framework_name="",
    )
    check_type_compatibility(evaluation)


@pytest.mark.parametrize(
    "model_types,benchmark_types",
    [
        (EndpointType.CHAT, EndpointType.COMPLETIONS),
        ("chat", "vlm"),
        ([EndpointType.CHAT], [[EndpointType.CHAT, EndpointType.VLM]]),
        (
            [EndpointType.CHAT, EndpointType.VLM],
            [[EndpointType.COMPLETIONS, EndpointType.VLM]],
        ),
    ],
)
def test_endpoint_type_single_incompatible(model_types, benchmark_types):
    evaluation_config = EvaluationConfig(supported_endpoint_types=benchmark_types)
    target_config = EvaluationTarget(
        api_endpoint=ApiEndpoint(type=model_types, url="localhost", model_id="my_model")
    )
    evaluation = Evaluation(
        config=evaluation_config,
        target=target_config,
        command="",
        pkg_name="",
        framework_name="",
    )
    with pytest.raises(
        MisconfigurationError, match=r".* does not support the model type .*"
    ):
        check_type_compatibility(evaluation)
