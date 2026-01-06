# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Extended tests for the mapping module (IR-based mapping)."""

import pathlib

import pytest

from nemo_evaluator_launcher.common.container_metadata.intermediate_repr import (
    TaskIntermediateRepresentation,
)
from nemo_evaluator_launcher.common.mapping import (
    _convert_irs_to_mapping_format,
    load_tasks_mapping,
)


def test_load_tasks_mapping_mapping_toml_is_rejected(tmp_path: pathlib.Path):
    mapping_file = tmp_path / "custom_mapping.toml"
    mapping_file.write_text("[custom]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping_toml is no longer supported"):
        _ = load_tasks_mapping(mapping_toml=mapping_file)


def test_convert_irs_to_mapping_format_selects_first_supported_endpoint_type():
    tasks = [
        TaskIntermediateRepresentation(
            name="task_a",
            description="desc",
            harness="harness_x",
            container="example.com/x:latest",
            container_digest=None,
            defaults={"config": {"supported_endpoint_types": ["completions", "chat"]}},
        )
    ]

    mapping = _convert_irs_to_mapping_format(tasks)
    assert mapping[("harness_x", "task_a")]["endpoint_type"] == "completions"


def test_convert_irs_to_mapping_format_duplicate_key_keeps_first():
    tasks = [
        TaskIntermediateRepresentation(
            name="task_a",
            description="desc1",
            harness="harness_x",
            container="img:first",
            container_digest=None,
            defaults={"config": {"supported_endpoint_types": ["chat"]}},
        ),
        TaskIntermediateRepresentation(
            name="task_a",
            description="desc2",
            harness="harness_x",
            container="img:second",
            container_digest=None,
            defaults={"config": {"supported_endpoint_types": ["completions"]}},
        ),
    ]

    mapping = _convert_irs_to_mapping_format(tasks)
    assert mapping[("harness_x", "task_a")]["container"] == "img:first"
    assert mapping[("harness_x", "task_a")]["endpoint_type"] == "chat"
