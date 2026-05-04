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
"""HuggingFace URI parsing in ``_load_hf``.

Locks down the path-segment-with-config behavior used by Sovereign
benchmarks (``hf://CohereForAI/Global-MMLU-Lite/en?split=test``) and the
filter-kwarg row dropping. We patch ``datasets.load_dataset`` so the tests
don't actually hit HuggingFace.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nemo_evaluator.environments.custom import _extract_filters, _load_hf


@pytest.fixture
def captured() -> dict:
    return {"calls": []}


@pytest.fixture
def fake_load_dataset(captured: dict):
    def _fake(*args, **kwargs):
        captured["calls"].append({"args": args, "kwargs": kwargs})
        return [{"row_id": 1, "text": "hello", "category": "x"}]

    with patch("datasets.load_dataset", side_effect=_fake) as mock:
        yield mock


def test_query_split_simple(fake_load_dataset, captured: dict) -> None:
    _load_hf("google/boolq?split=validation")
    call = captured["calls"][0]
    assert call["args"] == ("google/boolq",)
    assert call["kwargs"]["split"] == "validation"


def test_path_segment_config(fake_load_dataset, captured: dict) -> None:
    """``hf://ns/name/config?split=test`` — the config is the third path segment."""
    _load_hf("CohereForAI/Global-MMLU-Lite/en?split=test")
    call = captured["calls"][0]
    assert call["args"] == ("CohereForAI/Global-MMLU-Lite", "en")
    assert call["kwargs"]["split"] == "test"


def test_path_segment_config_and_split(fake_load_dataset, captured: dict) -> None:
    """``hf://ns/name/config/split`` — both from path segments."""
    _load_hf("CohereForAI/Global-MMLU-Lite/en/test")
    call = captured["calls"][0]
    assert call["args"] == ("CohereForAI/Global-MMLU-Lite", "en")
    assert call["kwargs"]["split"] == "test"


def test_query_overrides_path_split(fake_load_dataset, captured: dict) -> None:
    _load_hf("ns/name/cfg/train?split=test")
    call = captured["calls"][0]
    assert call["args"] == ("ns/name", "cfg")
    assert call["kwargs"]["split"] == "test"


def test_query_overrides_path_config(fake_load_dataset, captured: dict) -> None:
    _load_hf("ns/name/cfg?config=other&split=train")
    call = captured["calls"][0]
    assert call["args"] == ("ns/name", "other")
    assert call["kwargs"]["split"] == "train"


def test_num_examples_appended_to_split_when_no_slice(fake_load_dataset, captured: dict) -> None:
    _load_hf("ds?split=test", num_examples=5)
    assert captured["calls"][0]["kwargs"]["split"] == "test[:5]"


def test_num_examples_respects_existing_slice(fake_load_dataset, captured: dict) -> None:
    _load_hf("ds?split=test[:10]", num_examples=5)
    assert captured["calls"][0]["kwargs"]["split"] == "test[:10]"


def test_load_hf_filters_rows() -> None:
    rows = [{"category": "a", "n": 1}, {"category": "b", "n": 2}, {"category": "a", "n": 3}]
    with patch("datasets.load_dataset", return_value=rows):
        result = _load_hf("ds?split=test&filter_field=category&filter_value=a")
    assert [r["n"] for r in result] == [1, 3]


def test_load_hf_filters_with_numeric_suffix() -> None:
    rows = [
        {"category": "a", "lang": "en"},
        {"category": "a", "lang": "fr"},
        {"category": "b", "lang": "en"},
    ]
    with patch("datasets.load_dataset", return_value=rows):
        result = _load_hf(
            "ds?split=test&filter_field=category&filter_value=a&filter_field_1=lang&filter_value_1=en"
        )
    assert len(result) == 1
    assert result[0]["category"] == "a"
    assert result[0]["lang"] == "en"


def test_extract_filters_handles_no_filter() -> None:
    assert _extract_filters({"split": "test"}) == []


def test_extract_filters_pairs_and_suffix() -> None:
    params = {
        "filter_field": "x",
        "filter_value": "1",
        "filter_field_2": "y",
        "filter_value_2": "2",
    }
    assert _extract_filters(params) == [("x", "1"), ("y", "2")]
