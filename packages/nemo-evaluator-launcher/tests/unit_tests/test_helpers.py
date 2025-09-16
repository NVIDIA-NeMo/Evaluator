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

"""Tests for the helpers module."""

from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.helpers import (
    get_api_key_name,
    get_endpoint_url,
    get_eval_factory_dataset_size_from_run_config,
    get_health_url,
    get_served_model_name,
    get_timestamp_string,
)


def _cfg(obj: dict):
    return OmegaConf.create(obj)


def test_get_endpoint_url_none_uses_target_or_override():
    cfg = _cfg(
        {
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"url": "http://orig"}},
        }
    )
    user_task = _cfg(
        {"overrides": {"config.target.api_endpoint.url": "http://override"}}
    )
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    assert get_endpoint_url(cfg, user_task, task_def) == "http://override"


def test_get_endpoint_url_none_missing_raises_valueerror():
    cfg = _cfg(
        {"deployment": {"type": "none"}, "target": {"api_endpoint": {"url": "???"}}}
    )
    user_task = _cfg({})
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    import pytest

    with pytest.raises(ValueError, match="API endpoint URL is not set"):
        get_endpoint_url(cfg, user_task, task_def)


def test_get_endpoint_url_target_present_returns_target_with_override():
    cfg = _cfg(
        {
            "deployment": {
                "type": "vllm",
                "port": 8080,
                "endpoints": {"chat": "/v1/chat"},
            },
            "target": {"api_endpoint": {"url": "http://dyn"}},
        }
    )
    user_task = _cfg(
        {"overrides": {"config.target.api_endpoint.url": "http://dyn-ovr"}}
    )
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    assert get_endpoint_url(cfg, user_task, task_def) == "http://dyn-ovr"


def test_get_endpoint_url_local_builds_localhost():
    cfg = _cfg(
        {"deployment": {"type": "vllm", "port": 8081, "endpoints": {"chat": "/v1"}}}
    )
    user_task = _cfg({})
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    assert get_endpoint_url(cfg, user_task, task_def) == "http://127.0.0.1:8081/v1"


def test_get_health_url_none_returns_endpoint_url(monkeypatch):
    cfg = _cfg({"deployment": {"type": "none"}})
    assert get_health_url(cfg, "http://model.url") == "http://model.url"


def test_get_health_url_non_none_constructs_local():
    cfg = _cfg(
        {
            "deployment": {
                "type": "vllm",
                "port": 9000,
                "endpoints": {"health": "/health"},
            }
        }
    )
    assert get_health_url(cfg, "ignored") == "http://127.0.0.1:9000/health"


def test_get_served_model_name_branches():
    cfg_none = _cfg(
        {"deployment": {"type": "none"}, "target": {"api_endpoint": {"model_id": "m"}}}
    )
    assert get_served_model_name(cfg_none) == "m"
    cfg_other = _cfg({"deployment": {"type": "vllm", "served_model_name": "sv"}})
    assert get_served_model_name(cfg_other) == "sv"


def test_get_api_key_name_present_absent():
    cfg = _cfg({"target": {"api_endpoint": {"api_key_name": "API_KEY"}}})
    assert get_api_key_name(cfg) == "API_KEY"
    assert get_api_key_name(_cfg({})) is None


def test_get_timestamp_string_formats():
    ts1 = get_timestamp_string(True)
    ts2 = get_timestamp_string(False)
    assert ts1.count("_") == 2
    assert ts2.count("_") == 1
    assert len(ts1) > len(ts2)


def test_get_eval_factory_dataset_size_from_run_config_limit_samples():
    rc = {
        "framework_name": "simple_evals",
        "config": {"type": "mmlu", "params": {"limit_samples": 7}},
    }
    assert get_eval_factory_dataset_size_from_run_config(rc) == 7


def test_get_eval_factory_dataset_size_with_table_ratio_and_n_samples():
    rc = {
        "framework_name": "simple_evals",
        "config": {
            "type": "mmlu",
            "params": {"extra": {"downsampling_ratio": 0.5, "n_samples": 2}},
        },
    }
    # mmlu base 14042 -> 7021 with ratio 0.5 -> * n_samples 2 => 14042
    assert get_eval_factory_dataset_size_from_run_config(rc) == 14042


def test_get_eval_factory_dataset_size_unknown_returns_none():
    rc = {"framework_name": "unknown_fw", "config": {"type": "unknown", "params": {}}}
    assert get_eval_factory_dataset_size_from_run_config(rc) is None
