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

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.helpers import (
    get_api_key_name,
    get_endpoint_url,
    get_eval_factory_config,
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
            "evaluation": {},
        }
    )
    user_task = _cfg(
        {
            "nemo_evaluator_config": {
                "target": {"api_endpoint": {"url": "http://override"}}
            }
        }
    )
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    merged_config = get_eval_factory_config(cfg, user_task)
    assert (
        get_endpoint_url(cfg, merged_config, task_def["endpoint_type"])
        == "http://override"
    )


def test_get_endpoint_url_none_missing_raises_valueerror():
    cfg = _cfg(
        {
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"url": "???"}},
            "evaluation": {},
        }
    )
    user_task = _cfg({})
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    import pytest

    with pytest.raises(ValueError, match="API endpoint URL is not set"):
        merged_config = get_eval_factory_config(cfg, user_task)
        get_endpoint_url(cfg, merged_config, task_def["endpoint_type"])


def test_get_endpoint_url_target_present_returns_target_with_override():
    cfg = _cfg(
        {
            "deployment": {
                "type": "vllm",
                "port": 8080,
                "endpoints": {"chat": "/v1/chat"},
            },
            "target": {"api_endpoint": {"url": "http://dyn"}},
            "evaluation": {},
        }
    )
    user_task = _cfg(
        {
            "nemo_evaluator_config": {
                "target": {"api_endpoint": {"url": "http://dyn-ovr"}}
            }
        }
    )
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    merged_config = get_eval_factory_config(cfg, user_task)
    assert (
        get_endpoint_url(cfg, merged_config, task_def["endpoint_type"])
        == "http://dyn-ovr"
    )


def test_get_endpoint_url_local_builds_localhost():
    cfg = _cfg(
        {
            "deployment": {"type": "vllm", "port": 8081, "endpoints": {"chat": "/v1"}},
            "evaluation": {},
        }
    )
    user_task = _cfg({})
    task_def = {"endpoint_type": "chat", "task": "simple_evals.mmlu"}
    merged_config = get_eval_factory_config(cfg, user_task)
    assert (
        get_endpoint_url(cfg, merged_config, task_def["endpoint_type"])
        == "http://127.0.0.1:8081/v1"
    )


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


# Common configuration fixtures
@pytest.fixture
def global_config():
    """Common global configuration with nemo_evaluator_config."""
    return {
        "evaluation": {
            "nemo_evaluator_config": {
                "config": {
                    "params": {
                        "request_timeout": 3600,
                        "parallelism": 1,
                        "temperature": 0.0,  # Global default
                        "top_p": 0.9,
                    },
                    "target": {
                        "api_endpoint": {
                            "adapter_config": {
                                "process_reasoning_traces": False,
                                "use_system_prompt": True,
                                "custom_system_prompt": "Think step by step.",
                            }
                        }
                    },
                }
            }
        }
    }


@pytest.fixture
def task_config():
    """Task configuration."""
    return {
        "nemo_evaluator_config": {
            "config": {
                "params": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 8192}
            }
        }
    }


@pytest.fixture
def old_format_global_config():
    """Global config using old 'config' format."""
    return {
        "evaluation": {
            "config": {"params": {"request_timeout": 3600, "parallelism": 1}}
        }
    }


@pytest.fixture
def old_format_task_config():
    """Task config using old 'config' format."""
    return {"config": {"params": {"temperature": 0.6}}}


def test_start_deprecating_overrides():
    """This test will start failing to remind of removing overrides"""
    # What to do: remove all the respect of `overrides` in code.
    from datetime import datetime

    DEPRECATION_DATE = datetime(2025, 12, 1)
    if datetime.now() > DEPRECATION_DATE:
        pytest.fail(f"Deprectation of overrides should start {DEPRECATION_DATE}")


def test_get_eval_factory_config_global_config_only(global_config):
    """Test with only global nemo_evaluator_config."""
    cfg = _cfg(global_config)
    user_task_config = _cfg({})

    result = get_eval_factory_config(cfg, user_task_config)

    expected = {
        "config": {
            "params": {
                "request_timeout": 3600,
                "parallelism": 1,
                "temperature": 0.0,
                "top_p": 0.9,
            },
            "target": {
                "api_endpoint": {
                    "adapter_config": {
                        "process_reasoning_traces": False,
                        "use_system_prompt": True,
                        "custom_system_prompt": "Think step by step.",
                    }
                }
            },
        }
    }

    assert result == expected


def test_get_eval_factory_config_task_config_only():
    """Test with only task-specific nemo_evaluator_config."""
    cfg = _cfg({"evaluation": {}})
    user_task_config = _cfg(
        {
            "nemo_evaluator_config": {
                "config": {
                    "params": {
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "max_new_tokens": 8192,
                    }
                }
            }
        }
    )

    result = get_eval_factory_config(cfg, user_task_config)

    expected = {
        "config": {
            "params": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 8192}
        }
    }

    assert result == expected


def test_get_eval_factory_config_mixed_old_and_new_format(
    global_config, old_format_task_config
):
    """Test mixing old and new config formats."""
    cfg = _cfg(global_config)
    user_task_config = _cfg(old_format_task_config)

    result = get_eval_factory_config(cfg, user_task_config)

    # The function merges the configs, but the old format doesn't have the "config" wrapper
    # So the result will have both the "config" from global and the direct "params" from task
    expected = {
        "config": {
            "params": {
                "request_timeout": 3600,  # From global
                "parallelism": 1,  # From global
                "temperature": 0.0,  # From global
                "top_p": 0.9,  # From global
            },
            "target": {
                "api_endpoint": {
                    "adapter_config": {
                        "process_reasoning_traces": False,
                        "use_system_prompt": True,
                        "custom_system_prompt": "Think step by step.",
                    }
                }
            },
        },
        "params": {  # From task (old format)
            "temperature": 0.6
        },
    }

    assert result == expected


def test_get_eval_factory_config_empty_configs():
    """Test with empty configs."""
    cfg = _cfg({"evaluation": {}})
    user_task_config = _cfg({})

    result = get_eval_factory_config(cfg, user_task_config)

    assert result == {}


def test_get_eval_factory_config_complex_real_world_scenario(
    global_config, task_config
):
    """Test with a complex real-world scenario similar to the example config."""
    cfg = _cfg(global_config)
    user_task_config = _cfg(task_config)

    result = get_eval_factory_config(cfg, user_task_config)

    expected = {
        "config": {
            "params": {
                "request_timeout": 3600,  # From global
                "parallelism": 1,  # From global
                "temperature": 0.6,  # From task
                "top_p": 0.95,  # From task
                "max_new_tokens": 8192,  # From task
            },
            "target": {
                "api_endpoint": {
                    "adapter_config": {
                        "process_reasoning_traces": False,  # From global
                        "use_system_prompt": True,  # From global
                        "custom_system_prompt": "Think step by step.",  # From global
                    }
                }
            },
        }
    }

    assert result == expected


def test_get_eval_factory_command_pre_cmd_task_overrides_global(monkeypatch):
    from nemo_evaluator_launcher.common.helpers import get_eval_factory_command

    # Trust pre_cmd for the purpose of this unit test
    monkeypatch.setenv("NEMO_EVALUATOR_TRUST_PRE_CMD", "1")

    # Build a minimal config where endpoint URL resolution is simple
    cfg = _cfg(
        {
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"url": "http://example/v1", "model_id": "m"}},
            "evaluation": {
                "pre_cmd": "export GLOBAL_X=1",
                "nemo_evaluator_config": {"config": {"params": {}}},
            },
        }
    )
    user_task_config = _cfg(
        {
            "name": "some_task",
            "pre_cmd": "export TASK_Y=2",
            "nemo_evaluator_config": {"config": {"params": {}}},
        }
    )
    task_definition = {"endpoint_type": "chat", "task": "some_task"}

    cmd_and_dbg = get_eval_factory_command(cfg, user_task_config, task_definition)

    # pre_cmd is written into pre_cmd.sh and sourced
    assert "source pre_cmd.sh" in cmd_and_dbg.cmd

    # Task-level pre_cmd should win over global
    import base64

    expected_b64 = base64.b64encode(b"export TASK_Y=2").decode("utf-8")
    assert f'echo "{expected_b64}" | base64 -d > pre_cmd.sh' in cmd_and_dbg.cmd

    # Debug string should include human-readable contents
    assert "# Contents of pre_cmd.sh" in cmd_and_dbg.debug
    assert "# export TASK_Y=2" in cmd_and_dbg.debug

    # Command should invoke eval factory
    assert " run_eval --run_config config_ef.yaml" in cmd_and_dbg.cmd


def test_get_eval_factory_command_pre_cmd_from_global_when_task_absent(monkeypatch):
    from nemo_evaluator_launcher.common.helpers import get_eval_factory_command

    # Trust pre_cmd for the purpose of this unit test
    monkeypatch.setenv("NEMO_EVALUATOR_TRUST_PRE_CMD", "1")

    cfg = _cfg(
        {
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"url": "http://example/v1", "model_id": "m"}},
            "evaluation": {
                "pre_cmd": "echo hello",
                "nemo_evaluator_config": {"config": {"params": {}}},
            },
        }
    )
    user_task_config = _cfg({"name": "some_task"})
    task_definition = {"endpoint_type": "chat", "task": "some_task"}

    cmd_and_dbg = get_eval_factory_command(cfg, user_task_config, task_definition)

    import base64

    expected_b64 = base64.b64encode(b"echo hello").decode("utf-8")
    assert f'echo "{expected_b64}" | base64 -d > pre_cmd.sh' in cmd_and_dbg.cmd
    assert "# echo hello" in cmd_and_dbg.debug


def test_get_eval_factory_command_pre_cmd_empty_when_not_provided():
    from nemo_evaluator_launcher.common.helpers import get_eval_factory_command

    cfg = _cfg(
        {
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {"url": "http://example/v1", "model_id": "m"}},
            "evaluation": {"nemo_evaluator_config": {"config": {"params": {}}}},
        }
    )
    user_task_config = _cfg({"name": "some_task"})
    task_definition = {"endpoint_type": "chat", "task": "some_task"}

    cmd_and_dbg = get_eval_factory_command(cfg, user_task_config, task_definition)

    # Even with empty pre_cmd, the script is still created and sourced
    assert 'echo "" | base64 -d > pre_cmd.sh' in cmd_and_dbg.cmd
    assert "source pre_cmd.sh" in cmd_and_dbg.cmd
