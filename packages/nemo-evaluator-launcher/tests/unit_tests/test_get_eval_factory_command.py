# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import re

import yaml
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.helpers import get_eval_factory_command


def _extract_b64_from_echo_cmd(cmd: str) -> str:
    """Extract the base64 segment for the eval-factory config file.

    Supports commands with multiple echo|base64 segments. Prefer the one
    writing to either config_ef.yaml or ef_config.yaml; otherwise return
    the last occurrence.
    """
    matches = re.findall(r'echo "([^"]+)" \| base64 -d > ([^\s;&]+)', cmd)
    if not matches:
        raise ValueError("No base64 echo segments found in command")
    preferred_targets = {"ef_config.yaml"}
    for b64, target in matches:
        if target in preferred_targets:
            return b64
    return matches[-1][0]


def test_get_eval_factory_command_basic(monkeypatch):
    # Make versions deterministic
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.helpers.get_versions", lambda: "TEST_VER"
    )

    cfg = OmegaConf.create(
        {
            "evaluation": {
                # Use new-style config key
                "nemo_evaluator_config": {"config": {"foo": "bar\n"}},
            },
            "deployment": {
                "type": "none",
            },
            "target": {
                "api_endpoint": {
                    "url": "https://example.test/api",
                    "model_id": "model-123",
                }
            },
        }
    )

    user_task_config = OmegaConf.create(
        {
            # Old key should also be supported; keep empty so helper sets fields
            "config": {},
            "nemo_evaluator_config": {"config": {"baz": 42}},
        }
    )

    task_definition = {"endpoint_type": "chat", "task": "my_task"}

    result = get_eval_factory_command(cfg, user_task_config, task_definition)

    # Validate debug text
    assert result.debug.startswith("# Contents of")

    # Extract and decode YAML from the command
    b64 = _extract_b64_from_echo_cmd(result.cmd)
    decoded_yaml = base64.b64decode(b64.encode("utf-8")).decode("utf-8")
    merged = yaml.safe_load(decoded_yaml)

    # Core fields enriched by helper
    assert merged["target"]["api_endpoint"]["url"] == "https://example.test/api"
    assert merged["target"]["api_endpoint"]["model_id"] == "model-123"
    assert merged["target"]["api_endpoint"]["type"] == "chat"
    assert merged["target"]["api_endpoint"]["api_key"] == "API_KEY"

    assert merged["config"]["type"] == "my_task"
    assert merged["config"]["output_dir"] == "/results"

    # Metadata is populated, including resolved launcher config and versioning
    assert "metadata" in merged and isinstance(merged["metadata"], dict)
    assert merged["metadata"]["versioning"] == "TEST_VER"
    # Validate a few salient fields from the resolved config got embedded
    resolved = merged["metadata"]["launcher_resolved_config"]
    assert resolved["deployment"]["type"] == "none"
    assert resolved["target"]["api_endpoint"]["url"] == "https://example.test/api"
    assert resolved["target"]["api_endpoint"]["model_id"] == "model-123"
    assert resolved["evaluation"]["nemo_evaluator_config"]["config"] == {"foo": "bar\n"}

    # The command to run eval is present
    assert "&& $cmd run_eval --run_config config_ef.yaml" in result.cmd


def test_get_eval_factory_command_injects_sidecars_without_overriding(monkeypatch):
    monkeypatch.setattr(
        "nemo_evaluator_launcher.common.helpers.get_versions", lambda: "TEST_VER"
    )

    cfg = OmegaConf.create(
        {
            "evaluation": {
                "nemo_evaluator_config": {"config": {"params": {"extra": {}}}}
            },
            "deployment": {"type": "none"},
            "target": {
                "api_endpoint": {
                    "url": "https://example.test/api",
                    "model_id": "model-123",
                }
            },
        }
    )
    # User pre-defines agent_2.url -> launcher should NOT override it.
    user_task_config = OmegaConf.create(
        {
            "nemo_evaluator_config": {
                "config": {
                    "params": {
                        "extra": {
                            "agent_2": {"url": "http://predeployed:9999", "port": 9999}
                        }
                    }
                }
            }
        }
    )

    task_definition = {"endpoint_type": "chat", "task": "my_task"}
    result = get_eval_factory_command(
        cfg,
        user_task_config,
        task_definition,
        launcher_sidecars={
            "agent_1": {"url": "http://127.0.0.1:8080", "port": 8080},
            "agent_2": {"url": "http://127.0.0.1:9000", "port": 9000},
        },
    )

    b64 = _extract_b64_from_echo_cmd(result.cmd)
    decoded_yaml = base64.b64decode(b64.encode("utf-8")).decode("utf-8")
    merged = yaml.safe_load(decoded_yaml)

    extra = merged["config"]["params"]["extra"]
    assert extra["agent_1"]["url"] == "http://127.0.0.1:8080"
    assert extra["agent_1"]["port"] == 8080

    # Not overridden:
    assert extra["agent_2"]["url"] == "http://predeployed:9999"
    assert extra["agent_2"]["port"] == 9999
