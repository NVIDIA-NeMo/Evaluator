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

"""
Generate a minimal nemo-gym YAML config that points ``simple_agent`` at an
already-running ``NeMoEvaluatorResourcesServer`` and its auto-generated
JSONL dataset.

The generated config can be fed to ``ng_run`` or ``ng_collect_rollouts``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml


def generate_gym_config(
    resource_server_host: str,
    resource_server_port: int,
    data_dir: str,
    eval_type: str,
    model_server_url: str,
    model_id: str,
    model_server_name: str = "policy_model",
) -> dict[str, Any]:
    """Build a nemo-gym config dict suitable for ``ng_collect_rollouts``.

    The config references:
    - The already-running ``NeMoEvaluatorResourcesServer`` as a resource server.
    - A model server (the adapter / real endpoint) for the policy model.
    - A dataset JSONL that was written by the resource server at startup.
    """
    jsonl_fpath = str(Path(data_dir) / f"{eval_type}.jsonl")

    # The nemo-gym global config is keyed by "config path" at the top level.
    # Each entry has {server_type: {server_name: {actual config}}}.
    # See resources_servers/mcqa/configs/mcqa.yaml for the canonical format.
    return {
        "nemo_evaluator": {
            "resources_servers": {
                "nemo_evaluator": {
                    "entrypoint": "app.py",
                    "host": resource_server_host,
                    "port": resource_server_port,
                },
            },
        },
        "nemo_evaluator_simple_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "nemo_evaluator",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": model_server_name,
                    },
                    "datasets": [
                        {
                            "name": "eval",
                            "type": "eval",
                            "jsonl_fpath": jsonl_fpath,
                        },
                    ],
                },
            },
        },
        model_server_name: {
            "responses_api_models": {
                model_server_name: {
                    "entrypoint": "app.py",
                    "url": model_server_url,
                    "model": model_id,
                },
            },
        },
    }


def write_gym_config(
    config: dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Write the gym config dict to a YAML file.

    Returns the path to the written file.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".yaml", prefix="nemo_gym_eval_")
        os.close(fd)

    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    return output_path
