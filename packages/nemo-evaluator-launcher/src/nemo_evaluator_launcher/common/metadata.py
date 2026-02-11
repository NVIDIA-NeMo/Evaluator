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
"""Single source of truth for executor and deployment metadata.

This module provides:
- Dynamic discovery functions for executors and deployments
- CLI flag metadata for ls commands
- Config field metadata for validation
- Descriptions for user-facing output
"""
from __future__ import annotations

import pathlib
from functools import lru_cache


# --- Dynamic Discovery Functions ---


@lru_cache
def get_available_executors() -> list[str]:
    """Get executor names from _EXECUTOR_REGISTRY.

    This function imports the executors package which triggers the @register_executor
    decorators, populating the registry.
    """
    from nemo_evaluator_launcher.executors.registry import _EXECUTOR_REGISTRY

    # Import executors package to trigger @register_executor decorators
    import nemo_evaluator_launcher.executors  # noqa: F401

    return list(_EXECUTOR_REGISTRY.keys())


@lru_cache
def get_available_deployments() -> list[str]:
    """Get deployment names from configs/deployment/*.yaml files."""
    configs_dir = pathlib.Path(__file__).parent.parent / "configs" / "deployment"
    return sorted([f.stem for f in configs_dir.glob("*.yaml")])


# --- CLI Flag Metadata (for ls commands) ---

EXECUTOR_CLI_FLAGS: dict[str, dict[str, list[str]]] = {
    "local": {
        "required": ["--output-dir"],
        "optional": [],
    },
    "slurm": {
        "required": ["--slurm-hostname", "--slurm-account", "--output-dir"],
        "optional": ["--slurm-partition", "--slurm-walltime"],
    },
    "lepton": {
        "required": ["--output-dir"],
        "optional": [],
    },
}

DEPLOYMENT_CLI_FLAGS: dict[str, dict[str, list[str]]] = {
    "none": {
        "required": ["--model"],
        "optional": ["--url", "--api-key-env"],
    },
    "vllm": {
        "required": ["--checkpoint OR --hf-model", "--model-name"],
        "optional": ["--tp", "--dp"],
    },
    "sglang": {
        "required": ["--checkpoint OR --hf-model", "--model-name"],
        "optional": ["--tp", "--dp"],
    },
    "nim": {
        "required": ["--nim-model"],
        "optional": [],
    },
    "trtllm": {
        "required": ["--checkpoint", "--model-name"],
        "optional": ["--tp"],
    },
    "generic": {
        "required": ["--image"],
        "optional": [],
    },
}


# --- Config Field Metadata (for validation.py) ---

EXECUTOR_REQUIRED_CONFIG_FIELDS: dict[str, list[str]] = {
    "local": ["output_dir"],
    "slurm": ["hostname", "account", "output_dir"],
    "lepton": ["output_dir"],
}

DEPLOYMENT_REQUIRED_CONFIG_FIELDS: dict[str, list[str]] = {
    "none": [],  # API endpoint validation handled separately
    "vllm": ["checkpoint_path", "served_model_name"],
    "sglang": ["checkpoint_path", "served_model_name"],
    "nim": ["model_name"],
    "trtllm": ["checkpoint_path", "served_model_name"],
    "generic": ["image"],
}


# --- Descriptions ---

EXECUTOR_DESCRIPTIONS: dict[str, str] = {
    "local": "Run with Docker on this machine",
    "slurm": "Run on SLURM HPC cluster",
    "lepton": "Run on Lepton AI cloud",
}

DEPLOYMENT_DESCRIPTIONS: dict[str, str] = {
    "none": "Use existing API endpoint (no model deployment)",
    "vllm": "Deploy model with vLLM",
    "sglang": "Deploy model with SGLang",
    "nim": "Deploy NVIDIA NIM container",
    "trtllm": "Deploy model with TensorRT-LLM",
    "generic": "Deploy with custom container image",
}
