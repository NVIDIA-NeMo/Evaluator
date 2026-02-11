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
"""Build a complete NEL config by combining config template excerpts."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import List, Optional

import yaml


def _load_template(relative_path: str) -> dict:
    """Load a YAML config template from package resources."""
    content = (
        importlib.resources.files("nemo_evaluator_launcher.resources")
        .joinpath("config_templates", relative_path)
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(content) or {}


def resolve_output_path(
    output: Optional[Path],
    execution: str,
    deployment: str,
    model_type: str,
    benchmarks: List[str],
) -> Path:
    """Resolve the output file path, auto-generating a name if needed.

    Never overwrites — appends _1, _2, … when the file already exists.
    """
    filename = (
        f"{execution}_{deployment}_{model_type}_{'_'.join(sorted(benchmarks))}.yaml"
    )

    if output is not None and output.suffix in (".yaml", ".yml"):
        filepath = output
    else:
        directory = output or Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename

    # Find a path that doesn't already exist
    candidate = filepath
    counter = 1
    while candidate.exists():
        candidate = filepath.with_stem(f"{filepath.stem}_{counter}")
        counter += 1
    return candidate


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif (
            key in result and isinstance(result[key], list) and isinstance(value, list)
        ):
            # For lists (like tasks), extend rather than replace
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def build_config(
    execution: str,
    deployment: str,
    export: str,
    model_type: str,
    benchmarks: List[str],
    output: Optional[Path] = None,
) -> dict:
    """
    Build a complete NEL config by combining excerpts.

    Args:
        execution: Execution type (local, slurm)
        deployment: Deployment type (none, vllm, sglang, nim, trtllm)
        export: Export type (none, mlflow, wandb)
        model_type: Model type (base, chat, reasoning)
        benchmarks: List of benchmark types (standard, code, math_reasoning, safety)
        output: Optional output file path

    Returns:
        Combined config dictionary
    """
    config: dict = {}

    # 1. Load execution config (base execution settings)
    config = deep_merge(config, _load_template(f"execution/{execution}.yaml"))

    # 2. Load export config
    config = deep_merge(config, _load_template(f"export/{export}.yaml"))

    # 3. Load model type default config (sets default parallelism, temp, etc.)
    config = deep_merge(config, _load_template(f"evaluation/{model_type}/default.yaml"))

    # 4. Load benchmark configs
    for benchmark in benchmarks:
        template_path = f"evaluation/{model_type}/{benchmark}.yaml"
        try:
            config = deep_merge(config, _load_template(template_path))
        except FileNotFoundError:
            print(f"Warning: Benchmark config not found: {template_path}")

    # 5. Load deployment config (applied last so deployment-specific overrides take effect)
    # e.g., none.yaml sets parallelism: 1 for rate-limited external APIs
    config = deep_merge(config, _load_template(f"deployment/{deployment}.yaml"))

    # 6. Ensure _self_ is at the very end of defaults list (required by Hydra)
    if "defaults" in config:
        # Remove any existing _self_ entries
        config["defaults"] = [d for d in config["defaults"] if d != "_self_"]
        config["defaults"].append("_self_")
    else:
        config["defaults"] = ["_self_"]

    # 7. Write output if specified
    if output:
        with open(output, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Config written to: {output}")

    return config
