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


def _template_exists(relative_path: str) -> bool:
    """Check whether a config template exists in package resources."""
    resource = importlib.resources.files("nemo_evaluator_launcher.resources").joinpath(
        "config_templates", relative_path
    )
    try:
        resource.read_text(encoding="utf-8")
        return True
    except (FileNotFoundError, TypeError):
        return False


def generate_config_filename(
    execution: str,
    deployment: str,
    model_type: str,
    benchmarks: List[str],
) -> str:
    """Generate a config filename from the combination of choices."""
    bench_str = "_".join(sorted(benchmarks))
    return f"{execution}_{deployment}_{model_type}_{bench_str}.yaml"


def get_unique_filepath(filepath: Path) -> Path:
    """Return a unique filepath by adding numeric suffix if file exists."""
    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    counter = 1

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def resolve_output_path(
    output: Optional[Path],
    execution: str,
    deployment: str,
    model_type: str,
    benchmarks: List[str],
) -> Path:
    """
    Resolve the output path based on what was provided.

    Args:
        output: User-provided output path (file, directory, or None)
        execution, deployment, model_type, benchmarks: Config choices for filename

    Returns:
        Resolved output file path (unique, won't overwrite existing)
    """
    filename = generate_config_filename(execution, deployment, model_type, benchmarks)

    if output is None:
        # No output specified: use current directory with auto-generated filename
        filepath = Path.cwd() / filename
    elif output.suffix in (".yaml", ".yml"):
        # Looks like a file path: use as-is
        filepath = output
    elif output.is_dir() or not output.suffix:
        # Directory path: auto-generate filename in that directory
        output.mkdir(parents=True, exist_ok=True)
        filepath = output / filename
    else:
        # Assume it's a file path
        filepath = output

    return get_unique_filepath(filepath)


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
        if _template_exists(template_path):
            config = deep_merge(config, _load_template(template_path))
        else:
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
