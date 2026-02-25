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

"""BYOB compiler: transforms user benchmark modules into core_evals namespace packages."""

import importlib
import os
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from nemo_evaluator.contrib.byob.decorators import (
    BenchmarkDefinition,
    clear_registry,
    get_registered_benchmarks,
)
from nemo_evaluator.contrib.byob.defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
)
from nemo_evaluator.logging import get_logger

_logger = get_logger(__name__)


# Jinja2 command template for runner invocation
# NOTE: Use plain string concatenation to avoid f-string escaping issues with {{ }}
COMMAND_TEMPLATE = (
    "python -m nemo_evaluator.contrib.byob.runner"
    " --benchmark-module {{config.params.extra.benchmark_module}}"
    " --benchmark-name {{config.params.extra.benchmark_name}}"
    " --dataset {{config.params.extra.dataset}}"
    " --output-dir {{config.output_dir}}"
    " --model-url {{target.api_endpoint.url}}"
    " --model-id {{target.api_endpoint.model_id}}"
    " --model-type {{target.api_endpoint.type}}"
    " --temperature {{config.params.temperature}}"
    " --max-tokens {{config.params.max_new_tokens}}"
    "{% if config.params.limit_samples is not none %}"
    " --limit-samples {{config.params.limit_samples}}"
    "{% endif %}"
    "{% if target.api_endpoint.api_key_name is not none %}"
    " --api-key-name {{target.api_endpoint.api_key_name}}"
    "{% endif %}"
    "{% if config.params.parallelism is not none %}"
    " --parallelism {{config.params.parallelism}}"
    "{% endif %}"
    "{% if config.params.extra.n_repeats is defined %}"
    " --n-repeats {{config.params.extra.n_repeats}}"
    "{% endif %}"
)


def _build_fdf(
    normalized_name: str,
    bench: BenchmarkDefinition,
    benchmark_module_ref: str,
    dataset_path: str,
) -> dict:
    """Build a single Framework Definition Format (FDF) dict for a benchmark.

    Args:
        normalized_name: Normalized benchmark name.
        bench: BenchmarkDefinition from the registry.
        benchmark_module_ref: Module reference (absolute path or dotted name).
        dataset_path: Resolved dataset path.

    Returns:
        FDF dict ready for YAML serialization.
    """
    extra_params: dict = {
        "benchmark_module": benchmark_module_ref,
        "benchmark_name": normalized_name,
        "dataset": dataset_path,
        "requirements": bench.requirements,
    }
    # Propagate field_mapping if declared
    if bench.field_mapping:
        extra_params["field_mapping"] = bench.field_mapping
    # Propagate judge config(s) from @benchmark kwargs
    # Supports: judge={...}, judge_1={...}, judge_2={...}, etc.
    for key, value in bench.extra_config.items():
        if key == "judge" or (key.startswith("judge_") and key[6:].isdigit()):
            extra_params[key] = value
            extra_params["judge_support"] = True

    defaults: dict = {
        "config": {
            "params": {
                "limit_samples": None,
                "max_new_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "extra": extra_params,
            },
        },
        "target": {"api_endpoint": {}},
    }

    defaults["command"] = COMMAND_TEMPLATE

    return {
        "framework": {
            "name": f"byob_{normalized_name}",
            "pkg_name": f"byob_{normalized_name}",
        },
        "defaults": defaults,
        "evaluations": [
            {
                "name": bench.name,
                "description": f"BYOB benchmark: {bench.name}",
                "defaults": {
                    "config": {
                        "type": f"byob_{normalized_name}.{bench.name}",
                        "supported_endpoint_types": [bench.endpoint_type],
                    }
                },
            }
        ],
    }


def check_existing_benchmarks(normalized_name: str) -> Optional[str]:
    """Check if a benchmark name collides with an already-installed core_evals package.

    Scans both ``core_evals`` and ``nemo_evaluator`` namespaces for installed
    harness packages whose name matches the proposed BYOB package.

    Args:
        normalized_name: The normalized benchmark name (e.g. ``"global_mmlu"``).

    Returns:
        The name of the conflicting package if found, or None.
    """
    pkg_name = f"byob_{normalized_name}"
    existing_names: List[str] = []

    # Scan core_evals namespace
    try:
        import core_evals

        for importer, modname, ispkg in pkgutil.iter_modules(core_evals.__path__):
            existing_names.append(modname)
    except (ImportError, AttributeError):
        pass

    # Check for exact match or name collision (e.g. user creates "global_mmlu"
    # but "global_mmlu" already exists as a non-BYOB core eval)
    if pkg_name in existing_names:
        return pkg_name
    if normalized_name in existing_names:
        return normalized_name

    return None


def compile_benchmark(module_path: str) -> Dict[str, dict]:
    """
    Import a user's benchmark module and generate Framework Definition Format (FDF) dicts.

    Args:
        module_path: Path to .py file or Python module name

    Returns:
        Dict mapping normalized benchmark names to FDF dicts

    Raises:
        ValueError: If no benchmarks found in module
    """
    # Clear registry for fresh state
    clear_registry()

    # Save sys.path and sys.modules for restoration
    saved_path = sys.path[:]
    saved_modules = set(sys.modules.keys())

    try:
        # Resolve module path
        parent_dir = None
        if os.path.exists(module_path):
            # It's a file path
            abs_path = os.path.abspath(module_path)
            parent_dir = os.path.dirname(abs_path)
            module_name = Path(abs_path).stem

            # Add parent directory to sys.path for import resolution
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            resolved_module = module_name
            benchmark_module_ref = (
                abs_path  # Use absolute path for subprocess invocation
            )
        else:
            # It's a dotted module name
            module_name = module_path
            resolved_module = module_name
            benchmark_module_ref = (
                module_name  # Use module name for subprocess invocation
            )

        # Import or reload module (triggers decorator execution)
        if resolved_module in sys.modules:
            importlib.reload(sys.modules[resolved_module])
        else:
            importlib.import_module(resolved_module)

        # Read registered benchmarks
        benchmarks = get_registered_benchmarks()
        if not benchmarks:
            raise ValueError(
                f"No benchmarks found in module '{module_path}'. "
                "Did you use @benchmark and @scorer decorators?"
            )

        # Check for collisions with existing core_evals packages
        for normalized_name in benchmarks:
            collision = check_existing_benchmarks(normalized_name)
            if collision:
                _logger.warning(
                    "Benchmark name collides with existing core_evals package",
                    byob_name=f"byob_{normalized_name}",
                    existing=collision,
                )

        # Build FDF dict for each benchmark
        compiled = {}
        for normalized_name, bench in benchmarks.items():
            # Resolve dataset path to absolute if it's a file.
            # Check both CWD and the benchmark module's directory.
            if os.path.exists(bench.dataset):
                dataset_path = os.path.abspath(bench.dataset)
            elif parent_dir and os.path.exists(os.path.join(parent_dir, bench.dataset)):
                dataset_path = os.path.abspath(os.path.join(parent_dir, bench.dataset))
            else:
                dataset_path = bench.dataset

            compiled[normalized_name] = _build_fdf(
                normalized_name=normalized_name,
                bench=bench,
                benchmark_module_ref=benchmark_module_ref,
                dataset_path=dataset_path,
            )

        return compiled

    finally:
        # Restore sys.path
        sys.path[:] = saved_path

        # Remove modules added during compilation (but keep nemo_evaluator.*)
        new_modules = set(sys.modules.keys()) - saved_modules
        for mod_name in new_modules:
            if not mod_name.startswith("nemo_evaluator."):
                sys.modules.pop(mod_name, None)


def install_benchmark(
    normalized_name: str, fdf: dict, install_dir: Optional[str] = None
) -> str:
    """
    Install a compiled benchmark as a core_evals namespace package.

    Args:
        normalized_name: Normalized benchmark name
        fdf: Framework Definition Format dict
        install_dir: Installation directory (default: ~/.nemo-evaluator/byob_packages/)

    Returns:
        Path to the installed package directory
    """
    # Resolve install directory
    if install_dir is None:
        install_dir = os.path.expanduser("~/.nemo-evaluator/byob_packages/")

    pkg_name = f"byob_{normalized_name}"
    pkg_dir = os.path.join(install_dir, pkg_name)

    # Create directory structure
    os.makedirs(pkg_dir, exist_ok=True)

    # Extract user requirements from FDF
    user_reqs = (
        fdf.get("defaults", {})
        .get("config", {})
        .get("params", {})
        .get("extra", {})
        .get("requirements", [])
    )

    # Write pyproject.toml
    pyproject_path = os.path.join(pkg_dir, "pyproject.toml")
    with open(pyproject_path, "w") as f:
        f.write(_generate_pyproject_toml(pkg_name, user_requirements=user_reqs))

    # Create core_evals namespace package
    core_evals_dir = os.path.join(pkg_dir, "core_evals")
    os.makedirs(core_evals_dir, exist_ok=True)

    # Write core_evals/__init__.py (pkgutil namespace)
    core_evals_init = os.path.join(core_evals_dir, "__init__.py")
    with open(core_evals_init, "w") as f:
        f.write("__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n")

    # Create byob_{name} sub-package
    benchmark_pkg_dir = os.path.join(core_evals_dir, pkg_name)
    os.makedirs(benchmark_pkg_dir, exist_ok=True)

    # Write byob_{name}/__init__.py (empty)
    benchmark_init = os.path.join(benchmark_pkg_dir, "__init__.py")
    with open(benchmark_init, "w") as f:
        f.write("")

    # Write framework.yml
    framework_yml = os.path.join(benchmark_pkg_dir, "framework.yml")
    with open(framework_yml, "w") as f:
        yaml.safe_dump(fdf, f, default_flow_style=False, sort_keys=False)

    # Write output.py for parsing subprocess results
    output_py = os.path.join(benchmark_pkg_dir, "output.py")
    with open(output_py, "w") as f:
        f.write(_generate_output_py())

    return pkg_dir


def _generate_output_py() -> str:
    """Generate output.py content for parsing byob_results.json."""
    return '''"""Output parser for BYOB benchmark."""
import json
import os

from nemo_evaluator.api.api_dataclasses import (
    EvaluationResult,
    MetricResult,
    Score,
    ScoreStats,
    TaskResult,
)


def parse_output(output_dir: str) -> EvaluationResult:
    """Parse BYOB runner output into EvaluationResult."""
    results_path = os.path.join(output_dir, "byob_results.json")
    with open(results_path) as f:
        raw = json.load(f)

    tasks = {}
    for task_name, task_data in raw.get("tasks", {}).items():
        metrics = {}
        for metric_name, metric_data in task_data.get("metrics", {}).items():
            scores = {}
            for score_name, score_data in metric_data.get("scores", {}).items():
                # Construct ScoreStats with available fields
                # NOTE: Score.stats is NON-OPTIONAL in real Pydantic model
                scores[score_name] = Score(
                    value=score_data["value"],
                    stats=ScoreStats(
                        count=score_data.get("count"),
                        mean=score_data.get("mean"),
                        stderr=score_data.get("stderr"),
                        stddev=score_data.get("stddev"),
                        # Other ScoreStats fields default to None
                    ),
                )
            metrics[metric_name] = MetricResult(scores=scores)
        tasks[task_name] = TaskResult(metrics=metrics)

    return EvaluationResult(tasks=tasks)
'''


def _generate_pyproject_toml(
    pkg_name: str, user_requirements: Optional[List[str]] = None
) -> str:
    """Generate pyproject.toml content for the namespace package."""
    deps = ["nemo-evaluator"]
    if user_requirements:
        deps.extend(user_requirements)
    deps_str = ", ".join(f'"{d}"' for d in deps)

    return f"""[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "core-evals-{pkg_name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [{deps_str}]

[tool.setuptools.packages.find]
include = ["core_evals", "core_evals.*"]

[tool.setuptools.package-data]
"core_evals.{pkg_name}" = ["framework.yml"]
"""
