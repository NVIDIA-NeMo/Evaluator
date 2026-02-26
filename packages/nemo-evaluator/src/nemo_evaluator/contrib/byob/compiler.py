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

"""BYOB compiler: transforms user benchmark modules into nemo_evaluator namespace packages."""

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
    DEFAULT_TOP_P,
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
    " --top-p {{config.params.top_p}}"
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
    "{% if config.params.request_timeout is not none %}"
    " --request-timeout {{config.params.request_timeout}}"
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
                "task": normalized_name,
                "limit_samples": None,
                "max_new_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
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


_PTH_FILENAME = "nemo_evaluator_byob.pth"


def _make_pth_line(install_dir: str) -> str:
    """Build a single ``.pth`` line that extends ``nemo_evaluator.__path__``.

    Each line is a self-contained Python snippet that globs
    ``<install_dir>/*/nemo_evaluator`` and appends any matches to
    ``nemo_evaluator.__path__``.  Multiple lines (one per distinct
    ``--install-dir``) can coexist in the same ``.pth`` file.
    """
    # Normalize to absolute so the .pth line is stable across cwd changes
    abs_dir = os.path.abspath(install_dir)
    return (
        "import os, glob, nemo_evaluator; "
        "[nemo_evaluator.__path__.append(d) "
        f"for d in glob.glob(os.path.join({abs_dir!r}, '*', 'nemo_evaluator')) "
        "if d not in nemo_evaluator.__path__]"
    )


def _ensure_pth_file(install_dir: str) -> Optional[str]:
    """Ensure the ``.pth`` file in site-packages contains a line for *install_dir*.

    The ``.pth`` file extends ``nemo_evaluator.__path__`` at Python startup
    so the harness discovery code finds BYOB sub-packages — even when the
    main nemo-evaluator is installed in editable (development) mode.

    Each unique ``--install-dir`` adds one line.  Repeated compilations with
    the same directory are idempotent (the line is not duplicated).

    Args:
        install_dir: The BYOB package install directory to register.

    Returns:
        Path to the ``.pth`` file, or None if site-packages could not
        be determined.
    """
    import site

    # Find a writable site-packages directory
    candidates = []
    try:
        candidates.append(site.getusersitepackages())
    except AttributeError:
        pass
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass

    site_dir = None
    for sp in candidates:
        if os.path.isdir(sp):
            site_dir = sp
            break

    if site_dir is None:
        return None

    pth_path = os.path.join(site_dir, _PTH_FILENAME)
    new_line = _make_pth_line(install_dir)

    # Read existing lines (if any) to avoid duplicates
    existing_lines: List[str] = []
    if os.path.exists(pth_path):
        with open(pth_path) as f:
            existing_lines = [ln.rstrip("\n") for ln in f.readlines()]

    if new_line in existing_lines:
        return pth_path  # already registered

    try:
        os.makedirs(site_dir, exist_ok=True)
        with open(pth_path, "a") as f:
            f.write(new_line + "\n")
        _logger.info(
            "Registered BYOB install dir in .pth file",
            pth=pth_path,
            install_dir=install_dir,
        )
        return pth_path
    except OSError as e:
        _logger.warning(
            "Could not write .pth file (BYOB packages may need manual setup)",
            path=pth_path,
            error=str(e),
        )
        return None


def install_benchmark(
    normalized_name: str, fdf: dict, install_dir: Optional[str] = None
) -> str:
    """
    Install a compiled benchmark as a nemo_evaluator namespace sub-package.

    The compiled package creates ``nemo_evaluator/byob_<name>/`` inside the
    install directory.  A ``.pth`` file in site-packages extends
    ``nemo_evaluator.__path__`` at Python startup so the harness discovery
    code (``core/input.py:_get_harness_packages``) finds the sub-package
    without any changes to ``nemo_evaluator/__init__.py``.

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

    # Create nemo_evaluator namespace directory (no __init__.py — main package owns it)
    ns_dir = os.path.join(pkg_dir, "nemo_evaluator")
    os.makedirs(ns_dir, exist_ok=True)

    # Create byob_{name} sub-package
    benchmark_pkg_dir = os.path.join(ns_dir, pkg_name)
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

    # Ensure the .pth file registers this install_dir for namespace merging
    _ensure_pth_file(install_dir)

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
name = "nemo-evaluator-{pkg_name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [{deps_str}]

[tool.setuptools.packages]
find = {{namespaces = true, include = ["nemo_evaluator.{pkg_name}"]}}

[tool.setuptools.package-data]
"nemo_evaluator.{pkg_name}" = ["framework.yml"]
"""
