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

"""Template compilation tests.

Tests T007-T012: Verify each of the 6 templates compiles independently
and produces valid namespace package structure.
"""

import os

import pytest
import yaml

from nemo_evaluator.contrib.byob.compiler import compile_benchmark, install_benchmark
from nemo_evaluator.contrib.byob.decorators import clear_registry

from .conftest import TEMPLATE_DIR, TEMPLATES


@pytest.mark.parametrize("template_name", TEMPLATES)
def test_template_compiles_independently(template_name, tmp_path):
    """T007-T012: Each template compiles independently.

    Validates:
    - compile_benchmark() returns non-empty dict (FDF)
    - FDF has required keys: framework, defaults, evaluations
    - framework.name starts with "byob_"
    - framework.pkg_name starts with "byob_"
    - defaults.command contains "nemo_evaluator.contrib.byob.runner"
    - benchmark_module path is absolute
    - dataset path is absolute (or doesn't exist at compile time)
    - install_benchmark() produces valid namespace package structure
    - framework.yml round-trips correctly
    - output.py contains parse_output function
    """
    # Clear registry to ensure isolation
    clear_registry()

    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.py")
    assert os.path.isfile(template_path), (
        f"Template not found: {template_path}. "
        f"Check that {template_name}.py exists in examples/byob/templates/"
    )

    # Compile the template
    compiled = compile_benchmark(template_path)
    assert len(compiled) > 0, (
        f"compile_benchmark({template_name}) returned empty dict. "
        f"Expected at least one benchmark registration."
    )

    # Process each registered benchmark (templates should have exactly one)
    for normalized_name, fdf in compiled.items():
        # Validate FDF structure
        assert "framework" in fdf, (
            f"FDF for {template_name} missing 'framework' key. "
            f"Keys: {list(fdf.keys())}"
        )
        assert "defaults" in fdf, (
            f"FDF for {template_name} missing 'defaults' key. "
            f"Keys: {list(fdf.keys())}"
        )
        assert "evaluations" in fdf, (
            f"FDF for {template_name} missing 'evaluations' key. "
            f"Keys: {list(fdf.keys())}"
        )

        # Validate framework metadata
        framework_name = fdf["framework"]["name"]
        assert framework_name.startswith("byob_"), (
            f"Template {template_name} framework.name should start with 'byob_', "
            f"got: {framework_name}"
        )

        pkg_name = fdf["framework"]["pkg_name"]
        assert pkg_name.startswith("byob_"), (
            f"Template {template_name} framework.pkg_name should start with 'byob_', "
            f"got: {pkg_name}"
        )

        # Validate command template
        command = fdf["defaults"]["command"]
        assert "nemo_evaluator.contrib.byob.runner" in command, (
            f"Template {template_name} command should reference runner module. "
            f"Got: {command}"
        )

        # Validate paths are absolute
        extra = fdf["defaults"]["config"]["params"]["extra"]
        benchmark_module = extra["benchmark_module"]
        assert os.path.isabs(benchmark_module), (
            f"Template {template_name} benchmark_module should be absolute path. "
            f"Got: {benchmark_module}"
        )

        dataset_path = extra["dataset"]
        # Dataset path should be absolute if the file exists at compile time
        # (which it always does for templates in the repo)
        if os.path.exists(dataset_path):
            assert os.path.isabs(dataset_path), (
                f"Template {template_name} dataset path should be absolute "
                f"when file exists at compile time. Got: {dataset_path}"
            )

        # Install the benchmark to a temporary directory
        pkg_dir = install_benchmark(normalized_name, fdf, install_dir=str(tmp_path))

        # Verify namespace package structure
        pkg_name_dir = f"byob_{normalized_name}"

        pyproject_path = os.path.join(pkg_dir, "pyproject.toml")
        assert os.path.isfile(pyproject_path), (
            f"Template {template_name} installation missing pyproject.toml. "
            f"Expected at: {pyproject_path}"
        )

        core_evals_init = os.path.join(pkg_dir, "core_evals", "__init__.py")
        assert os.path.isfile(core_evals_init), (
            f"Template {template_name} installation missing core_evals/__init__.py. "
            f"Expected at: {core_evals_init}"
        )

        pkg_init = os.path.join(pkg_dir, "core_evals", pkg_name_dir, "__init__.py")
        assert os.path.isfile(pkg_init), (
            f"Template {template_name} installation missing package __init__.py. "
            f"Expected at: {pkg_init}"
        )

        framework_yml_path = os.path.join(pkg_dir, "core_evals", pkg_name_dir, "framework.yml")
        assert os.path.isfile(framework_yml_path), (
            f"Template {template_name} installation missing framework.yml. "
            f"Expected at: {framework_yml_path}"
        )

        output_py_path = os.path.join(pkg_dir, "core_evals", pkg_name_dir, "output.py")
        assert os.path.isfile(output_py_path), (
            f"Template {template_name} installation missing output.py. "
            f"Expected at: {output_py_path}"
        )

        # Validate framework.yml round-trips
        with open(framework_yml_path) as f:
            loaded_yml = yaml.safe_load(f)

        assert loaded_yml["framework"]["name"] == fdf["framework"]["name"], (
            f"Template {template_name} framework.yml does not round-trip correctly. "
            f"Expected framework.name: {fdf['framework']['name']}, "
            f"got: {loaded_yml['framework']['name']}"
        )

        # Validate output.py contains parse_output function
        with open(output_py_path) as f:
            output_py_content = f.read()

        assert "def parse_output" in output_py_content or "parse_output =" in output_py_content, (
            f"Template {template_name} output.py missing parse_output function. "
            f"This function is required by the evaluation framework."
        )
