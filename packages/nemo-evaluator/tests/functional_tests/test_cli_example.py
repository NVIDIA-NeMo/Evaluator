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

import subprocess

import pytest

from nemo_evaluator.cli.example import add_example_files


@pytest.mark.parametrize("use_cli", [True, False])
def test_cli_example(tmp_path, use_cli):
    import logging

    logging.basicConfig(level=logging.DEBUG)
    if use_cli:
        subprocess.run(["nemo_evaluator_example", "my_cool_package", tmp_path])
    else:
        add_example_files(package_name="my_cool_package", destination_folder=tmp_path)

    assert (tmp_path / "core_evals" / "my_cool_package" / "__init__.py").exists()
    assert (tmp_path / "core_evals" / "my_cool_package" / "framework.yml").exists()
    assert (tmp_path / "core_evals" / "my_cool_package" / "output.py").exists()

    with open(tmp_path / "core_evals" / "my_cool_package" / "framework.yml", "r") as f:
        content = f.read()
    assert "my_cool_package" in content

    with open(tmp_path / "core_evals" / "my_cool_package" / "output.py", "r") as f:
        content = f.read()
    assert "parse_output(output_dir: str) -> EvaluationResult" in content


def test_cli_example_with_existing_files(tmp_path):
    content = "This is a test file"
    (tmp_path / "core_evals" / "my_cool_package").mkdir(parents=True, exist_ok=True)
    for target in ["__init__.py", "framework.yml", "framework_entrypoint.py"]:
        with open(tmp_path / "core_evals" / "my_cool_package" / target, "w") as f:
            f.write(content)

    add_example_files(package_name="my_cool_package", destination_folder=tmp_path)
    for target in ["__init__.py", "framework.yml", "framework_entrypoint.py"]:
        with open(tmp_path / "core_evals" / "my_cool_package" / target, "r") as f:
            assert f.read() == content
