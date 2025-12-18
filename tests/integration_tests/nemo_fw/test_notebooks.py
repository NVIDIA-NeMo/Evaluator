# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import subprocess
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

dir_path = Path(__file__).parent.parent.parent.parent / "tutorials"
print(dir_path)


@pytest.fixture(autouse=True)
def uninstall_nvidia_simple_evals():
    """nvidia-simple-evals is installed in simple-evals.ipynb,
    this fixture removes it after each test to ensure cleanup.
    """
    yield
    subprocess.run(["pip", "uninstall", "-y", "nvidia-simple-evals"])


# FIXME(martas): Errors out due to an MCore bug on deployment side
# enable once fixed in Export-Deploy
@pytest.mark.pleasefixme
@pytest.mark.parametrize(
    "notebook_path",
    [
        dir_path / name
        for name in ["mmlu.ipynb", "simple-evals.ipynb", "wikitext.ipynb"]
    ],
)
def test_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor()

    executed_nb, resources = ep.preprocess(nb)

    assert executed_nb is not None
