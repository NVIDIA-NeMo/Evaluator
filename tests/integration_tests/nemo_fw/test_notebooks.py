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
