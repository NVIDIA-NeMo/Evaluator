import os
import signal
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


@pytest.fixture(scope="module", autouse=True)
def fastapi_process():
    # FIXME(martas): uvicorn is now part of the deploy_inframework_triton.py script
    # this fixture should be removed for Export-Deploy@fcec69d and port and host should be
    # passed to the script in the notebook
    fastapi_proc = subprocess.Popen(
        [
            "python",
            "-c",
            "import uvicorn; uvicorn.run('nemo_deploy.service.fastapi_interface_to_pytriton:app', host='0.0.0.0', port=8080)",
        ]
    )
    yield fastapi_proc

    fastapi_proc.send_signal(signal.SIGINT)
    try:
        fastapi_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(fastapi_proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize(
    "notebook_path",
    [
        dir_path / name
        for name in ["mmlu.ipynb", "simple-evals.ipynb", "wikitext.ipynb"]
    ],
)
def test_notebook(notebook_path, fastapi_process):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor()

    executed_nb, resources = ep.preprocess(nb)

    assert executed_nb is not None
