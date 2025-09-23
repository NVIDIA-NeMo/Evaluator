import os

import pytest


@pytest.fixture(autouse=True, scope="module")
def set_env_vars():
    """Set environment variables for the tests."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = "/home/TestData/HF_HOME"
    os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"
    yield
