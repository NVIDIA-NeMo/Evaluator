import os
import signal
import subprocess

import pytest

from nemo_eval.utils.base import check_endpoint, check_health


@pytest.fixture
def completions_request():
    return {
        "model": "megatron_model",
        "prompt": "Hello, world!",
        "max_tokens": 10,
        "temperature": 0.5,
    }


@pytest.fixture
def logprobs_request():
    return {
        "model": "megatron_model",
        "prompt": "Hello, world!",
        "logprobs": 1,
        "echo": True,
    }


@pytest.fixture
def chat_request():
    return {
        "model": "megatron_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 10,
        "temperature": 0.5,
    }


@pytest.mark.parametrize("serving_backend", ["pytriton", "ray"])
def test_deployment(serving_backend, completions_request, logprobs_request, chat_request):
    """Fixture to create a Flask app with an OpenAI response.

    Being a "proper" fake endpoint, it responds with a payload which can be
    set via app.config.response.
    """
    # Create and run the fake endpoint server
    nemo2_ckpt_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua"
    max_batch_size = 4
    legacy_ckpt = True
    port = 8886
    if serving_backend == "pytriton":
        health_url = f"http://0.0.0.0:{port}/v1/triton_health"
    elif serving_backend == "ray":
        health_url = f"http://0.0.0.0:{port}/v1/health"
    else:
        raise ValueError(f"Invalid serving backend: {serving_backend}")

    try:
        deploy_proc = subprocess.Popen(
            [
                "python",
                "tests/functional_tests/deploy_in_fw_script.py",
                "--nemo2_ckpt_path",
                nemo2_ckpt_path,
                "--max_batch_size",
                str(max_batch_size),
                "--port",
                str(port),
                "--serving_backend",
                serving_backend,
            ]
            + (["--legacy_ckpt"] if legacy_ckpt else [])
        )
        assert check_health(health_url, max_retries=100)
        # Test completions
        assert check_endpoint(f"http://0.0.0.0:{port}/v1/completions", "completions", "megatron_model", max_retries=10)
        # TODO: not supported with ray yet
        # response = requests.post(f"http://0.0.0.0:{port}/v1/completions", json=logprobs_request)
        # assert response.status_code == 200
        # TODO: we need a chat checkpoint for this
        # response = requests.post(f"http://0.0.0.0:{port}/v1/chat/completions", json=chat_request)
        # assert response.status_code == 200
    finally:
        deploy_proc.send_signal(signal.SIGTERM)
        try:
            deploy_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(deploy_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            subprocess.run(["pkill", f"-{signal.SIGTERM}", "tritonserver"], check=False)
