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

import logging
import os
import shutil
import signal
import subprocess
import tempfile

import pytest
from nvidia_eval_commons.api.api_dataclasses import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
from nvidia_eval_commons.core.evaluate import evaluate

from nemo_eval.utils.base import check_endpoint

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def deployment_process():
    """Fixture to create a Flask app with an OpenAI response.

    Being a "proper" fake endpoint, it responds with a payload which can be
    set via app.config.response.
    """
    # Create and run the fake endpoint server
    nemo2_ckpt_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua"
    max_batch_size = 4
    legacy_ckpt = True
    port = 8886
    # Run deployment
    deploy_proc = subprocess.Popen(
        [
            "coverage",
            "run",
            "--data-file=/workspace/.coverage",
            "--source=/workspace/",
            "tests/functional_tests/deploy_in_fw_script.py",
            "--nemo2_ckpt_path",
            nemo2_ckpt_path,
            "--max_batch_size",
            str(max_batch_size),
            "--port",
            str(port),
        ]
        + (["--legacy_ckpt"] if legacy_ckpt else [])
    )

    yield deploy_proc  # We only need the process reference for cleanup

    deploy_proc.send_signal(signal.SIGINT)
    try:
        deploy_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(deploy_proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    finally:
        subprocess.run(["pkill", f"-{signal.SIGTERM}", "tritonserver"], check=False)


@pytest.fixture(autouse=True)
def cleanup_results():
    """Clean up results directory after each test."""
    yield
    results_dir = "results"
    if os.path.exists(results_dir):
        logger.info(f"Cleaning up results directory: {results_dir}")
        shutil.rmtree(results_dir)


# Broken in NVIDIA/CUDA, needs flash-attn
# @pytest.mark.pleasefixme
class TestEvaluation:
    """
    Test evaluation with NVIDIA Evals Factory on nemo2 model deployed on PyTriton.
    """

    def test_dummy_test(self):
        return True

    @pytest.mark.run_only_on("GPU")
    def test_gsm8k_evaluation(self, deployment_process):
        """
        Test GSM8K evaluation benchmark.
        """

        eval_type = "gsm8k"
        limit = 1
        port = 8886

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = "/home/TestData/HF_HOME"
        os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

        # Wait for server readiness
        logger.info("Waiting for server readiness...")
        server_ready = check_endpoint(
            endpoint_url=f"http://0.0.0.0:{port}/v1/completions/",
            endpoint_type="completions",
            model_name="megatron_model",
            max_retries=600,
        )
        assert server_ready, "Server is not ready. Please look at the deploy process log for the error"

        # Run evaluation
        logger.info("Starting evaluation...")
        api_endpoint = ApiEndpoint(
            url=f"http://0.0.0.0:{port}/v1/completions/", type="completions", model_id="megatron_model"
        )
        eval_target = EvaluationTarget(api_endpoint=api_endpoint)
        eval_params = {"limit_samples": limit, "request_timeout": 360}
        temp_dir = tempfile.TemporaryDirectory()
        eval_config = EvaluationConfig(type=eval_type, params=ConfigParams(**eval_params), output_dir=temp_dir.name)
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
        logger.info("Evaluation completed.")

    @pytest.mark.run_only_on("GPU")
    def test_arc_challenge_evaluation(self, deployment_process):
        """
        Test ARC Challenge evaluation benchmark.
        """
        tokenizer_path = "/home/TestData/nemo2_ckpt/llama3-1b-lingua/context/lingua"
        eval_type = "arc_challenge"
        limit = 1
        port = 8886

        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = "/home/TestData/HF_HOME"
        os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

        # Wait for server readiness
        logger.info("Waiting for server readiness...")
        server_ready = check_endpoint(
            endpoint_url=f"http://0.0.0.0:{port}/v1/completions/",
            endpoint_type="completions",
            model_name="megatron_model",
            max_retries=600,
        )
        assert server_ready, "Server is not ready. Please look at the deploy process log for the error"

        # Run evaluation
        logger.info("Starting evaluation...")
        api_endpoint = ApiEndpoint(
            url=f"http://0.0.0.0:{port}/v1/completions/", type="completions", model_id="megatron_model"
        )
        eval_target = EvaluationTarget(api_endpoint=api_endpoint)
        eval_params = {
            "limit_samples": limit,
            "extra": {
                "tokenizer_backend": "huggingface",
                "tokenizer": tokenizer_path,
            },
        }
        temp_dir = tempfile.TemporaryDirectory()
        eval_config = EvaluationConfig(type=eval_type, params=ConfigParams(**eval_params), output_dir=temp_dir.name)
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
        logger.info("Evaluation completed.")
