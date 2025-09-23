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
import signal
import subprocess

import pytest
from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationResult,
    EvaluationTarget,
)

# FIXME(martas): EF packages pre 25.09 use old imports from nvidia_eval_commons
from nvidia_eval_commons.api.api_dataclasses import (
    EvaluationResult as LegacyEvaluationResult,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def deployment_process(set_env_vars):
    """Fixture to deploy the HF model on Ray server."""
    hf_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "hf_model"
    port = 8886
    # Run deployment
    deploy_proc = subprocess.Popen(
        [
            "python",
            "/opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py",
            "--model_path",
            hf_model_id,
            "--model_id",
            model_name,
            "--port",
            str(port),
        ]
    )

    completions_ready = check_endpoint(
        endpoint_url=f"http://0.0.0.0:{port}/v1/completions/",
        endpoint_type="completions",
        model_name=model_name,
        max_retries=600,
    )
    assert completions_ready, (
        "Completions endpoint is not ready. Please look at the deploy process log for the error"
    )

    chat_ready = check_endpoint(
        endpoint_url=f"http://0.0.0.0:{port}/v1/chat/completions",
        endpoint_type="chat",
        model_name=model_name,
        max_retries=600,
    )
    assert chat_ready, (
        "Chat endpoint is not ready. Please look at the deploy process log for the error"
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


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "eval_type,endpoint_type,eval_params",
    [
        ("gsm8k", "completions", {"limit_samples": 1, "request_timeout": 360}),
        (
            "arc_challenge",
            "completions",
            {
                "limit_samples": 1,
                "request_timeout": 360,
                "extra": {
                    "tokenizer_backend": "huggingface",
                    "tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
                },
            },
        ),
        ("ifeval", "chat", {"limit_samples": 1, "request_timeout": 360}),
    ],
)
def test_evaluation(
    deployment_process, eval_type, endpoint_type, eval_params, tmp_path
):
    """
    Test evaluation of a nemo model deployed with triton backend.
    """
    port = 8886
    if endpoint_type == "completions":
        url = f"http://0.0.0.0:{port}/v1/completions/"
    elif endpoint_type == "chat":
        url = f"http://0.0.0.0:{port}/v1/chat/completions"
    else:
        raise ValueError(f"Invalid endpoint type: {endpoint_type}")

    api_endpoint = ApiEndpoint(url=url, type=endpoint_type, model_id="megatron_model")
    eval_target = EvaluationTarget(api_endpoint=api_endpoint)
    eval_config = EvaluationConfig(
        type=eval_type, params=ConfigParams(**eval_params), output_dir=str(tmp_path)
    )
    results = evaluate(target_cfg=eval_target, eval_cfg=eval_config)
    # FIXME(martas): EF packages pre 25.09 use old imports from nvidia_eval_commons
    assert isinstance(results, EvaluationResult) or isinstance(
        results, LegacyEvaluationResult
    )
    logger.info("Evaluation completed.")
