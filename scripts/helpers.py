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
import time

import requests
from nemo_evaluator.api import evaluate

logger = logging.getLogger(__name__)


def check_endpoint(
    endpoint_url: str,
    endpoint_type: str,
    model_name: str,
    max_retries: int = 600,
    retry_interval: int = 2,
) -> bool:
    """
    Check if the endpoint is responsive and ready to accept requests.
    """

    payload = {"model": model_name, "max_tokens": 1}
    if endpoint_type == "completions":
        payload["prompt"] = "hello, my name is"
    elif endpoint_type == "chat":
        payload["messages"] = [{"role": "user", "content": "hello, what is your name?"}]
    else:
        raise ValueError(f"Invalid endpoint type: {endpoint_type}")

    for _ in range(max_retries):
        try:
            response = requests.post(endpoint_url, json=payload)
            if response.status_code == 200:
                return True
            logger.info(f"Server replied with status code: {response.status_code}")
            time.sleep(retry_interval)
        except requests.exceptions.RequestException:
            logger.info("Server is not ready")
            time.sleep(retry_interval)
    return False


def wait_and_evaluate(target_cfg, eval_cfg):
    server_ready = check_endpoint(
        endpoint_url=target_cfg.api_endpoint.url,
        endpoint_type=target_cfg.api_endpoint.type,
        model_name=target_cfg.api_endpoint.model_id,
    )
    if not server_ready:
        raise RuntimeError(
            "Server is not ready to accept requests. Check the deployment logs for errors."
        )
    return evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)
