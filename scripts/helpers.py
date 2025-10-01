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

import requests
from nemo_evaluator.core.evaluate import evaluate


def check_endpoint(endpoint_url: str, endpoint_type: str, model_name: str, max_retries: int = 60, retry_interval: int = 2) -> bool:
    """Check if endpoint is ready for requests."""
    import time
    
    test_payload = {
        "model": model_name,
        "prompt": "Hello" if endpoint_type == "completions" else None,
        "messages": [{"role": "user", "content": "Hello"}] if endpoint_type == "chat" else None,
        "max_tokens": 5
    }
    
    for _ in range(max_retries):
        try:
            response = requests.post(endpoint_url, json=test_payload, timeout=10)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
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
