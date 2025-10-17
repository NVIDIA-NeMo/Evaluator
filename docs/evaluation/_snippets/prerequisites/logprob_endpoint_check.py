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
#!/usr/bin/env python3
"""
Pre-flight check: Verify completions endpoint with log-probability support.
"""

import os
import sys

import requests


def check_logprob_endpoint(endpoint_url: str, api_key: str, model_id: str) -> bool:
    """Check if completions endpoint supports log probabilities."""
    # [snippet-start]
    try:
        response = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "prompt": "Hello",
                "max_tokens": 10,
                "logprobs": 1,  # Request log probabilities
            },
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Endpoint returned status {response.status_code}"
        )
        assert "logprobs" in response.json().get("choices", [{}])[0], (
            "Endpoint doesn't support logprobs"
        )
        print("✓ Completions endpoint ready with log-probability support")
        return True
    except Exception as e:
        print(f"✗ Endpoint check failed: {e}")
        return False
    # [snippet-end]


if __name__ == "__main__":
    endpoint_url = os.getenv("ENDPOINT_URL", "http://0.0.0.0:8080/v1/completions")
    api_key = os.getenv("YOUR_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "megatron_model")

    if not api_key:
        print("Error: Set YOUR_API_KEY environment variable")
        sys.exit(1)

    success = check_logprob_endpoint(endpoint_url, api_key, model_id)
    sys.exit(0 if success else 1)
