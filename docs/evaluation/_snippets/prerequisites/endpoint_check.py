#!/usr/bin/env python3
"""
Pre-flight check: Verify endpoint connectivity before running evaluations.
"""
import os
import sys

import requests


def check_endpoint(endpoint_url: str, api_key: str, model_id: str) -> bool:
    """Check if endpoint is ready for evaluation."""
    # [snippet-start]
    try:
        response = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        assert response.status_code == 200, f"Endpoint returned status {response.status_code}"
        print("✓ Endpoint ready for evaluation")
        return True
    except Exception as e:
        print(f"✗ Endpoint check failed: {e}")
        print("Ensure your API key is valid and the endpoint is accessible")
        return False
    # [snippet-end]


if __name__ == "__main__":
    # Example usage
    endpoint_url = os.getenv("ENDPOINT_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    api_key = os.getenv("YOUR_API_KEY", "")
    model_id = os.getenv("MODEL_ID", "meta/llama-3.1-8b-instruct")
    
    if not api_key:
        print("Error: Set YOUR_API_KEY environment variable")
        sys.exit(1)
    
    success = check_endpoint(endpoint_url, api_key, model_id)
    sys.exit(0 if success else 1)

