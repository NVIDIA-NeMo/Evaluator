# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BYOB runner: evaluation loop and CLI entrypoint."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from nemo_evaluator.byob.eval_logic import import_benchmark, run_eval_loop


def load_dataset(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load JSONL dataset from file.

    Args:
        path: Path to JSONL file.
        limit: Optional limit on number of samples to load.

    Returns:
        List of sample dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip blank lines
                continue
            data.append(json.loads(line))

    if limit and limit > 0:
        return data[:limit]
    return data


def call_model_chat(
    url: str,
    model_id: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> str:
    """Call OpenAI-compatible chat completions endpoint.

    Args:
        url: Base URL for model endpoint.
        model_id: Model identifier.
        prompt: Prompt text to send as user message.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: Optional Bearer token for Authorization header.

    Returns:
        Generated response text.

    Raises:
        requests.HTTPError: On non-2xx response.
        requests.Timeout: On timeout.
    """
    endpoint = f"{url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def call_model_completions(
    url: str,
    model_id: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> str:
    """Call OpenAI-compatible completions endpoint.

    Args:
        url: Base URL for model endpoint.
        model_id: Model identifier.
        prompt: Prompt text.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: Optional Bearer token for Authorization header.

    Returns:
        Generated response text.

    Raises:
        requests.HTTPError: On non-2xx response.
        requests.Timeout: On timeout.
    """
    endpoint = f"{url}/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    return response.json()["choices"][0]["text"]


def aggregate_scores(scores: List[Dict], benchmark_name: str) -> Dict:
    """Aggregate per-sample scores into summary statistics.

    This is the core aggregation logic:
    1. Collect all numeric keys (bool, int, float) across all sample dicts.
    2. For each key:
       - Convert booleans to 0.0/1.0.
       - Compute: mean, variance (population, /n), stddev, stderr = stddev / sqrt(n).
       - If n <= 1: variance=0, stderr=0.
       - Detect binary: is_binary = all(v in (0.0, 1.0) for v in values).
       - Binary metrics: scale display values by 100 (percentage).
       - Round all values to 4 decimal places.
    3. Output structure conforms to engine expectations.

    Args:
        scores: List of score dictionaries from scorer function.
        benchmark_name: Name of benchmark (used as task name in output).

    Returns:
        Nested dict with structure:
        {
            "tasks": {
                "<benchmark_name>": {
                    "metrics": {
                        "pass@1": {
                            "scores": {
                                "<key>": {
                                    "value": N,
                                    "count": N,
                                    "mean": N,
                                    "stderr": N,
                                    "stddev": N
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    if not scores:
        return {}

    # Collect all numeric keys across all samples
    all_keys = set()
    for score_dict in scores:
        for key, value in score_dict.items():
            if isinstance(value, (bool, int, float)):
                all_keys.add(key)

    # Compute statistics for each key
    aggregated_scores = {}
    for key in all_keys:
        # Extract values for this key, converting booleans to 0.0/1.0
        values = []
        for score_dict in scores:
            if key in score_dict:
                val = score_dict[key]
                if isinstance(val, bool):
                    values.append(1.0 if val else 0.0)
                elif isinstance(val, (int, float)):
                    values.append(float(val))

        if not values:
            continue

        n = len(values)
        mean_val = sum(values) / n

        # Population variance (divide by n, not n-1)
        if n <= 1:
            variance = 0.0
            stddev = 0.0
            stderr = 0.0
        else:
            variance = sum((v - mean_val) ** 2 for v in values) / n
            stddev = math.sqrt(variance)
            stderr = stddev / math.sqrt(n)

        # Detect binary (all values in {0.0, 1.0})
        is_binary = all(v in (0.0, 1.0) for v in values)

        # Scale binary metrics to percentage (0-100)
        if is_binary:
            display_value = mean_val * 100
            display_mean = mean_val * 100
            display_stderr = stderr * 100
            display_stddev = stddev * 100
        else:
            display_value = mean_val
            display_mean = mean_val
            display_stderr = stderr
            display_stddev = stddev

        # Round to 4 decimal places
        aggregated_scores[key] = {
            "value": round(display_value, 4),
            "count": n,
            "mean": round(display_mean, 4),
            "stderr": round(display_stderr, 4),
            "stddev": round(display_stddev, 4),
        }

    return {
        "tasks": {
            benchmark_name: {
                "metrics": {
                    "pass@1": {
                        "scores": aggregated_scores
                    }
                }
            }
        }
    }


def main():
    """CLI entrypoint for BYOB runner."""
    parser = argparse.ArgumentParser(
        description="BYOB runner: execute benchmark evaluation"
    )
    parser.add_argument(
        "--benchmark-module",
        required=True,
        help="Path to .py file or module name",
    )
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help="Normalized benchmark name",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for byob_results.json",
    )
    parser.add_argument(
        "--model-url",
        required=True,
        help="Model endpoint URL (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier",
    )
    parser.add_argument(
        "--model-type",
        default="chat",
        choices=["chat", "completions"],
        help="Endpoint type: 'chat' or 'completions'",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit number of dataset samples",
    )
    parser.add_argument(
        "--api-key-name",
        default=None,
        help="Environment variable name for Bearer token",
    )

    args = parser.parse_args()

    # Resolve API key from environment variable
    api_key = None
    if args.api_key_name:
        api_key = os.environ.get(args.api_key_name)

    # Import benchmark using shared logic
    bench = import_benchmark(args.benchmark_module, args.benchmark_name)

    # Load dataset
    dataset = load_dataset(args.dataset, limit=args.limit_samples)

    # Create model_call_fn that wraps the HTTP calls
    def model_call_fn(prompt: str, endpoint_type: str) -> str:
        """Model call function that routes through subprocess HTTP calls."""
        if endpoint_type == "chat":
            return call_model_chat(
                url=args.model_url,
                model_id=args.model_id,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key=api_key,
            )
        else:
            return call_model_completions(
                url=args.model_url,
                model_id=args.model_id,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key=api_key,
            )

    # Run evaluation loop using shared logic
    all_scores = run_eval_loop(
        bench=bench,
        dataset=dataset,
        model_call_fn=model_call_fn,
        endpoint_type=args.model_type,
    )

    # Aggregate scores
    results = aggregate_scores(all_scores, args.benchmark_name)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / "byob_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
