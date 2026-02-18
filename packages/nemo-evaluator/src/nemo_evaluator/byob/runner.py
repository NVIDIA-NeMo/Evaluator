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
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from nemo_evaluator.byob.dataset import load_dataset
from nemo_evaluator.byob.eval_logic import import_benchmark, run_eval_loop


def call_model_chat(
    url: str,
    model_id: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    timeout: float = 120,
) -> str:
    """Call OpenAI-compatible chat completions endpoint.

    Args:
        url: Base URL for model endpoint.
        model_id: Model identifier.
        prompt: Prompt text to send as user message.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: Optional Bearer token for Authorization header.
        timeout: Request timeout in seconds.

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

    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def call_model_completions(
    url: str,
    model_id: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    timeout: float = 120,
) -> str:
    """Call OpenAI-compatible completions endpoint.

    Args:
        url: Base URL for model endpoint.
        model_id: Model identifier.
        prompt: Prompt text.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: Optional Bearer token for Authorization header.
        timeout: Request timeout in seconds.

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

    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    return response.json()["choices"][0]["text"]


def aggregate_scores(scores: List[Dict], benchmark_name: str) -> Dict:
    """Aggregate per-sample scores into summary statistics.

    This is the core aggregation logic:
    1. Collect all numeric keys (bool, int, float) across all sample dicts.
    2. For each key:
       - Convert booleans to 0.0/1.0.
       - Compute: mean, sample variance (Bessel's correction, /(n-1)), stddev, stderr = stddev / sqrt(n).
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
    for key in sorted(all_keys):
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

        # Sample variance (Bessel's correction: divide by n-1)
        if n <= 1:
            variance = 0.0
            stddev = 0.0
            stderr = 0.0
        else:
            variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
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
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="Save per-sample predictions to byob_predictions.jsonl in output directory",
    )
    parser.add_argument(
        "--timeout-per-sample",
        type=float,
        default=120,
        help="Timeout in seconds for each model call (default: 120)",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Raise error on any skipped sample (missing field or model error)",
    )
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default="text",
        help="Log output format: text (default) or json",
    )

    args = parser.parse_args()

    # Configure JSON logging if requested
    if args.log_format == "json":
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                }
                return json.dumps(log_entry)

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        byob_logger = logging.getLogger("nemo_evaluator.byob")
        byob_logger.addHandler(handler)
        byob_logger.setLevel(logging.INFO)

    # Resolve API key from environment variable
    api_key = None
    if args.api_key_name:
        api_key = os.environ.get(args.api_key_name)

    # Import benchmark using shared logic
    bench = import_benchmark(args.benchmark_module, args.benchmark_name)

    # Load dataset
    dataset = load_dataset(args.dataset, limit=args.limit_samples)

    # Create model_call_fn that wraps the HTTP calls
    timeout = args.timeout_per_sample

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
                timeout=timeout,
            )
        else:
            return call_model_completions(
                url=args.model_url,
                model_id=args.model_id,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                api_key=api_key,
                timeout=timeout,
            )

    # Run evaluation loop using shared logic
    all_scores, all_predictions = run_eval_loop(
        bench=bench,
        dataset=dataset,
        model_call_fn=model_call_fn,
        endpoint_type=args.model_type,
        save_predictions=args.save_predictions,
        fail_on_skip=args.fail_on_skip,
    )

    # Aggregate scores
    results = aggregate_scores(all_scores, args.benchmark_name)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / "byob_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    # Write predictions if requested
    if args.save_predictions and all_predictions:
        from dataclasses import asdict
        predictions_path = Path(args.output_dir) / "byob_predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in all_predictions:
                f.write(json.dumps(asdict(pred)) + "\n")
        print(f"Predictions written to {predictions_path}")

    print(f"Results written to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
