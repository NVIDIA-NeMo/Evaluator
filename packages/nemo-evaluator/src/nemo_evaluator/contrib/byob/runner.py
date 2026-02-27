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
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nemo_evaluator.contrib.byob.aggregation import (
    aggregate_scores,  # noqa: F401 — re-export for backward compat
)
from nemo_evaluator.contrib.byob.dataset import load_dataset
from nemo_evaluator.contrib.byob.defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOP_P,
)
from nemo_evaluator.contrib.byob.eval_logic import import_benchmark, run_eval_loop
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


def create_session(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> requests.Session:
    """Create a requests.Session with connection pooling and retry logic.

    Retries on 429 (rate-limit), 500, 502, 503, 504.
    Does NOT retry on 400 (client error).

    Args:
        max_retries: Maximum number of retries per request.
        backoff_factor: Multiplier for exponential backoff between retries.

    Returns:
        Configured requests.Session.
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def call_model_chat(
    url: str,
    model_id: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    """Call OpenAI-compatible chat completions endpoint.

    Args:
        url: Base URL for model endpoint.
        model_id: Model identifier.
        prompt: Prompt text to send as user message.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Top-p (nucleus) sampling parameter. None omits from payload.
        api_key: Optional Bearer token for Authorization header.
        timeout: Request timeout in seconds.
        session: Optional requests.Session for connection pooling.
        system_prompt: Optional system message prepended to the conversation.

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

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    http = session or requests
    response = http.post(endpoint, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def call_model_completions(
    url: str,
    model_id: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    top_p: Optional[float] = None,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    """Call OpenAI-compatible completions endpoint.

    Args:
        url: Base URL for model endpoint.
        model_id: Model identifier.
        prompt: Prompt text.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Top-p (nucleus) sampling parameter. None omits from payload.
        api_key: Optional Bearer token for Authorization header.
        timeout: Request timeout in seconds.
        session: Optional requests.Session for connection pooling.
        system_prompt: Optional system prompt prepended to the prompt text.

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

    full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt

    payload = {
        "model": model_id,
        "prompt": full_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    http = session or requests
    response = http.post(endpoint, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    return response.json()["choices"][0]["text"]


def check_requirements(requirements: List[str]) -> List[str]:
    """Validate that declared requirements are installed.

    Uses ``importlib.metadata.version()`` and ``packaging.version.Version``
    to verify each requirement is installed and meets version constraints.

    Args:
        requirements: List of PEP 508 requirement strings
            (e.g. ``["numpy>=1.20", "pandas"]``).

    Returns:
        List of warning messages for missing or incompatible packages.
        Empty list means all requirements are satisfied.
    """
    import importlib.metadata
    import re

    warnings = []
    for req_str in requirements:
        req_str = req_str.strip()
        if not req_str:
            continue

        # Parse package name and optional version specifier
        # Handles: "numpy", "numpy>=1.20", "numpy>=1.20,<2.0", "numpy==1.24"
        match = re.match(r"^([a-zA-Z0-9_.-]+)\s*(.*)?$", req_str)
        if not match:
            warnings.append(f"Cannot parse requirement: {req_str}")
            continue

        pkg_name = match.group(1)
        version_spec = (match.group(2) or "").strip()

        try:
            installed_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            warnings.append(f"Missing package: {pkg_name} (required: {req_str})")
            continue

        if version_spec:
            try:
                from packaging.specifiers import SpecifierSet
                from packaging.version import Version

                spec = SpecifierSet(version_spec)
                if Version(installed_version) not in spec:
                    warnings.append(
                        f"Version mismatch: {pkg_name}=={installed_version} "
                        f"does not satisfy {req_str}"
                    )
            except Exception:
                # packaging not available or parse error — skip version check
                pass

    return warnings


def ensure_requirements(requirements: List[str]) -> None:
    """Check declared requirements and install any that are missing.

    First checks which packages are missing or have version mismatches,
    then installs them via ``pip install``.

    Args:
        requirements: List of PEP 508 requirement strings.
    """
    import subprocess

    warnings = check_requirements(requirements)
    if not warnings:
        return

    to_install = []
    for req_str in requirements:
        req_str = req_str.strip()
        if not req_str:
            continue
        # Reject strings that look like pip flags (security: prevent argument injection)
        if req_str.startswith("-"):
            raise ValueError(f"Invalid requirement (looks like a flag): {req_str}")
        # Check if this specific requirement triggered a warning
        import re

        match = re.match(r"^([a-zA-Z0-9_.-]+)", req_str)
        if not match:
            continue
        pkg_name = match.group(1)
        if any(pkg_name in w for w in warnings):
            to_install.append(req_str)

    if not to_install:
        return

    logger.info(
        "Installing missing benchmark requirements",
        packages=to_install,
    )
    # Try uv pip first (for uv-managed environments), fall back to pip
    import shutil

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "pip", "install", *to_install]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *to_install]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Requirements installed successfully", packages=to_install)
    except subprocess.CalledProcessError as e:
        logger.error(
            "Failed to install requirements",
            packages=to_install,
            cmd=" ".join(cmd),
            stderr=e.stderr.strip() if e.stderr else "",
        )
        raise RuntimeError(
            f"Failed to install benchmark requirements {to_install}: "
            f"{e.stderr.strip() if e.stderr else 'unknown error'}"
        ) from e


def _create_session_model_call_fn(
    args: argparse.Namespace,
    api_key: Optional[str],
    session: requests.Session,
) -> Callable:
    """Create a model call function backed by raw HTTP requests.

    Args:
        args: Parsed CLI arguments.
        api_key: Resolved API key value.
        session: Configured requests.Session.

    Returns:
        Callable matching model_call_fn(prompt, endpoint_type, *, system_prompt=None, timeout=None) -> str.
    """
    default_timeout = (
        args.request_timeout
        if args.request_timeout is not None
        else args.timeout_per_sample
    )

    def model_call_fn(
        prompt: str,
        endpoint_type: str,
        *,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        effective_timeout = timeout if timeout is not None else default_timeout
        if endpoint_type == "chat":
            return call_model_chat(
                url=args.model_url,
                model_id=args.model_id,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                api_key=api_key,
                timeout=effective_timeout,
                session=session,
                system_prompt=system_prompt,
            )
        return call_model_completions(
            url=args.model_url,
            model_id=args.model_id,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            api_key=api_key,
            timeout=effective_timeout,
            session=session,
            system_prompt=system_prompt,
        )

    return model_call_fn


def create_client_model_call_fn(
    args: argparse.Namespace,
    api_key: Optional[str],
) -> tuple:
    """Create a model call function backed by NeMoEvaluatorClient.

    Lazily imports ``nemo_evaluator.client`` to avoid hard dependency on
    ``openai``, ``tenacity``, and ``tqdm``.  Callers should catch
    ``ImportError`` and fall back to raw HTTP requests.

    Args:
        args: Parsed CLI arguments (model_url, model_id, temperature, max_tokens).
        api_key: Resolved API key value.

    Returns:
        Tuple of (model_call_fn, cleanup_fn).

    Raises:
        ImportError: If nemo_evaluator.client is not available.
    """
    import asyncio

    from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
    from nemo_evaluator.client import NeMoEvaluatorClient

    endpoint_config = EndpointModelConfig(
        url=args.model_url,
        model_id=args.model_id,
        api_key_name=args.api_key_name,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
    client = NeMoEvaluatorClient(endpoint_config, output_dir=args.output_dir)

    def model_call_fn(
        prompt: str,
        endpoint_type: str,
        *,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        if endpoint_type == "chat":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            result = asyncio.run(client.chat_completion(messages))
        else:
            full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
            result = asyncio.run(client.completion(full_prompt))
        return result

    def cleanup_fn():
        try:
            asyncio.run(client.aclose())
        except Exception:
            pass

    return model_call_fn, cleanup_fn


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
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
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
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds for each model call (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Raise error on any skipped sample (missing field or model error)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of HTTP retries per model call (default: 3)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.5,
        help="Backoff factor for exponential retry delay (default: 0.5)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of concurrent evaluation threads (default: 1, sequential)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation (default: 1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p (nucleus) sampling parameter (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: None, falls back to --timeout-per-sample)",
    )

    args = parser.parse_args()

    # Resolve API key from environment variable
    api_key = None
    if args.api_key_name:
        api_key = os.environ.get(args.api_key_name)

    # Import benchmark using shared logic
    bench = import_benchmark(args.benchmark_module, args.benchmark_name)

    # Install missing requirements declared by the benchmark
    if bench.requirements:
        ensure_requirements(bench.requirements)

    # Load dataset
    dataset = load_dataset(
        args.dataset,
        limit=args.limit_samples,
        field_mapping=bench.field_mapping,
    )

    # Create model call function — try NeMoEvaluatorClient, fall back to raw HTTP
    cleanup_fn = None
    try:
        model_call_fn, cleanup_fn = create_client_model_call_fn(args, api_key)
        logger.info(
            "Using NeMoEvaluatorClient for model calls",
            model_url=args.model_url,
            model_id=args.model_id,
        )
    except ImportError as e:
        logger.info(
            "NeMoEvaluatorClient not available, using raw HTTP requests",
            reason=str(e),
        )
        session = create_session(
            max_retries=args.max_retries,
            backoff_factor=args.retry_backoff,
        )
        model_call_fn = _create_session_model_call_fn(args, api_key, session)

    # Run evaluation loop (with n-repeats support)
    all_scores: List[Dict] = []
    all_predictions: List = []

    for repeat_idx in range(args.n_repeats):
        if args.n_repeats > 1:
            logger.info("Starting repeat", repeat=repeat_idx + 1, total=args.n_repeats)

        scores, predictions = run_eval_loop(
            bench=bench,
            dataset=dataset,
            model_call_fn=model_call_fn,
            endpoint_type=args.model_type,
            save_predictions=args.save_predictions,
            fail_on_skip=args.fail_on_skip,
            parallelism=args.parallelism,
            request_timeout=(
                args.request_timeout
                if args.request_timeout is not None
                else args.timeout_per_sample
            ),
        )

        # Offset sample_id and add _repeat metadata for repeats
        if args.n_repeats > 1:
            offset = repeat_idx * len(dataset)
            for pred in predictions:
                pred.sample_id = pred.sample_id + offset
                pred.metadata = {**pred.metadata, "_repeat": repeat_idx}

        all_scores.extend(scores)
        all_predictions.extend(predictions)

    # Clean up client resources if applicable
    if cleanup_fn is not None:
        cleanup_fn()

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
