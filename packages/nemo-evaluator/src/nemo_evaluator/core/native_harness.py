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

"""Native harness interface and registry for in-process evaluation execution."""

from typing import Callable, Dict, Optional, Protocol, TypeAlias, runtime_checkable

from nemo_evaluator.api.api_dataclasses import Evaluation, EvaluationResult

# Type alias for the model call function injected into native harnesses
# Signature: (prompt: str, endpoint_type: str) -> response_text: str
ModelCallFn: TypeAlias = Callable[[str, str], str]


@runtime_checkable
class NativeHarness(Protocol):
    """Protocol for evaluation harnesses that execute in-process.

    A NativeHarness receives the fully-merged Evaluation configuration
    and a model_call_fn that routes calls through the adapter pipeline.
    It returns an EvaluationResult directly -- no subprocess, no JSON
    intermediary, no output.py parsing.
    """

    def execute(
        self,
        evaluation: Evaluation,
        model_call_fn: ModelCallFn,
    ) -> EvaluationResult:
        """Execute the evaluation in-process.

        Args:
            evaluation: Fully-merged evaluation configuration.
            model_call_fn: Callable that sends prompts through the adapter
                          pipeline and returns model responses. The harness
                          MUST use this instead of raw HTTP calls to get
                          caching, logging, and token tracking.

        Returns:
            EvaluationResult with tasks and optional groups populated.

        Raises:
            Any exception -- the engine will catch and report.
        """
        ...


# --- Registry ---

_NATIVE_HARNESS_REGISTRY: Dict[str, Callable[[], NativeHarness]] = {}
_discovered: bool = False


def register_native_harness(
    pkg_name_prefix: str, factory: Callable[[], NativeHarness]
) -> None:
    """Register a NativeHarness factory for a package name prefix.

    Args:
        pkg_name_prefix: Package name prefix to match (e.g., "byob_" matches all BYOB packages)
        factory: Callable that returns a NativeHarness instance
    """
    _NATIVE_HARNESS_REGISTRY[pkg_name_prefix] = factory


def get_native_harness(pkg_name: str) -> NativeHarness:
    """Look up and instantiate a NativeHarness for the given package name.

    Matches by longest prefix first.

    Args:
        pkg_name: Package name to match against registered prefixes.

    Returns:
        A NativeHarness instance for the matching prefix.

    Raises:
        ValueError: If no registered harness matches.
    """
    _ensure_discovered()
    # Sort by prefix length descending for most-specific match
    for prefix in sorted(_NATIVE_HARNESS_REGISTRY.keys(), key=len, reverse=True):
        if pkg_name.startswith(prefix):
            return _NATIVE_HARNESS_REGISTRY[prefix]()
    raise ValueError(
        f"No native harness registered for package '{pkg_name}'. "
        f"Registered prefixes: {list(_NATIVE_HARNESS_REGISTRY.keys())}"
    )


def _ensure_discovered() -> None:
    """Lazy discovery of built-in native harness implementations.

    Called once by the engine at evaluation time, NOT at import time.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    # BYOB native harness
    try:
        from nemo_evaluator.byob.native_runner import ByobNativeHarness
        register_native_harness("byob_", ByobNativeHarness)
    except ImportError:
        pass  # BYOB not installed or not available -- fine


# --- ModelCallFn constructors ---


def make_model_call_fn_via_server(
    adapter_url: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str],
) -> ModelCallFn:
    """Create ModelCallFn routing through adapter server proxy.

    Args:
        adapter_url: URL of the adapter server (typically localhost).
        model_id: Model identifier to include in requests.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: Optional API key for authorization header.

    Returns:
        ModelCallFn that makes HTTP calls to the adapter server.
    """
    import requests as req

    def call(prompt: str, endpoint_type: str) -> str:
        if endpoint_type == "chat":
            url = f"{adapter_url}/chat/completions"
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:
            url = f"{adapter_url}/completions"
            payload = {
                "model": model_id,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = req.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        if endpoint_type == "chat":
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return resp.json()["choices"][0]["text"]

    return call


def make_model_call_fn_direct(
    model_url: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str],
) -> ModelCallFn:
    """Create ModelCallFn with direct HTTP calls (no adapter).

    Args:
        model_url: URL of the model endpoint.
        model_id: Model identifier to include in requests.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: Optional API key for authorization header.

    Returns:
        ModelCallFn that makes direct HTTP calls without adapter pipeline.
    """
    from nemo_evaluator.byob.runner import call_model_chat, call_model_completions

    def call(prompt: str, endpoint_type: str) -> str:
        if endpoint_type == "chat":
            return call_model_chat(
                url=model_url,
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        else:
            return call_model_completions(
                url=model_url,
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

    return call
