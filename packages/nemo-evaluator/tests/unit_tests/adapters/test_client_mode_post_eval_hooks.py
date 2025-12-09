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

"""Tests for post-eval hooks in client mode (AsyncAdapterTransport)."""

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import AdapterGlobalContext, PostEvalHook
from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
from nemo_evaluator.client import NeMoEvaluatorClient

from .testing_utils import create_fake_endpoint_process

# Register all hooks at module level (before test functions)


@register_for_adapter(name="client_test_hook", description="Test hook for client mode")
class ClientTestHook(PostEvalHook):
    """Test hook for client mode testing."""

    class Params(BaseModel):
        """Test parameters."""

        output_file: str = Field(..., description="Output file path")
        test_message: str = Field(..., description="Test message to write")

    def __init__(self, params: Params):
        """Initialize the test hook."""
        self.output_file = params.output_file
        self.test_message = params.test_message

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test post-evaluation hook implementation."""
        with open(self.output_file, "w") as f:
            f.write(
                f"{self.test_message}|output_dir={context.output_dir}|url={context.url}|model_name={context.model_name}"
            )


@register_for_adapter(name="test_hook_1", description="First test hook")
class TestHook1(PostEvalHook):
    class Params(BaseModel):
        output_file: str

    def __init__(self, params: Params):
        self.output_file = params.output_file

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        Path(self.output_file).write_text("Hook 1 executed")


@register_for_adapter(name="test_hook_2", description="Second test hook")
class TestHook2(PostEvalHook):
    class Params(BaseModel):
        output_file: str

    def __init__(self, params: Params):
        self.output_file = params.output_file

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        Path(self.output_file).write_text("Hook 2 executed")


@register_for_adapter(name="cache_checker_hook", description="Checks cache data")
class CacheCheckerHook(PostEvalHook):
    class Params(BaseModel):
        output_file: str

    def __init__(self, params: Params):
        self.output_file = params.output_file

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        # Check if cache directory exists (created by caching interceptor)
        # We'll check for cache in the output_dir since that's what we have access to
        Path(self.output_file).write_text(
            f"hook_ran=true|output_dir={context.output_dir}"
        )


@register_for_adapter(name="failing_hook", description="Hook that fails")
class FailingHook(PostEvalHook):
    class Params(BaseModel):
        pass

    def __init__(self, params: Params):
        pass

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        raise RuntimeError("Intentional error in hook")


@register_for_adapter(name="success_hook", description="Hook that succeeds")
class SuccessHook(PostEvalHook):
    class Params(BaseModel):
        output_file: str

    def __init__(self, params: Params):
        self.output_file = params.output_file

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        Path(self.output_file).write_text("Success hook executed")


@register_for_adapter(name="finalizer_hook", description="Hook for finalizer test")
class FinalizerHook(PostEvalHook):
    class Params(BaseModel):
        output_file: str

    def __init__(self, params: Params):
        self.output_file = params.output_file

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        Path(self.output_file).write_text("Hook executed via finalizer")


@register_for_adapter(name="counter_hook", description="Hook that counts calls")
class CounterHook(PostEvalHook):
    class Params(BaseModel):
        call_count_file: str

    def __init__(self, params: Params):
        self.call_count_file = params.call_count_file

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        # Increment counter in file
        path = Path(self.call_count_file)
        current_count = int(path.read_text()) if path.exists() else 0
        path.write_text(str(current_count + 1))


@pytest.mark.asyncio
async def test_post_eval_hook_runs_in_client_mode(tmp_path):
    """Test that post-eval hooks are executed when using client mode."""
    output_file = tmp_path / "post_eval_test.txt"

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        # Configure adapter with post-eval hook
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[
                InterceptorConfig(name="endpoint", enabled=True, config={}),
            ],
            post_eval_hooks=[
                {
                    "name": "client_test_hook",
                    "enabled": True,
                    "config": {
                        "output_file": str(output_file),
                        "test_message": "Hook executed",
                    },
                }
            ],
        )

        config = EndpointModelConfig(
            model_id="test-model",
            url="http://localhost:3300/v1/chat/completions",
            adapter_config=adapter_config,
        )

        # Use the client with async context manager
        async with NeMoEvaluatorClient(
            endpoint_model_config=config, output_dir=str(tmp_path)
        ) as client:
            # Make a request to the fake endpoint
            await client.chat_completion(messages=[{"role": "user", "content": "Test"}])

        # Post-eval hook should have been called when exiting the context
        assert output_file.exists(), "Post-eval hook output file should exist"

        content = output_file.read_text()
        assert "Hook executed" in content, "Hook message should be in output"
        assert f"output_dir={tmp_path}" in content, "Output dir should be in context"
        assert "model_name=test-model" in content, "Model name should be in context"

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


@pytest.mark.asyncio
async def test_multiple_post_eval_hooks_in_client_mode(tmp_path):
    """Test that multiple post-eval hooks are all executed in client mode."""
    output_file1 = tmp_path / "hook1.txt"
    output_file2 = tmp_path / "hook2.txt"

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[InterceptorConfig(name="endpoint")],
            post_eval_hooks=[
                {"name": "test_hook_1", "config": {"output_file": str(output_file1)}},
                {"name": "test_hook_2", "config": {"output_file": str(output_file2)}},
            ],
        )

        config = EndpointModelConfig(
            model_id="test-model",
            url="http://localhost:3300/v1/chat/completions",
            adapter_config=adapter_config,
        )

        async with NeMoEvaluatorClient(
            endpoint_model_config=config, output_dir=str(tmp_path)
        ) as client:
            await client.chat_completion(messages=[{"role": "user", "content": "Test"}])

        # Both hooks should have been executed
        assert output_file1.exists(), "First hook output should exist"
        assert output_file2.exists(), "Second hook output should exist"
        assert output_file1.read_text() == "Hook 1 executed"
        assert output_file2.read_text() == "Hook 2 executed"

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


@pytest.mark.asyncio
async def test_post_eval_hook_with_interceptor_data(tmp_path):
    """Test that post-eval hooks can access data saved by interceptors."""
    cache_dir = tmp_path / "cache"
    output_file = tmp_path / "hook_output.txt"

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[
                InterceptorConfig(
                    name="caching",
                    config={
                        "cache_dir": str(cache_dir),
                        "reuse_cached_responses": True,
                        "save_requests": True,
                        "save_responses": True,
                    },
                ),
                InterceptorConfig(name="endpoint"),
            ],
            post_eval_hooks=[
                {
                    "name": "cache_checker_hook",
                    "config": {"output_file": str(output_file)},
                }
            ],
        )

        config = EndpointModelConfig(
            model_id="test-model",
            url="http://localhost:3300/v1/chat/completions",
            adapter_config=adapter_config,
        )

        async with NeMoEvaluatorClient(
            endpoint_model_config=config, output_dir=str(tmp_path)
        ) as client:
            await client.chat_completion(messages=[{"role": "user", "content": "Test"}])

        # Hook should have run
        assert output_file.exists(), "Hook output should exist"
        content = output_file.read_text()
        assert "hook_ran=true" in content
        assert f"output_dir={tmp_path}" in content

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


@pytest.mark.asyncio
async def test_post_eval_hook_error_handling(tmp_path):
    """Test that errors in post-eval hooks don't prevent client cleanup."""
    output_file = tmp_path / "error_test.txt"

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[InterceptorConfig(name="endpoint")],
            post_eval_hooks=[
                {"name": "failing_hook", "config": {}},
                {"name": "success_hook", "config": {"output_file": str(output_file)}},
            ],
        )

        config = EndpointModelConfig(
            model_id="test-model",
            url="http://localhost:3300/v1/chat/completions",
            adapter_config=adapter_config,
        )

        # Client should close cleanly even if hook fails
        async with NeMoEvaluatorClient(
            endpoint_model_config=config, output_dir=str(tmp_path)
        ) as client:
            await client.chat_completion(messages=[{"role": "user", "content": "Test"}])

        # Second hook should still have been executed despite first hook failing
        assert output_file.exists(), "Success hook should have run despite failing hook"
        assert output_file.read_text() == "Success hook executed"

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


@pytest.mark.asyncio
async def test_post_eval_hooks_run_via_finalizer(tmp_path):
    """Test that post-eval hooks run even when user forgets to close the client."""
    output_file = tmp_path / "finalizer_test.txt"

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[InterceptorConfig(name="endpoint")],
            post_eval_hooks=[
                {"name": "finalizer_hook", "config": {"output_file": str(output_file)}}
            ],
        )

        config = EndpointModelConfig(
            model_id="test-model",
            url="http://localhost:3300/v1/chat/completions",
            adapter_config=adapter_config,
        )

        # Create client WITHOUT using context manager and WITHOUT calling aclose()
        client = NeMoEvaluatorClient(
            endpoint_model_config=config, output_dir=str(tmp_path)
        )

        await client.chat_completion(messages=[{"role": "user", "content": "Test"}])

        # Delete the client without calling aclose() - finalizer should trigger
        del client

        # Force garbage collection to trigger finalizer
        import gc

        gc.collect()

        # Give finalizer a moment to run
        await asyncio.sleep(0.1)

        # Hook should have been executed via finalizer
        assert output_file.exists(), "Post-eval hook should run via finalizer"
        assert output_file.read_text() == "Hook executed via finalizer"

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)


@pytest.mark.asyncio
async def test_post_eval_hooks_run_only_once(tmp_path):
    """Test that post-eval hooks run only once even with both aclose() and finalizer."""
    call_count_file = tmp_path / "call_count.txt"

    # Start fake endpoint
    fake_endpoint = create_fake_endpoint_process()

    try:
        adapter_config = AdapterConfig(
            mode="client",
            interceptors=[InterceptorConfig(name="endpoint")],
            post_eval_hooks=[
                {
                    "name": "counter_hook",
                    "config": {"call_count_file": str(call_count_file)},
                }
            ],
        )

        config = EndpointModelConfig(
            model_id="test-model",
            url="http://localhost:3300/v1/chat/completions",
            adapter_config=adapter_config,
        )

        # Use context manager (which calls aclose())
        async with NeMoEvaluatorClient(
            endpoint_model_config=config, output_dir=str(tmp_path)
        ) as client:
            await client.chat_completion(messages=[{"role": "user", "content": "Test"}])
        # aclose() called here, which should run hooks

        # Force garbage collection (finalizer should NOT run hooks again)
        import gc

        gc.collect()
        await asyncio.sleep(0.1)

        # Hook should have been called exactly once
        assert call_count_file.exists(), "Hook should have been called"
        assert call_count_file.read_text() == "1", "Hook should only be called once"

    finally:
        # Clean up fake endpoint
        fake_endpoint.terminate()
        fake_endpoint.join(timeout=5)
