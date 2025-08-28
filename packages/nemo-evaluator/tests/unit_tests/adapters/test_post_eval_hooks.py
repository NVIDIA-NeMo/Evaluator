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

"""Tests for PostEvalHook functionality."""

import pytest
import requests
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.server import AdapterServer
from nemo_evaluator.adapters.types import AdapterGlobalContext, PostEvalHook
from pydantic import BaseModel, Field


@register_for_adapter(
    name="test_hook", description="Test post-evaluation hook for testing"
)
class SamplePostEvalHook(PostEvalHook):
    """Sample implementation of PostEvalHook."""

    class Params(BaseModel):
        """Test parameters."""

        test_value: str = Field(..., description="Test value")
        output_file: str = Field(..., description="Output file path")

    def __init__(self, params: Params):
        """Initialize the test hook."""
        self.test_value = params.test_value
        self.output_file = params.output_file
        self.called = False

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test post-evaluation hook implementation."""
        self.called = True
        # Write test output to file
        with open(self.output_file, "w") as f:
            f.write(f"{self.test_value}:{context.output_dir}:{context.url}")


class SamplePostEvalHookWithoutParams(PostEvalHook):
    """Sample PostEvalHook without Params class (should fail registration)."""

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test implementation."""
        pass


class SamplePostEvalHookWithWrongInit(PostEvalHook):
    """Sample PostEvalHook with wrong __init__ signature (should fail registration)."""

    class Params(BaseModel):
        """Test parameters."""

        test_value: str = Field(..., description="Test value")

    def __init__(self, wrong_param: str):  # Wrong parameter name
        """Initialize with wrong parameter name."""
        self.test_value = wrong_param

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test implementation."""
        pass


def test_post_eval_hook_registration():
    """Test that PostEvalHook can be registered correctly."""
    registry = InterceptorRegistry.get_instance()
    registry.clear_cache()

    # The SamplePostEvalHook should be registered automatically when imported
    available_hooks = registry.get_post_eval_hooks()
    assert "test_hook" in available_hooks
    assert available_hooks["test_hook"].supports_post_eval_hook()


def test_post_eval_hook_creation():
    """Test that PostEvalHook instances can be created correctly."""
    registry = InterceptorRegistry.get_instance()

    config = {"test_value": "hello", "output_file": "/tmp/test_output.txt"}

    hook = registry._get_or_create_instance("test_hook", config)
    assert isinstance(hook, SamplePostEvalHook)
    assert hook.test_value == "hello"
    assert hook.output_file == "/tmp/test_output.txt"


def test_post_eval_hook_execution(tmpdir):
    """Test that PostEvalHook executes correctly."""
    registry = InterceptorRegistry.get_instance()

    output_file = tmpdir / "test_output.txt"
    config = {"test_value": "test_execution", "output_file": str(output_file)}

    hook = registry._get_or_create_instance("test_hook", config)

    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com"
    )

    # Execute the hook
    hook.post_eval_hook(context)

    # Verify it was called
    assert hook.called

    # Verify output was written
    assert output_file.exists()
    content = output_file.read()
    assert content == f"test_execution:{tmpdir}:http://test.example.com"


def test_post_eval_hook_without_params_fails():
    """Test that PostEvalHook without Params class fails registration."""
    with pytest.raises(ValueError, match="must have a nested Params class"):
        register_for_adapter(
            name="test_hook_no_params", description="Test hook without params"
        )(SamplePostEvalHookWithoutParams)


def test_post_eval_hook_with_wrong_init_fails():
    """Test that PostEvalHook with wrong __init__ signature fails registration."""
    with pytest.raises(
        ValueError, match="must take exactly one parameter named 'params'"
    ):
        register_for_adapter(
            name="test_hook_wrong_init", description="Test hook with wrong init"
        )(SamplePostEvalHookWithWrongInit)


def test_post_eval_hook_discovery():
    """Test that post-eval hooks are discovered correctly."""
    registry = InterceptorRegistry.get_instance()
    registry.reset()

    # Test discovery from modules
    registry.discover_components(
        modules=["tests.unit_tests.adapters.test_post_eval_hooks"]
    )

    post_eval_hooks = registry.get_post_eval_hooks()
    assert "test_hook" in post_eval_hooks


def test_post_eval_hooks_endpoint(tmpdir):
    """Test that the post-eval hooks endpoint works correctly."""
    # Create a test hook that writes to a file
    output_file = tmpdir / "endpoint_test.txt"

    @register_for_adapter(
        name="endpoint_test_hook", description="Test hook for endpoint testing"
    )
    class EndpointTestHook(PostEvalHook):
        """Test hook for endpoint testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.called = False

        def post_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test post-evaluation hook implementation."""
            self.called = True
            with open(self.output_file, "w") as f:
                f.write(f"endpoint_test:{context.output_dir}:{context.url}")

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            {
                "name": "endpoint_test_hook",
                "enabled": True,
                "config": {"output_file": str(output_file)},
            }
        ],
        discovery={"modules": [], "dirs": []},
    )

    # Create and start adapter server
    adapter = AdapterServer(
        api_url="http://test.example.com",
        output_dir=str(tmpdir),
        adapter_config=config,
    )

    # Test the endpoint
    with adapter.app.test_client() as client:
        response = client.post("/adapterserver/run-post-hook")
        assert response.status_code == 200
        assert response.json["status"] == "success"

    # Verify the hook was called and output was written
    assert output_file.exists()
    content = output_file.read()
    assert content == f"endpoint_test:{tmpdir}:http://test.example.com"


def test_integration_post_eval_hooks_flow(tmpdir):
    """Integration test that simulates the complete flow from entrypoint to adapter server."""
    # Create a test hook that writes to a file
    output_file = tmpdir / "integration_test.txt"

    @register_for_adapter(
        name="integration_test_hook", description="Test hook for integration testing"
    )
    class IntegrationTestHook(PostEvalHook):
        """Test hook for integration testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.called = False

        def post_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test post-evaluation hook implementation."""
            self.called = True
            with open(self.output_file, "w") as f:
                f.write(f"integration_test:{context.output_dir}:{context.url}")

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            {
                "name": "integration_test_hook",
                "enabled": True,
                "config": {"output_file": str(output_file)},
            }
        ],
        discovery={"modules": [], "dirs": []},
    )

    # Create and start adapter server in a separate process
    import multiprocessing
    import time

    def run_server():
        adapter = AdapterServer(
            api_url="http://test.example.com",
            output_dir=str(tmpdir),
            adapter_config=config,
        )
        adapter.run()

    # Start server in background process
    server_process = multiprocessing.Process(target=run_server, daemon=True)
    server_process.start()

    try:
        # Wait for server to start
        time.sleep(1)

        # Simulate the entrypoint calling the post-eval hooks endpoint
        post_hook_url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}/adapterserver/run-post-hook"
        response = requests.post(post_hook_url, timeout=30)

        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Verify the hook was called and output was written
        assert output_file.exists()
        content = output_file.read()
        assert content == f"integration_test:{tmpdir}:http://test.example.com"

    finally:
        # Clean up
        server_process.terminate()
        server_process.join(timeout=5)


def test_post_eval_hooks_run_only_once(tmpdir):
    """Test that post-eval hooks run only once when called multiple times."""
    # Create a test hook that tracks how many times it's called
    output_file = tmpdir / "once_test.txt"

    @register_for_adapter(
        name="once_test_hook", description="Test hook for once-only execution"
    )
    class OnceTestHook(PostEvalHook):
        """Test hook that tracks execution count."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.call_count = 0

        def post_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test post-evaluation hook implementation."""
            self.call_count += 1
            with open(self.output_file, "w") as f:
                f.write(
                    f"call_count:{self.call_count}:{context.output_dir}:{context.url}"
                )

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            {
                "name": "once_test_hook",
                "enabled": True,
                "config": {"output_file": str(output_file)},
            }
        ],
        discovery={"modules": [], "dirs": []},
    )

    # Create adapter server
    adapter = AdapterServer(
        api_url="http://test.example.com",
        output_dir=str(tmpdir),
        adapter_config=config,
    )

    # Get the hook instance to check call count
    hook_instance = adapter.post_eval_hooks[0]

    # Call run_post_eval_hooks multiple times
    adapter.run_post_eval_hooks()  # First call
    adapter.run_post_eval_hooks()  # Second call - should be skipped
    adapter.run_post_eval_hooks()  # Third call - should be skipped

    # Verify the hook was called only once
    assert hook_instance.call_count == 1

    # Verify output was written only once
    assert output_file.exists()
    content = output_file.read()
    assert content == f"call_count:1:{tmpdir}:http://test.example.com"


def test_post_eval_hooks_endpoint_run_only_once(tmpdir):
    """Test that the post-eval hooks endpoint also respects the once-only execution guard."""
    # Create a test hook that tracks how many times it's called
    output_file = tmpdir / "endpoint_once_test.txt"

    @register_for_adapter(
        name="endpoint_once_test_hook",
        description="Test hook for endpoint once-only execution",
    )
    class EndpointOnceTestHook(PostEvalHook):
        """Test hook that tracks execution count for endpoint testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.call_count = 0

        def post_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test post-evaluation hook implementation."""
            self.call_count += 1
            with open(self.output_file, "w") as f:
                f.write(
                    f"endpoint_call_count:{self.call_count}:{context.output_dir}:{context.url}"
                )

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            {
                "name": "endpoint_once_test_hook",
                "enabled": True,
                "config": {"output_file": str(output_file)},
            }
        ],
        discovery={"modules": [], "dirs": []},
    )

    # Create adapter server
    adapter = AdapterServer(
        api_url="http://test.example.com",
        output_dir=str(tmpdir),
        adapter_config=config,
    )

    # Get the hook instance to check call count
    hook_instance = adapter.post_eval_hooks[0]

    # Test the endpoint multiple times
    with adapter.app.test_client() as client:
        # First call
        response1 = client.post("/adapterserver/run-post-hook")
        assert response1.status_code == 200
        assert response1.json["status"] == "success"

        # Second call - should be skipped
        response2 = client.post("/adapterserver/run-post-hook")
        assert response2.status_code == 200
        assert response2.json["status"] == "success"

        # Third call - should be skipped
        response3 = client.post("/adapterserver/run-post-hook")
        assert response3.status_code == 200
        assert response3.json["status"] == "success"

    # Verify the hook was called only once
    assert hook_instance.call_count == 1

    # Verify output was written only once
    assert output_file.exists()
    content = output_file.read()
    assert content == f"endpoint_call_count:1:{tmpdir}:http://test.example.com"
