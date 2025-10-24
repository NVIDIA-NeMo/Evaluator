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

"""Tests for PreEvalHook functionality."""

import pytest
from pydantic import BaseModel, Field

from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.server import AdapterServer
from nemo_evaluator.adapters.types import AdapterGlobalContext, PreEvalHook


@register_for_adapter(
    name="test_pre_hook", description="Test pre-evaluation hook for testing"
)
class SamplePreEvalHook(PreEvalHook):
    """Sample implementation of PreEvalHook."""

    class Params(BaseModel):
        """Test parameters."""

        test_value: str = Field(..., description="Test value")
        output_file: str = Field(..., description="Output file path")

    def __init__(self, params: Params):
        """Initialize the test hook."""
        self.test_value = params.test_value
        self.output_file = params.output_file
        self.called = False

    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test pre-evaluation hook implementation."""
        self.called = True
        # Write test output to file
        with open(self.output_file, "w") as f:
            f.write(f"{self.test_value}:{context.output_dir}:{context.url}")


class SamplePreEvalHookWithoutParams(PreEvalHook):
    """Sample PreEvalHook without Params class (should fail registration)."""

    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test implementation."""
        pass


class SamplePreEvalHookWithWrongInit(PreEvalHook):
    """Sample PreEvalHook with wrong __init__ signature (should fail registration)."""

    class Params(BaseModel):
        """Test parameters."""

        test_value: str = Field(..., description="Test value")

    def __init__(self, wrong_param: str):  # Wrong parameter name
        """Initialize with wrong parameter name."""
        self.test_value = wrong_param

    def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Test implementation."""
        pass


def test_pre_eval_hook_registration():
    """Test that PreEvalHook can be registered correctly."""
    registry = InterceptorRegistry.get_instance()
    registry.clear_cache()

    # The SamplePreEvalHook should be registered automatically when imported
    available_hooks = registry.get_pre_eval_hooks()
    assert "test_pre_hook" in available_hooks
    assert available_hooks["test_pre_hook"].supports_pre_eval_hook()


def test_pre_eval_hook_creation():
    """Test that PreEvalHook instances can be created correctly."""
    registry = InterceptorRegistry.get_instance()

    config = {"test_value": "hello", "output_file": "/tmp/test_output.txt"}

    hook = registry._get_or_create_instance("test_pre_hook", config)
    assert isinstance(hook, SamplePreEvalHook)
    assert hook.test_value == "hello"
    assert hook.output_file == "/tmp/test_output.txt"


def test_pre_eval_hook_execution(tmpdir):
    """Test that PreEvalHook executes correctly."""
    registry = InterceptorRegistry.get_instance()

    output_file = tmpdir / "test_output.txt"
    config = {"test_value": "test_execution", "output_file": str(output_file)}

    hook = registry._get_or_create_instance("test_pre_hook", config)

    context = AdapterGlobalContext(
        output_dir=str(tmpdir), url="http://test.example.com"
    )

    # Execute the hook
    hook.pre_eval_hook(context)

    # Verify it was called
    assert hook.called

    # Verify output was written
    assert output_file.exists()
    content = output_file.read()
    assert content == f"test_execution:{tmpdir}:http://test.example.com"


def test_pre_eval_hook_without_params_fails():
    """Test that PreEvalHook without Params class fails registration."""
    with pytest.raises(ValueError, match="must have a nested Params class"):
        register_for_adapter(
            name="test_pre_hook_no_params", description="Test hook without params"
        )(SamplePreEvalHookWithoutParams)


def test_pre_eval_hook_with_wrong_init_fails():
    """Test that PreEvalHook with wrong __init__ signature fails registration."""
    with pytest.raises(
        ValueError, match="must take exactly one parameter named 'params'"
    ):
        register_for_adapter(
            name="test_pre_hook_wrong_init", description="Test hook with wrong init"
        )(SamplePreEvalHookWithWrongInit)


def test_pre_eval_hook_discovery():
    """Test that pre-eval hooks are discovered correctly."""
    registry = InterceptorRegistry.get_instance()
    registry.reset()

    # Test discovery from modules
    registry.discover_components(
        modules=["tests.unit_tests.adapters.test_pre_eval_hooks"]
    )

    pre_eval_hooks = registry.get_pre_eval_hooks()
    assert "test_pre_hook" in pre_eval_hooks


def test_pre_eval_hooks_endpoint(tmpdir):
    """Test that the pre-eval hooks endpoint works correctly.
    
    Note: In normal operation, pre-hooks run in the parent process before server start.
    This endpoint is kept for manual/testing purposes.
    """
    # Create a test hook that writes to a file
    output_file = tmpdir / "endpoint_test.txt"

    @register_for_adapter(
        name="endpoint_test_pre_hook", description="Test hook for endpoint testing"
    )
    class EndpointTestPreHook(PreEvalHook):
        """Test hook for endpoint testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.called = False

        def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test pre-evaluation hook implementation."""
            self.called = True
            with open(self.output_file, "w") as f:
                f.write(f"endpoint_test:{context.output_dir}:{context.url}")

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        pre_eval_hooks=[
            {
                "name": "endpoint_test_pre_hook",
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
        response = client.post("/adapterserver/run-pre-hook")
        assert response.status_code == 200
        assert response.json["status"] == "success"

    # Verify the hook was called and output was written
    assert output_file.exists()
    content = output_file.read()
    assert content == f"endpoint_test:{tmpdir}:http://test.example.com"


def test_integration_pre_eval_hooks_flow(tmpdir):
    """Integration test that simulates the complete flow with pre-hooks in parent process."""
    # Create a test hook that writes to a file
    output_file = tmpdir / "integration_test.txt"

    @register_for_adapter(
        name="integration_test_pre_hook",
        description="Test hook for integration testing",
    )
    class IntegrationTestPreHook(PreEvalHook):
        """Test hook for integration testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.called = False

        def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test pre-evaluation hook implementation."""
            self.called = True
            with open(self.output_file, "w") as f:
                f.write(f"integration_test:{context.output_dir}:{context.url}")

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        pre_eval_hooks=[
            {
                "name": "integration_test_pre_hook",
                "enabled": True,
                "config": {"output_file": str(output_file)},
            }
        ],
        discovery={"modules": [], "dirs": []},
    )

    # Simulate running pre-hooks in parent process (as done in AdapterServerProcess.__enter__)
    registry = InterceptorRegistry.get_instance()
    registry.discover_components(modules=[], dirs=[])
    
    # Build and execute hooks
    pre_eval_hooks = []
    for hook_config in config.pre_eval_hooks:
        if hook_config.enabled:
            hook = registry._get_or_create_instance(hook_config.name, hook_config.config)
            pre_eval_hooks.append(hook)
    
    context = AdapterGlobalContext(
        output_dir=str(tmpdir),
        url="http://test.example.com",
    )
    
    for hook in pre_eval_hooks:
        hook.pre_eval_hook(context)
    
    # Verify the hook was called and output was written
    assert output_file.exists()
    content = output_file.read()
    assert content == f"integration_test:{tmpdir}:http://test.example.com"


def test_pre_eval_hooks_run_only_once(tmpdir):
    """Test that pre-eval hooks run only once when called multiple times."""
    # Create a test hook that tracks how many times it's called
    output_file = tmpdir / "once_test.txt"

    @register_for_adapter(
        name="once_test_pre_hook", description="Test hook for once-only execution"
    )
    class OnceTestPreHook(PreEvalHook):
        """Test hook that tracks execution count."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.call_count = 0

        def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test pre-evaluation hook implementation."""
            self.call_count += 1
            with open(self.output_file, "w") as f:
                f.write(
                    f"call_count:{self.call_count}:{context.output_dir}:{context.url}"
                )

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        pre_eval_hooks=[
            {
                "name": "once_test_pre_hook",
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
    hook_instance = adapter.pre_eval_hooks[0]

    # Call run_pre_eval_hooks multiple times
    adapter.run_pre_eval_hooks()  # First call
    adapter.run_pre_eval_hooks()  # Second call - should be skipped
    adapter.run_pre_eval_hooks()  # Third call - should be skipped

    # Verify the hook was called only once
    assert hook_instance.call_count == 1

    # Verify output was written only once
    assert output_file.exists()
    content = output_file.read()
    assert content == f"call_count:1:{tmpdir}:http://test.example.com"


def test_pre_eval_hooks_endpoint_run_only_once(tmpdir):
    """Test that the pre-eval hooks endpoint also respects the once-only execution guard.
    
    Note: In normal operation, pre-hooks run in the parent process before server start.
    This endpoint is kept for manual/testing purposes.
    """
    # Create a test hook that tracks how many times it's called
    output_file = tmpdir / "endpoint_once_test.txt"

    @register_for_adapter(
        name="endpoint_once_test_pre_hook",
        description="Test hook for endpoint once-only execution",
    )
    class EndpointOnceTestPreHook(PreEvalHook):
        """Test hook that tracks execution count for endpoint testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file
            self.call_count = 0

        def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test pre-evaluation hook implementation."""
            self.call_count += 1
            with open(self.output_file, "w") as f:
                f.write(
                    f"endpoint_call_count:{self.call_count}:{context.output_dir}:{context.url}"
                )

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        pre_eval_hooks=[
            {
                "name": "endpoint_once_test_pre_hook",
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
    hook_instance = adapter.pre_eval_hooks[0]

    # Test the endpoint multiple times
    with adapter.app.test_client() as client:
        # First call
        response1 = client.post("/adapterserver/run-pre-hook")
        assert response1.status_code == 200
        assert response1.json["status"] == "success"

        # Second call - should be skipped
        response2 = client.post("/adapterserver/run-pre-hook")
        assert response2.status_code == 200
        assert response2.json["status"] == "success"

        # Third call - should be skipped
        response3 = client.post("/adapterserver/run-pre-hook")
        assert response3.status_code == 200
        assert response3.json["status"] == "success"

    # Verify the hook was called only once
    assert hook_instance.call_count == 1

    # Verify output was written only once
    assert output_file.exists()
    content = output_file.read()
    assert content == f"endpoint_call_count:1:{tmpdir}:http://test.example.com"


def test_pre_eval_hooks_build_correctly(tmpdir):
    """Test that pre-eval hooks are built correctly in the server."""
    output_file = tmpdir / "build_test.txt"

    @register_for_adapter(
        name="build_test_pre_hook", description="Test hook for build testing"
    )
    class BuildTestPreHook(PreEvalHook):
        """Test hook for build testing."""

        class Params(BaseModel):
            """Test parameters."""

            output_file: str = Field(..., description="Output file path")

        def __init__(self, params: Params):
            """Initialize the test hook."""
            self.output_file = params.output_file

        def pre_eval_hook(self, context: AdapterGlobalContext) -> None:
            """Test pre-evaluation hook implementation."""
            with open(self.output_file, "w") as f:
                f.write("build_test")

    # Create adapter config with the test hook
    config = AdapterConfig(
        interceptors=[],
        pre_eval_hooks=[
            {
                "name": "build_test_pre_hook",
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

    # Verify the pre-eval hooks were built
    assert len(adapter.pre_eval_hooks) == 1
    assert isinstance(adapter.pre_eval_hooks[0], BuildTestPreHook)

