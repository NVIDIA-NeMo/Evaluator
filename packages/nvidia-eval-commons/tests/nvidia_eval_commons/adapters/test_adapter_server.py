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

from typing import Any, Generator
from unittest.mock import patch

import pytest
import requests
from nvidia_eval_commons.adapters.adapter_config import AdapterConfig
from nvidia_eval_commons.adapters.server import (
    AdapterServer,
    spawn_adapter_server,
    wait_for_server,
)


@pytest.fixture
def adapter_server(
    fake_openai_endpoint,
    tmp_path,
) -> Generator[AdapterServer, Any, Any]:
    api_url = "http://localhost:3300/v1/chat/completions"
    output_dir = tmp_path
    adapter_config = AdapterConfig(
        interceptors=[
            dict(
                name="caching",
                enabled=True,
                config={
                    "cache_dir": str(tmp_path / "cache"),
                    "reuse_cached_responses": True,
                    "save_requests": False,
                    "save_responses": True,
                },
            ),
            dict(
                name="endpoint",
                enabled=True,
                config={},
            ),
            dict(
                name="response_logging",
                enabled=True,
                config={},
            ),
            dict(
                name="reasoning",
                enabled=True,
                config={"end_reasoning_token": "</think>"},
            ),
        ]
    )
    p = spawn_adapter_server(api_url, output_dir, adapter_config)
    yield p
    p.terminate()
    p.join(timeout=5)


def test_adapter_server_post_request(adapter_server, capfd):
    url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

    # Wait for server to be ready
    wait_for_server("localhost", 3825)

    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
    }
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "choices" in response.json()
    assert len(response.json()["choices"]) > 0
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "choices" in response.json()
    assert len(response.json()["choices"]) > 0
    assert "</think>" not in response.json()["choices"][0]["message"]["content"]


def test_adapter_server_with_discovery_config():
    """Test that AdapterServer properly passes discovery configuration to registry."""
    adapter_config = AdapterConfig(
        discovery={
            "modules": ["test.module", "another.module"],
            "dirs": ["/path/to/plugins", "/another/path"],
        },
        interceptors=[
            dict(name="endpoint", enabled=True, config={}),
        ],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ) as mock_discover:
        # This triggers discovery
        _ = AdapterServer(
            api_url="http://localhost:3000",
            output_dir="/tmp",
            adapter_config=adapter_config,
        )

        mock_discover.assert_called_once_with(
            modules=["test.module", "another.module"],
            dirs=["/path/to/plugins", "/another/path"],
        )


def test_adapter_server_without_discovery_config():
    """Test that AdapterServer works without discovery configuration."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="endpoint", enabled=True, config={}),
        ]
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ) as mock_discover:
        # This triggers discovery
        _ = AdapterServer(
            api_url="http://localhost:3000",
            output_dir="/tmp",
            adapter_config=adapter_config,
        )

        mock_discover.assert_called_once_with(modules=[], dirs=[])


def test_adapter_server_with_interceptors():
    """Test that the adapter server works with interceptors."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="endpoint", enabled=True, config={}),
        ]
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ) as mock_discover:
        # This triggers discovery
        _ = AdapterServer(
            api_url="http://localhost:3000",
            output_dir="/tmp",
            adapter_config=adapter_config,
        )

        mock_discover.assert_called_once_with(modules=[], dirs=[])


# Tests for the new validation functionality
def test_adapter_server_validation_empty_config():
    """Test that AdapterServer raises RuntimeError when no enabled interceptors or
    post-eval hooks are configured."""
    adapter_config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ):
        with pytest.raises(RuntimeError) as exc_info:
            AdapterServer(
                api_url="http://localhost:3000",
                output_dir="/tmp",
                adapter_config=adapter_config,
            )

        error_msg = str(exc_info.value)
        assert "Adapter server cannot start" in error_msg
        assert "No enabled interceptors or post-eval hooks found" in error_msg
        assert "Configured interceptors: []" in error_msg
        assert "Configured post-eval hooks: []" in error_msg


def test_adapter_server_validation_disabled_interceptors():
    """Test that AdapterServer raises RuntimeError when all interceptors are
    disabled."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="caching", enabled=False, config={}),
            dict(name="endpoint", enabled=False, config={}),
        ],
        post_eval_hooks=[],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ):
        with pytest.raises(RuntimeError) as exc_info:
            AdapterServer(
                api_url="http://localhost:3000",
                output_dir="/tmp",
                adapter_config=adapter_config,
            )

        error_msg = str(exc_info.value)
        assert "Adapter server cannot start" in error_msg
        assert "No enabled interceptors or post-eval hooks found" in error_msg
        assert "Configured interceptors: ['caching', 'endpoint']" in error_msg
        assert "Configured post-eval hooks: []" in error_msg


def test_adapter_server_validation_disabled_post_eval_hooks():
    """Test that AdapterServer raises RuntimeError when all post-eval hooks are disabled."""
    adapter_config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            dict(name="report", enabled=False, config={}),
            dict(name="post_eval_report", enabled=False, config={}),
        ],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ):
        with pytest.raises(RuntimeError) as exc_info:
            AdapterServer(
                api_url="http://localhost:3000",
                output_dir="/tmp",
                adapter_config=adapter_config,
            )

        error_msg = str(exc_info.value)
        assert "Adapter server cannot start" in error_msg
        assert "No enabled interceptors or post-eval hooks found" in error_msg
        assert "Configured interceptors: []" in error_msg
        assert "Configured post-eval hooks: ['report', 'post_eval_report']" in error_msg


def test_adapter_server_validation_mixed_disabled():
    """Test that AdapterServer raises RuntimeError when both interceptors and post-eval hooks are all disabled."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="caching", enabled=False, config={}),
            dict(name="endpoint", enabled=False, config={}),
        ],
        post_eval_hooks=[
            dict(name="report", enabled=False, config={}),
            dict(name="post_eval_report", enabled=False, config={}),
        ],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ):
        with pytest.raises(RuntimeError) as exc_info:
            AdapterServer(
                api_url="http://localhost:3000",
                output_dir="/tmp",
                adapter_config=adapter_config,
            )

        error_msg = str(exc_info.value)
        assert "Adapter server cannot start" in error_msg
        assert "No enabled interceptors or post-eval hooks found" in error_msg
        assert "Configured interceptors: ['caching', 'endpoint']" in error_msg
        assert "Configured post-eval hooks: ['report', 'post_eval_report']" in error_msg


def test_adapter_server_validation_with_enabled_interceptor():
    """Test that AdapterServer starts successfully when at least one interceptor is enabled."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="caching", enabled=False, config={}),
            dict(name="endpoint", enabled=True, config={}),
        ],
        post_eval_hooks=[],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ):
        # Should not raise an exception
        server = AdapterServer(
            api_url="http://localhost:3000",
            output_dir="/tmp",
            adapter_config=adapter_config,
        )
        assert server is not None


def test_adapter_server_validation_with_enabled_post_eval_hook():
    """Test that AdapterServer starts successfully when at least one post-eval hook is enabled."""
    adapter_config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            dict(name="report", enabled=False, config={}),
            dict(name="post_eval_report", enabled=True, config={}),
        ],
    )

    with patch(
        "nvidia_eval_commons.adapters.registry.InterceptorRegistry.discover_components"
    ):
        # Should not raise an exception
        server = AdapterServer(
            api_url="http://localhost:3000",
            output_dir="/tmp",
            adapter_config=adapter_config,
        )
        assert server is not None


def test_spawn_adapter_server_validation_empty_config():
    """Test that spawn_adapter_server returns None when no enabled interceptors or post-eval hooks are configured."""
    adapter_config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[],
    )

    result = spawn_adapter_server(
        api_url="http://localhost:3000",
        output_dir="/tmp",
        adapter_config=adapter_config,
    )

    assert result is None


def test_spawn_adapter_server_validation_disabled_interceptors():
    """Test that spawn_adapter_server returns None when all interceptors are disabled."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="caching", enabled=False, config={}),
            dict(name="endpoint", enabled=False, config={}),
        ],
        post_eval_hooks=[],
    )

    result = spawn_adapter_server(
        api_url="http://localhost:3000",
        output_dir="/tmp",
        adapter_config=adapter_config,
    )

    assert result is None


def test_spawn_adapter_server_validation_disabled_post_eval_hooks():
    """Test that spawn_adapter_server returns None when all post-eval hooks are disabled."""
    adapter_config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            dict(name="report", enabled=False, config={}),
            dict(name="post_eval_report", enabled=False, config={}),
        ],
    )

    result = spawn_adapter_server(
        api_url="http://localhost:3000",
        output_dir="/tmp",
        adapter_config=adapter_config,
    )

    assert result is None


def test_spawn_adapter_server_validation_with_enabled_interceptor():
    """Test that spawn_adapter_server returns a process when at least one interceptor is enabled."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="caching", enabled=False, config={}),
            dict(name="endpoint", enabled=True, config={}),
        ],
        post_eval_hooks=[],
    )

    with patch("multiprocessing.get_context") as mock_context:
        mock_process = mock_context.return_value.Process.return_value
        mock_process.start.return_value = None
        mock_process.terminate.return_value = None
        mock_process.join.return_value = None

        with patch("nvidia_eval_commons.adapters.server.wait_for_server") as mock_wait:
            mock_wait.return_value = True

            result = spawn_adapter_server(
                api_url="http://localhost:3000",
                output_dir="/tmp",
                adapter_config=adapter_config,
            )

            assert result is not None
            assert result == mock_process


def test_spawn_adapter_server_validation_with_enabled_post_eval_hook():
    """Test that spawn_adapter_server returns a process when at least one post-eval hook is enabled."""
    adapter_config = AdapterConfig(
        interceptors=[],
        post_eval_hooks=[
            dict(name="report", enabled=False, config={}),
            dict(name="post_eval_report", enabled=True, config={}),
        ],
    )

    with patch("multiprocessing.get_context") as mock_context:
        mock_process = mock_context.return_value.Process.return_value
        mock_process.start.return_value = None
        mock_process.terminate.return_value = None
        mock_process.join.return_value = None

        with patch("nvidia_eval_commons.adapters.server.wait_for_server") as mock_wait:
            mock_wait.return_value = True

            result = spawn_adapter_server(
                api_url="http://localhost:3000",
                output_dir="/tmp",
                adapter_config=adapter_config,
            )

            assert result is not None
            assert result == mock_process


def test_spawn_adapter_server_validation_server_fails_to_start():
    """Test that spawn_adapter_server raises RuntimeError when server fails to start."""
    adapter_config = AdapterConfig(
        interceptors=[
            dict(name="endpoint", enabled=True, config={}),
        ],
        post_eval_hooks=[],
    )

    with patch("multiprocessing.get_context") as mock_context:
        mock_process = mock_context.return_value.Process.return_value
        mock_process.start.return_value = None
        mock_process.terminate.return_value = None
        mock_process.join.return_value = None

        with patch("nvidia_eval_commons.adapters.server.wait_for_server") as mock_wait:
            mock_wait.return_value = False

            with pytest.raises(RuntimeError) as exc_info:
                spawn_adapter_server(
                    api_url="http://localhost:3000",
                    output_dir="/tmp",
                    adapter_config=adapter_config,
                )

            error_msg = str(exc_info.value)
            assert "Adapter server failed to start" in error_msg
            assert "localhost:3825" in error_msg


def test_adapter_server_process_uses_original_url_in_args():
    """Test that AdapterServerProcess passes original_url as first argument to _run_adapter_server."""
    from nvidia_eval_commons.adapters.server import AdapterServerProcess
    from nvidia_eval_commons.api.api_dataclasses import (
        ApiEndpoint,
        Evaluation,
        EvaluationConfig,
        EvaluationTarget,
    )

    original_url = "http://original-api.example.com:8080/v1/chat/completions"
    evaluation = Evaluation(
        config=EvaluationConfig(output_dir="/tmp/test"),
        target=EvaluationTarget(api_endpoint=ApiEndpoint(url=original_url)),
        pkg_name="test_package",
        command="test_command",
        framework_name="test_framework",
    )

    mock_config = AdapterConfig(
        interceptors=[
            dict(
                name="endpoint",
                enabled=True,
                config={},
            ),
        ]
    )

    with (
        patch(
            "nvidia_eval_commons.adapters.server.AdapterConfig.get_validated_config",
            return_value=mock_config,
        ),
        patch(
            "nvidia_eval_commons.adapters.server.multiprocessing.get_context"
        ) as mock_context,
        patch("nvidia_eval_commons.adapters.server.wait_for_server", return_value=True),
    ):
        # Mock the Process to capture the arguments

        adapter_process = AdapterServerProcess(evaluation)

        with adapter_process:
            # Verify that the process was created with original_url as first argument
            mock_context.return_value.Process.assert_called_once()
            call_args = mock_context.return_value.Process.call_args
            # The first argument to _run_adapter_server should be original_url
            assert call_args[1]["args"][0] == original_url


def test_evaluate_function_url_replacement():
    """Test that the evaluate() function correctly handles URL replacement in the full evaluation flow."""
    from nvidia_eval_commons.adapters.server import AdapterServer
    from nvidia_eval_commons.api.api_dataclasses import (
        ApiEndpoint,
        EvaluationConfig,
        EvaluationTarget,
    )
    from nvidia_eval_commons.core.evaluate import evaluate

    # Original NVIDIA API URL
    original_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/eb017772-7aa0-4bb8-9aff-9d9dce85734c"

    # Create evaluation configs
    eval_config = EvaluationConfig(output_dir="/tmp/test")
    target_config = EvaluationTarget(api_endpoint=ApiEndpoint(url=original_url))

    # Capture the dict that gets passed to yaml.dump
    captured_dict = None

    def mock_yaml_dump(data, *args, **kwargs):
        nonlocal captured_dict
        captured_dict = data
        return "mocked_yaml"

    # Mock all the necessary components
    with (
        patch(
            "nvidia_eval_commons.core.evaluate.validate_configuration"
        ) as mock_validate,
        patch("nvidia_eval_commons.core.evaluate.prepare_output_directory"),
        patch("nvidia_eval_commons.core.evaluate.AdapterServerProcess"),
        patch("nvidia_eval_commons.core.evaluate.run_command") as mock_run_command,
        patch("nvidia_eval_commons.core.evaluate.parse_output") as mock_parse_output,
        patch("nvidia_eval_commons.core.evaluate.monitor_memory_usage") as mock_monitor,
        patch("builtins.open", create=True),
        patch("os.path.join") as mock_join,
        patch(
            "nvidia_eval_commons.core.evaluate.yaml.dump", side_effect=mock_yaml_dump
        ),
        patch("nvidia_eval_commons.core.evaluate.logger"),
    ):
        # Mock the evaluation object with a command template that uses the URL
        mock_evaluation = mock_validate.return_value
        mock_evaluation.command = "helm-generate-dynamic-model-configs --base-url {{ target.api_endpoint.url }} --model-name test-model"
        mock_evaluation.config.output_dir = "/tmp/test"
        mock_evaluation.pkg_name = "test_package"
        mock_evaluation.framework_name = "test_framework"

        # Mock the evaluation result
        mock_result = mock_parse_output.return_value
        mock_monitor.return_value = (mock_result, {})

        # Mock the adapter process context manager to simulate URL replacement
        def context_manager_side_effect():
            # Simulate what happens inside the context manager
            # The URL should be replaced with localhost:3825
            expected_localhost_url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"

            # Mock the render_command to return the command with replaced URL
            def mock_render_command():
                return f"helm-generate-dynamic-model-configs --base-url {expected_localhost_url} --model-name test-model"

            mock_evaluation.render_command = mock_render_command

            # Execute the command
            cmd = mock_evaluation.render_command()
            mock_run_command(cmd, verbose=True)
            return mock_result

        mock_monitor.side_effect = lambda func, **kwargs: (
            context_manager_side_effect(),
            {},
        )

        # Mock file operations
        mock_join.return_value = "/tmp/test/eval_factory_metrics.json"

        # Call the evaluate function
        result = evaluate(eval_config, target_config)

        # Examine the captured dict
        assert captured_dict is not None, "No dict was captured"
        print(f"Captured dict: {captured_dict}")

        # Check the command field in the result dictionary
        command_value = captured_dict.get("command")
        print(f"Command field value: {command_value}")
        print(f"Command field type: {type(command_value)}")

        # Verify that the command contains the localhost URL, not the original NVIDIA URL
        expected_localhost_url = f"http://{AdapterServer.DEFAULT_ADAPTER_HOST}:{AdapterServer.DEFAULT_ADAPTER_PORT}"
        assert expected_localhost_url in command_value, (
            f"Expected localhost URL '{expected_localhost_url}' in command, got '{command_value}'"
        )
        assert original_url not in command_value, (
            f"Original URL '{original_url}' should not be in command, got '{command_value}'"
        )

        # Verify the specific replacement in the command
        expected_command_part = f"--base-url {expected_localhost_url}"
        assert expected_command_part in command_value, (
            f"Expected command part '{expected_command_part}' not found in '{command_value}'"
        )

        # Verify that run_command was called with the rendered command
        mock_run_command.assert_called_once()
        call_args = mock_run_command.call_args
        assert call_args[0][0] == command_value, (
            f"Expected command '{command_value}', got '{call_args[0][0]}'"
        )
        assert call_args[1]["verbose"]

        # Verify that the evaluation result is returned
        assert result == mock_result
