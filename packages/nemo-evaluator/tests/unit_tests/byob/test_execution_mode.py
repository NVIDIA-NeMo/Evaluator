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

"""Tests for ExecutionMode enum, Evaluation model changes, and native harness registry."""

import pytest
from pydantic import ValidationError

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    EndpointType,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
    ExecutionMode,
)
from nemo_evaluator.core.native_harness import (
    NativeHarness,
    get_native_harness,
    register_native_harness,
    _NATIVE_HARNESS_REGISTRY,
)


@pytest.fixture(autouse=True)
def _clear_native_harness_registry():
    """Clear native harness registry before and after each test."""
    from nemo_evaluator.core import native_harness

    original_registry = _NATIVE_HARNESS_REGISTRY.copy()
    original_discovered = native_harness._discovered

    _NATIVE_HARNESS_REGISTRY.clear()
    native_harness._discovered = False

    yield

    _NATIVE_HARNESS_REGISTRY.clear()
    _NATIVE_HARNESS_REGISTRY.update(original_registry)
    native_harness._discovered = original_discovered


class TestExecutionModeEnum:
    """Tests for ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test ExecutionMode enum has correct string values."""
        assert ExecutionMode.SUBPROCESS.value == "subprocess", \
            f"Expected SUBPROCESS='subprocess', got {ExecutionMode.SUBPROCESS.value!r}"
        assert ExecutionMode.NATIVE.value == "native", \
            f"Expected NATIVE='native', got {ExecutionMode.NATIVE.value!r}"

    def test_execution_mode_from_string(self):
        """Test ExecutionMode can be created from string values."""
        subprocess_mode = ExecutionMode("subprocess")
        assert subprocess_mode == ExecutionMode.SUBPROCESS, \
            f"Expected ExecutionMode.SUBPROCESS, got {subprocess_mode}"

        native_mode = ExecutionMode("native")
        assert native_mode == ExecutionMode.NATIVE, \
            f"Expected ExecutionMode.NATIVE, got {native_mode}"

    def test_execution_mode_invalid_value(self):
        """Test ExecutionMode rejects invalid values."""
        with pytest.raises((ValueError, KeyError)):
            ExecutionMode("invalid")

    def test_execution_mode_case_sensitive(self):
        """Test ExecutionMode values are case-sensitive (lowercase required)."""
        with pytest.raises((ValueError, KeyError)):
            ExecutionMode("SUBPROCESS")
        with pytest.raises((ValueError, KeyError)):
            ExecutionMode("Native")


class TestEvaluationModel:
    """Tests for Evaluation model changes supporting native mode."""

    def test_evaluation_subprocess_mode_with_command(self):
        """Test Evaluation accepts subprocess mode with command template."""
        eval_cfg = Evaluation(
            command="python -m test",
            execution_mode=ExecutionMode.SUBPROCESS,
            framework_name="test_fw",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                type="test.task",
                supported_endpoint_types=["chat"],
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(
                    url="http://localhost",
                    model_id="test-model",
                    type=EndpointType.CHAT,
                )
            ),
        )

        assert eval_cfg.execution_mode == ExecutionMode.SUBPROCESS, \
            f"Expected SUBPROCESS mode, got {eval_cfg.execution_mode}"
        assert eval_cfg.command == "python -m test", \
            f"Command should be preserved, got {eval_cfg.command!r}"

    def test_evaluation_native_mode_no_command(self):
        """Test Evaluation accepts native mode without command."""
        eval_cfg = Evaluation(
            command=None,
            execution_mode=ExecutionMode.NATIVE,
            framework_name="test_fw",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                type="test.task",
                supported_endpoint_types=["chat"],
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(
                    url="http://localhost",
                    model_id="test-model",
                    type=EndpointType.CHAT,
                )
            ),
        )

        assert eval_cfg.execution_mode == ExecutionMode.NATIVE, \
            f"Expected NATIVE mode, got {eval_cfg.execution_mode}"
        assert eval_cfg.command is None, \
            f"Native mode should not require command, got {eval_cfg.command!r}"

    def test_evaluation_default_execution_mode(self):
        """Test Evaluation defaults to subprocess mode for backward compatibility."""
        eval_cfg = Evaluation(
            command="python -m test",
            # execution_mode NOT specified
            framework_name="test_fw",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                type="test.task",
                supported_endpoint_types=["chat"],
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(
                    url="http://localhost",
                    model_id="test-model",
                    type=EndpointType.CHAT,
                )
            ),
        )

        assert eval_cfg.execution_mode == ExecutionMode.SUBPROCESS, \
            f"Default execution_mode should be SUBPROCESS, got {eval_cfg.execution_mode}"

    def test_evaluation_subprocess_missing_command_raises(self):
        """Test Evaluation raises ValidationError when subprocess mode missing command."""
        with pytest.raises(ValidationError, match="command is required"):
            Evaluation(
                command=None,  # Missing command
                execution_mode=ExecutionMode.SUBPROCESS,
                framework_name="test_fw",
                pkg_name="test_pkg",
                config=EvaluationConfig(
                    type="test.task",
                    supported_endpoint_types=["chat"],
                ),
                target=EvaluationTarget(
                    api_endpoint=ApiEndpoint(
                        url="http://localhost",
                        model_id="test-model",
                        type=EndpointType.CHAT,
                    )
                ),
            )

    def test_evaluation_subprocess_empty_command_raises(self):
        """Test Evaluation raises ValidationError when subprocess mode has empty command."""
        with pytest.raises(ValidationError, match="command is required"):
            Evaluation(
                command="",  # Empty command
                execution_mode=ExecutionMode.SUBPROCESS,
                framework_name="test_fw",
                pkg_name="test_pkg",
                config=EvaluationConfig(
                    type="test.task",
                    supported_endpoint_types=["chat"],
                ),
                target=EvaluationTarget(
                    api_endpoint=ApiEndpoint(
                        url="http://localhost",
                        model_id="test-model",
                        type=EndpointType.CHAT,
                    )
                ),
            )

    def test_evaluation_native_mode_with_command_allowed(self):
        """Test native mode can have command field (ignored at runtime)."""
        # Native mode MAY have a command field, but it should not be used
        eval_cfg = Evaluation(
            command="python -m unused",
            execution_mode=ExecutionMode.NATIVE,
            framework_name="test_fw",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                type="test.task",
                supported_endpoint_types=["chat"],
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(
                    url="http://localhost",
                    model_id="test-model",
                    type=EndpointType.CHAT,
                )
            ),
        )

        assert eval_cfg.execution_mode == ExecutionMode.NATIVE
        # Validation should not fail even if command is present for native mode

    def test_evaluation_render_command_fails_for_native_mode(self):
        """Test render_command() raises error when called on native mode evaluation without command."""
        eval_cfg = Evaluation(
            command=None,
            execution_mode=ExecutionMode.NATIVE,
            framework_name="test_fw",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                type="test.task",
                supported_endpoint_types=["chat"],
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(
                    url="http://localhost",
                    model_id="test-model",
                    type=EndpointType.CHAT,
                )
            ),
        )

        with pytest.raises(ValueError, match="Cannot render command"):
            eval_cfg.render_command()

    def test_evaluation_from_dict_subprocess_mode(self):
        """Test Evaluation can be constructed from dict with execution_mode='subprocess'."""
        data = {
            "command": "python -m test",
            "execution_mode": "subprocess",
            "framework_name": "test_fw",
            "pkg_name": "test_pkg",
            "config": {
                "type": "test.task",
                "supported_endpoint_types": ["chat"],
            },
            "target": {
                "api_endpoint": {
                    "url": "http://localhost",
                    "model_id": "test-model",
                    "type": "chat",
                }
            },
        }

        eval_cfg = Evaluation(**data)
        assert eval_cfg.execution_mode == ExecutionMode.SUBPROCESS

    def test_evaluation_from_dict_native_mode(self):
        """Test Evaluation can be constructed from dict with execution_mode='native'."""
        data = {
            "command": None,
            "execution_mode": "native",
            "framework_name": "test_fw",
            "pkg_name": "test_pkg",
            "config": {
                "type": "test.task",
                "supported_endpoint_types": ["chat"],
            },
            "target": {
                "api_endpoint": {
                    "url": "http://localhost",
                    "model_id": "test-model",
                    "type": "chat",
                }
            },
        }

        eval_cfg = Evaluation(**data)
        assert eval_cfg.execution_mode == ExecutionMode.NATIVE

    def test_evaluation_extra_field_forbidden(self):
        """Test Evaluation rejects unknown fields (extra='forbid')."""
        with pytest.raises(ValidationError):
            Evaluation(
                command="python -m test",
                execution_mode=ExecutionMode.SUBPROCESS,
                framework_name="test_fw",
                pkg_name="test_pkg",
                unknown_field="should_fail",  # Unknown field
                config=EvaluationConfig(
                    type="test.task",
                    supported_endpoint_types=["chat"],
                ),
                target=EvaluationTarget(
                    api_endpoint=ApiEndpoint(
                        url="http://localhost",
                        model_id="test-model",
                        type=EndpointType.CHAT,
                    )
                ),
            )


class TestNativeHarnessRegistry:
    """Tests for native harness registry and lookup."""

    def test_register_and_get_native_harness(self):
        """Test registering and retrieving a native harness by prefix."""
        # Create a mock harness class
        class MockHarness:
            def execute(self, evaluation, model_call_fn):
                return None

        # Register with prefix
        register_native_harness("test_", MockHarness)

        # Retrieve by exact prefix match
        harness = get_native_harness("test_pkg_name")
        assert isinstance(harness, MockHarness), \
            f"Expected MockHarness instance, got {type(harness)}"

    def test_get_native_harness_longest_prefix_match(self):
        """Test registry matches longest prefix first for specificity."""
        class HarnessA:
            def execute(self, evaluation, model_call_fn):
                return "A"

        class HarnessB:
            def execute(self, evaluation, model_call_fn):
                return "B"

        # Register two prefixes where one is longer
        register_native_harness("test_", HarnessA)
        register_native_harness("test_specific_", HarnessB)

        # Longer prefix should win
        harness = get_native_harness("test_specific_pkg")
        assert isinstance(harness, HarnessB), \
            "Should match longest prefix 'test_specific_'"

        # Shorter prefix should match when longer does not
        harness = get_native_harness("test_other_pkg")
        assert isinstance(harness, HarnessA), \
            "Should match shorter prefix 'test_'"

    def test_get_native_harness_not_found(self):
        """Test get_native_harness raises ValueError when no prefix matches."""
        register_native_harness("byob_", lambda: None)

        with pytest.raises(ValueError, match="No native harness registered"):
            get_native_harness("unknown_pkg")

    def test_registry_isolation_between_tests(self):
        """Test registry is cleared between tests by autouse fixture."""
        # This test verifies the autouse fixture works correctly
        # At this point, registry should be empty due to fixture
        from nemo_evaluator.core.native_harness import _NATIVE_HARNESS_REGISTRY
        assert len(_NATIVE_HARNESS_REGISTRY) == 0, \
            f"Registry should be empty at test start, got {len(_NATIVE_HARNESS_REGISTRY)} entries"

    def test_registry_multiple_prefixes(self):
        """Test multiple prefixes can be registered and retrieved independently."""
        class HarnessA:
            def execute(self, evaluation, model_call_fn):
                return "A"

        class HarnessB:
            def execute(self, evaluation, model_call_fn):
                return "B"

        register_native_harness("prefix_a_", HarnessA)
        register_native_harness("prefix_b_", HarnessB)

        harness_a = get_native_harness("prefix_a_pkg")
        harness_b = get_native_harness("prefix_b_pkg")

        assert isinstance(harness_a, HarnessA), "Should retrieve HarnessA"
        assert isinstance(harness_b, HarnessB), "Should retrieve HarnessB"

    def test_register_overwrites_existing_prefix(self):
        """Test re-registering same prefix overwrites previous registration."""
        class HarnessOld:
            def execute(self, evaluation, model_call_fn):
                return "old"

        class HarnessNew:
            def execute(self, evaluation, model_call_fn):
                return "new"

        register_native_harness("test_", HarnessOld)
        register_native_harness("test_", HarnessNew)

        harness = get_native_harness("test_pkg")
        assert isinstance(harness, HarnessNew), \
            "Should retrieve most recently registered harness"

    def test_native_harness_protocol_compliance(self):
        """Test that registered harness satisfies NativeHarness protocol."""
        # Create a class that explicitly implements the protocol
        class CompliantHarness:
            def execute(self, evaluation, model_call_fn):
                return None

        # Protocol check
        from nemo_evaluator.core.native_harness import NativeHarness
        assert isinstance(CompliantHarness(), NativeHarness), \
            "CompliantHarness should satisfy NativeHarness protocol"

    def test_registry_factory_pattern(self):
        """Test registry accepts factory functions (callables that return instances)."""
        class TestHarness:
            def __init__(self):
                self.initialized = True

            def execute(self, evaluation, model_call_fn):
                return None

        # Register factory function
        register_native_harness("factory_", TestHarness)

        # Get instance
        harness = get_native_harness("factory_pkg")
        assert isinstance(harness, TestHarness)
        assert harness.initialized is True
