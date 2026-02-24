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

"""Unit tests for BYOB defaults module.

Verifies that centralized default constants are importable, have correct
values and types, and are actually consumed by runner.py and compiler.py.
"""

import inspect

from nemo_evaluator.contrib.byob.defaults import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
)


class TestDefaultConstants:
    """Tests that defaults.py constants are importable with correct values and types."""

    def test_default_max_tokens_value(self):
        """Validate DEFAULT_MAX_TOKENS is 4096 and an integer."""
        assert DEFAULT_MAX_TOKENS == 4096, (
            f"Expected DEFAULT_MAX_TOKENS == 4096, got {DEFAULT_MAX_TOKENS}"
        )
        assert isinstance(DEFAULT_MAX_TOKENS, int), (
            f"Expected DEFAULT_MAX_TOKENS to be int, got {type(DEFAULT_MAX_TOKENS).__name__}"
        )

    def test_default_temperature_value(self):
        """Validate DEFAULT_TEMPERATURE is 0.0 and a float."""
        assert DEFAULT_TEMPERATURE == 0.0, (
            f"Expected DEFAULT_TEMPERATURE == 0.0, got {DEFAULT_TEMPERATURE}"
        )
        assert isinstance(DEFAULT_TEMPERATURE, float), (
            f"Expected DEFAULT_TEMPERATURE to be float, got {type(DEFAULT_TEMPERATURE).__name__}"
        )

    def test_default_timeout_value(self):
        """Validate DEFAULT_TIMEOUT_SECONDS is 120.0 and a float."""
        assert DEFAULT_TIMEOUT_SECONDS == 120.0, (
            f"Expected DEFAULT_TIMEOUT_SECONDS == 120.0, got {DEFAULT_TIMEOUT_SECONDS}"
        )
        assert isinstance(DEFAULT_TIMEOUT_SECONDS, float), (
            f"Expected DEFAULT_TIMEOUT_SECONDS to be float, "
            f"got {type(DEFAULT_TIMEOUT_SECONDS).__name__}"
        )

    def test_runner_imports_defaults(self):
        """Validate runner.py call_model_chat/completions default params match defaults.py.

        Inspects the function signatures to confirm that the default values
        for temperature, max_tokens, and timeout originate from defaults.py.
        """
        from nemo_evaluator.contrib.byob.runner import call_model_chat, call_model_completions

        for fn in (call_model_chat, call_model_completions):
            sig = inspect.signature(fn)

            temp_default = sig.parameters["temperature"].default
            assert temp_default == DEFAULT_TEMPERATURE, (
                f"{fn.__name__} temperature default is {temp_default}, "
                f"expected DEFAULT_TEMPERATURE ({DEFAULT_TEMPERATURE})"
            )

            max_tok_default = sig.parameters["max_tokens"].default
            assert max_tok_default == DEFAULT_MAX_TOKENS, (
                f"{fn.__name__} max_tokens default is {max_tok_default}, "
                f"expected DEFAULT_MAX_TOKENS ({DEFAULT_MAX_TOKENS})"
            )

            timeout_default = sig.parameters["timeout"].default
            assert timeout_default == DEFAULT_TIMEOUT_SECONDS, (
                f"{fn.__name__} timeout default is {timeout_default}, "
                f"expected DEFAULT_TIMEOUT_SECONDS ({DEFAULT_TIMEOUT_SECONDS})"
            )

    def test_compiler_imports_defaults(self):
        """Validate compiler FDF blocks use DEFAULT_MAX_TOKENS and DEFAULT_TEMPERATURE.

        Reads the _build_fdf source to confirm that the defaults dict references
        the constants from defaults.py rather than hard-coded magic numbers.
        """
        from nemo_evaluator.contrib.byob.compiler import _build_fdf

        source = inspect.getsource(_build_fdf)

        assert "DEFAULT_MAX_TOKENS" in source, (
            "_build_fdf source should reference DEFAULT_MAX_TOKENS from defaults.py"
        )
        assert "DEFAULT_TEMPERATURE" in source, (
            "_build_fdf source should reference DEFAULT_TEMPERATURE from defaults.py"
        )

    def test_aggregation_importable(self):
        """Validate aggregate_scores is importable from both aggregation and runner modules.

        The runner re-exports aggregate_scores for backward compatibility, so
        both import paths should resolve to the same function object.
        """
        from nemo_evaluator.contrib.byob.aggregation import aggregate_scores as from_aggregation
        from nemo_evaluator.contrib.byob.runner import aggregate_scores as from_runner

        assert from_aggregation is from_runner, (
            "aggregate_scores imported from aggregation and runner should be the same function"
        )
