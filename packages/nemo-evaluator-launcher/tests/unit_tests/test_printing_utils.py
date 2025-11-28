# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for printing_utils.py functionality."""

import os
import sys
from unittest.mock import patch


class TestPrintingUtilsColors:
    """Test cases for printing utils color functionality."""

    def test_colors_disabled_when_not_tty(self):
        """Test that colors are disabled when stdout is not a TTY."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            # Reload module to pick up the TTY check
            import importlib

            import nemo_evaluator_launcher.common.printing_utils as printing_utils

            importlib.reload(printing_utils)

            output = printing_utils.red("test")
            # Should not contain ANSI color codes
            assert "\033[" not in output
            assert output == "test"

    def test_colors_enabled_when_tty(self):
        """Test that colors are enabled when stdout is a TTY."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            with patch.dict(os.environ, {}, clear=True):
                # Reload module to pick up the TTY check
                import importlib

                import nemo_evaluator_launcher.common.printing_utils as printing_utils

                importlib.reload(printing_utils)

                output = printing_utils.red("test")
                # Should contain ANSI color codes
                assert "\033[" in output

    def test_env_var_disables_colors(self, monkeypatch):
        """Test that environment variable can disable colors even when TTY."""
        monkeypatch.setenv("NEMO_EVALUATOR_DISABLE_COLOR", "1")
        with patch.object(sys.stdout, "isatty", return_value=True):
            # Reload module to pick up the env var
            import importlib

            import nemo_evaluator_launcher.common.printing_utils as printing_utils

            importlib.reload(printing_utils)

            output = printing_utils.red("test")
            # Should not contain ANSI color codes due to env var
            assert "\033[" not in output
            assert output == "test"

        monkeypatch.delenv("NEMO_EVALUATOR_DISABLE_COLOR", raising=False)
