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

"""Shared fixtures for nemo-skills integration tests."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def deterministic_client():
    """Mock client with MD5-based deterministic responses.

    Same prompt always produces same response. For golden file tests,
    this fixture is typically customized per-test to return perfect answers
    matching the expected_answer field.

    Response format: 'The answer is \\boxed{<expected_answer>}'
    """
    client = MagicMock()

    async def mock_chat_completion(messages, **kwargs):
        prompt_text = json.dumps(messages)
        hash_val = hashlib.md5(prompt_text.encode()).hexdigest()
        # For golden file tests: return perfect answers
        # The test will override this behavior with sample-specific responses
        return "The answer is \\boxed{42}"

    client.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    return client


@pytest.fixture
def golden_data_dir() -> Path:
    """Return the path to the golden data directory.

    Golden data files are stored at:
    tests/golden_data/nemo_skills/
    """
    tests_root = Path(__file__).parent.parent.parent
    return tests_root / "golden_data" / "nemo_skills"


@pytest.fixture
def ns_plugin_installed(tmp_path):
    """Install nemo_skills namespace package for discovery tests.

    Creates the core_evals.nemo_skills namespace package structure
    in a temporary directory and adds it to sys.path.

    This allows tests to verify that the plugin discovery mechanism
    can find and load the nemo_skills plugin.
    """
    import sys
    import pkgutil

    pkg_dir = tmp_path / "core_evals" / "nemo_skills"
    pkg_dir.mkdir(parents=True)

    # Namespace package markers
    (tmp_path / "core_evals" / "__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
    )
    (pkg_dir / "__init__.py").write_text("")

    # framework.yml stub (minimal valid content)
    framework_yml_content = """
framework:
  name: nemo_skills
  pkg_name: nemo_skills

defaults:
  command: "run_benchmark {{benchmark_name}}"
  execution_mode: subprocess

evaluations:
  - name: gsm8k
    type: nemo_skills.gsm8k
    execution_mode: native
    native_entrypoint: "nemo_evaluator.plugins.nemo_skills.runner:evaluate"
"""
    (pkg_dir / "framework.yml").write_text(framework_yml_content)

    # output.py stub
    (pkg_dir / "output.py").write_text(
        "def parse_output(output_dir):\n    from nemo_evaluator.api.api_dataclasses import EvaluationResult\n    return EvaluationResult()\n"
    )

    original_path = sys.path.copy()
    sys.path.insert(0, str(tmp_path))

    # Clear any cached imports
    for key in list(sys.modules.keys()):
        if key.startswith("core_evals"):
            del sys.modules[key]

    yield tmp_path

    # Cleanup
    sys.path[:] = original_path
    for key in list(sys.modules.keys()):
        if key.startswith("core_evals"):
            del sys.modules[key]
