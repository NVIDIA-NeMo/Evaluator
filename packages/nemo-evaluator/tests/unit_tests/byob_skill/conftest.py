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

"""Shared fixtures for BYOB skill tests."""

import os

import pytest

from nemo_evaluator.contrib.byob.decorators import (
    clear_registry,
)

# Resolve paths - REPO_ROOT points to packages/nemo-evaluator/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEMPLATE_DIR = os.path.join(REPO_ROOT, "examples", "byob", "templates")
SKILL_PROMPT_PATH = os.path.join(REPO_ROOT, ".claude", "commands", "byob.md")


@pytest.fixture(autouse=True)
def clean_registry_fixture():
    """Ensure clean benchmark registry for every test.

    This fixture runs before AND after each test to prevent global state
    leakage between tests. The registry is a module-level dict in
    nemo_evaluator.contrib.byob.decorators.
    """
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def template_dir():
    """Path to template directory."""
    return TEMPLATE_DIR


@pytest.fixture
def skill_prompt_path():
    """Path to skill prompt file."""
    return SKILL_PROMPT_PATH
