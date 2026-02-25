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

import importlib
import os
import sys
from typing import Callable

import pytest

from nemo_evaluator.contrib.byob.decorators import (
    clear_registry,
    get_registered_benchmarks,
)


# Resolve paths - REPO_ROOT points to packages/nemo-evaluator/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEMPLATE_DIR = os.path.join(REPO_ROOT, "examples", "byob", "templates")
SKILL_PROMPT_PATH = os.path.join(REPO_ROOT, ".claude", "commands", "byob.md")

TEMPLATES = [
    "math_reasoning",
    "multichoice",
    "open_qa",
    "classification",
    "safety",
    "code_generation",
]


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


def import_scorer(template_name: str) -> Callable:
    """Import a template's scorer function by name.

    Returns the scorer callable from the first registered benchmark.

    Args:
        template_name: Name of the template (e.g., "math_reasoning")

    Returns:
        The scorer function from the template's registered benchmark.
    """
    clear_registry()
    template_path = os.path.join(TEMPLATE_DIR, f"{template_name}.py")
    parent = os.path.dirname(template_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Force reload to ensure fresh registration
    if template_name in sys.modules:
        importlib.reload(sys.modules[template_name])
    else:
        importlib.import_module(template_name)

    benchmarks = get_registered_benchmarks()
    if not benchmarks:
        raise ValueError(f"No benchmarks registered after importing {template_name}")

    bench = list(benchmarks.values())[0]
    return bench.scorer_fn


@pytest.fixture
def scorer_import_helper():
    """Fixture that provides the import_scorer helper function."""
    return import_scorer
