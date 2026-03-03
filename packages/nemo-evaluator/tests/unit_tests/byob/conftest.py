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

"""Shared fixtures for BYOB unit tests."""

from unittest.mock import patch

import pytest

from nemo_evaluator.contrib.byob.dataset import _FETCHER_REGISTRY
from nemo_evaluator.contrib.byob.decorators import clear_registry


@pytest.fixture(autouse=True)
def _clear_byob_registry():
    """Ensure clean BYOB registry state for every test.

    This fixture runs before AND after each test to prevent
    global state leakage between tests. The registry is a
    module-level dict in nemo_evaluator.contrib.byob.decorators.
    """
    clear_registry()
    yield
    clear_registry()


@pytest.fixture(autouse=True)
def _reset_fetcher_registry():
    """Ensure clean DatasetFetcher registry state for every test.

    The fetcher registry is a module-level list in
    nemo_evaluator.contrib.byob.dataset. Save and restore it around
    each test to prevent cross-test pollution from custom
    fetchers registered via register_fetcher().
    """
    saved = _FETCHER_REGISTRY.copy()
    yield
    _FETCHER_REGISTRY.clear()
    _FETCHER_REGISTRY.extend(saved)


@pytest.fixture(autouse=True)
def _mock_pth_file():
    """Prevent tests from writing to the real .pth file in site-packages.

    install_benchmark() calls _ensure_pth_file() which appends lines to
    the real nemo_evaluator_byob.pth. Without this mock, every test run
    pollutes the .pth file with stale /tmp/pytest-* paths.
    """
    with patch(
        "nemo_evaluator.contrib.byob.compiler._ensure_pth_file", return_value=None
    ):
        yield
