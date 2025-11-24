# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root conftest for all tests - ensures interceptors are registered."""

import pytest

from nemo_evaluator.adapters.registry import InterceptorRegistry


@pytest.fixture(scope="session", autouse=True)
def register_all_interceptors():
    """Register all built-in interceptors before any tests run.

    This runs once at the start of the entire test session and ensures
    all interceptors are available in the registry for all tests.
    """
    registry = InterceptorRegistry.get_instance()
    # Trigger discovery of built-in modules
    registry.discover_components(modules=[], dirs=[])
    yield
