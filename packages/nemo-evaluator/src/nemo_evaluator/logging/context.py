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

"""Request context utilities for logging and tracing."""

import uuid
from contextlib import contextmanager
from typing import Optional

import structlog

from nemo_evaluator.logging.utils import get_logger


def bind_request_id(request_id: Optional[str] = None) -> str:
    """
    Bind a request ID to the current logging context.

    Args:
        request_id: Optional request ID. If None, a new UUID will be generated.

    Returns:
        The request ID that was bound to the context.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    # Bind the request ID to structlog context variables
    import structlog

    structlog.contextvars.bind_contextvars(request_id=request_id)
    return request_id


@contextmanager
def request_context(request_id: Optional[str] = None):
    """
    Context manager for binding and clearing request context.

    Args:
        request_id: Optional request ID. If None, a new UUID will be generated.

    Yields:
        The request ID that was bound to the context.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    # Bind the request ID to the context
    structlog.contextvars.bind_contextvars(request_id=request_id)

    try:
        yield request_id
    finally:
        # Clear the context variables when exiting
        structlog.contextvars.clear_contextvars()


def get_current_request_id() -> Optional[str]:
    """
    Get the current request ID from the context variables.

    Returns:
        The current request ID if set, None otherwise.
    """
    try:
        context_vars = structlog.contextvars.get_contextvars()
        return context_vars.get("request_id")
    except Exception:
        return None


def get_bound_logger(request_id: Optional[str] = None, logger_name: str = None):
    """
    Get a logger with the request ID bound to it.

    Args:
        request_id: Optional request ID. If None, a new UUID will be generated.
        logger_name: Optional logger name. If None, uses the calling module.

    Returns:
        A structlog logger with the request ID bound to it.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    # Get the logger
    if logger_name:
        logger = get_logger(logger_name)
    else:
        logger = get_logger()

    # Bind the request ID directly to the logger
    return logger.bind(request_id=request_id)
