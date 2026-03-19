"""Base class for custom NEL interceptors.

Extends ``litellm.integrations.custom_logger.CustomLogger`` with
evaluation-specific context.  LiteLLM is imported lazily so this module
can be loaded without the ``proxy`` extra installed.
"""
from __future__ import annotations

from typing import Any


def _require_custom_logger() -> type:
    try:
        from litellm.integrations.custom_logger import CustomLogger
        return CustomLogger
    except ImportError:
        raise ImportError(
            "Interceptors require the litellm package. "
            "Install with: pip install nemo-evaluator[proxy]"
        ) from None


class BaseInterceptor(_require_custom_logger()):  # type: ignore[misc]
    """Base interceptor for NEL evaluations.

    Subclass and override any of the ``CustomLogger`` hooks — the most
    common ones are:

    * ``async_log_success_event(kwargs, response_obj, start_time, end_time)``
    * ``async_log_failure_event(kwargs, response_obj, start_time, end_time)``
    * ``async_log_pre_api_call(model, messages, kwargs)``

    The full list is in ``litellm.integrations.custom_logger.CustomLogger``.
    """

    def __init__(self, **context: Any) -> None:
        super().__init__()
        self.context = context
