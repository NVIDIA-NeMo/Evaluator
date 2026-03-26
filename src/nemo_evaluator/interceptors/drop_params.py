"""Interceptor that strips named params from every outgoing LLM request.

Useful when a provider does not accept certain params (e.g. ``max_tokens``,
``max_completion_tokens``) that LiteLLM or the agent injects by default.

Configuration (via YAML)::

    proxy:
      interceptors:
        - name: drop_params
          config:
            params: [max_tokens, max_completion_tokens]
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class Interceptor(CustomLogger):
    """Per-request hook that removes named params from the request body."""

    def __init__(self, *, params: list[str] | None = None) -> None:
        super().__init__()
        self._params = set(params or [])
        self._lock = threading.Lock()
        self._logged_once = False

    def _strip(self, data: dict) -> None:
        for p in self._params:
            data.pop(p, None)

        with self._lock:
            if not self._logged_once:
                logger.info("drop_params: %s", sorted(self._params))
                self._logged_once = True

    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        self._strip(data)
        return data
