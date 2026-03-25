"""Interceptor that enables reasoning/thinking for vLLM-served models.

Injects ``chat_template_kwargs``, ``skip_special_tokens``, and other
extra body fields into every outgoing LLM request so that vLLM activates
the model's chain-of-thought reasoning mode.

Configuration (via YAML)::

    interceptors:
      - enable_reasoning:
          enable_thinking: true
          skip_special_tokens: false
          drop_params: [max_tokens, max_completion_tokens]

This interceptor is designed for the LiteLLM **proxy** path — it hooks
into ``async_pre_call_hook`` which receives the raw request ``data`` dict
before the proxy forwards it upstream.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class Interceptor(CustomLogger):
    """Injects reasoning-related params into every LLM request.

    Constructor kwargs (from YAML config):

    * ``enable_thinking`` — passed inside ``extra_body.chat_template_kwargs``
    * ``skip_special_tokens`` — added to ``extra_body``
    * ``drop_params`` — list of param names to remove from the request
    * ``extra_body`` — arbitrary dict merged into the request's ``extra_body``
    """

    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        skip_special_tokens: bool = False,
        drop_params: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._enable_thinking = enable_thinking
        self._skip_special_tokens = skip_special_tokens
        self._drop_params = set(drop_params or [])
        self._extra_body = extra_body or {}
        self._lock = threading.Lock()
        self._logged_once = False

    def _inject(self, data: dict) -> None:
        extra = data.setdefault("extra_body", {})

        if self._enable_thinking:
            ctk = extra.setdefault("chat_template_kwargs", {})
            ctk["enable_thinking"] = True

        extra["skip_special_tokens"] = self._skip_special_tokens

        for p in self._drop_params:
            data.pop(p, None)

        extra.update(self._extra_body)

        with self._lock:
            if not self._logged_once:
                logger.info(
                    "enable_reasoning: injected thinking=%s, skip_special_tokens=%s, drop=%s",
                    self._enable_thinking,
                    self._skip_special_tokens,
                    sorted(self._drop_params),
                )
                self._logged_once = True

    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        self._inject(data)
        return data
