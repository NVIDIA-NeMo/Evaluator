"""Example interceptor: log token usage per LLM call."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class Interceptor(CustomLogger):
    """Logs input/output token counts for every successful LLM call."""

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return
        model = kwargs.get("model", "?")
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        elapsed_ms = (end_time - start_time).total_seconds() * 1000
        logger.info(
            "[log_tokens] model=%s  prompt=%d  completion=%d  total=%d  elapsed=%.0fms",
            model, prompt_tokens, completion_tokens,
            prompt_tokens + completion_tokens, elapsed_ms,
        )
