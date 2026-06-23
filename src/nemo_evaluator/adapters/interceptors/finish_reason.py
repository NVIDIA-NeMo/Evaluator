# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Reconcile ``finish_reason`` with token usage and surface empty generations.

Some serving stacks report ``finish_reason="stop"`` even when a generation
exhausted its token budget — for reasoning models this happens when the
budget is consumed entirely inside the thinking phase, leaving empty content
and no terminating token. The ``"stop"`` label then hides a truncated,
unusable generation from every downstream consumer.

This interceptor stashes the request's token cap, and on the response relabels
such cases to ``"length"`` (when the generated token count reached the cap)
and logs a warning for empty generations so the failure is visible in the run
logs. It never alters scores or token counts.
"""

from __future__ import annotations

import logging

from nemo_evaluator.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)

logger = logging.getLogger(__name__)

_CAP_REQUEST_KEYS = ("max_completion_tokens", "max_tokens")
_CAP_CTX_KEY = "finish_reason_token_cap"


class Interceptor(RequestInterceptor, ResponseInterceptor):
    """Relabel cap-truncated ``stop`` responses and warn on empty generations.

    Must run in the request phase (before the upstream call) so the token cap
    is captured for the matching response.
    """

    stream_safe = False
    best_effort = True

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        body = req.body
        if isinstance(body, dict):
            for key in _CAP_REQUEST_KEYS:
                cap = body.get(key)
                if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
                    req.ctx.extra[_CAP_CTX_KEY] = cap
                    break
        return req

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        body = resp.body
        if not isinstance(body, dict):
            return resp

        cap = resp.ctx.extra.get(_CAP_CTX_KEY)
        generated = self._generated_tokens(body)
        truncated = isinstance(cap, int) and isinstance(generated, int) and generated >= cap

        for choice in body.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if truncated and choice.get("finish_reason") == "stop":
                choice["finish_reason"] = "length"
                logger.warning(
                    "finish_reason relabeled stop->length: generated_tokens=%d reached max_tokens=%d",
                    generated,
                    cap,
                )
            if self._is_empty(choice):
                logger.warning(
                    "empty generation (finish_reason=%s, generated_tokens=%s)",
                    choice.get("finish_reason"),
                    generated,
                )
        return resp

    @staticmethod
    def _generated_tokens(body: dict) -> int | None:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None
        completion = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        counts = [t for t in (completion, reasoning) if isinstance(t, int) and not isinstance(t, bool)]
        return max(counts) if counts else None

    @staticmethod
    def _is_empty(choice: dict) -> bool:
        message = choice.get("message")
        if isinstance(message, dict):
            if message.get("tool_calls"):
                return False
            content = message.get("content")
        else:
            content = choice.get("text")
        return not (content or "").strip()
