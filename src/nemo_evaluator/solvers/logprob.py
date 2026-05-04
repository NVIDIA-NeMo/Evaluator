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
"""LogprobRankingSolver: score candidate continuations by sum-loglikelihood.

Mirrors lm-evaluation-harness's ``loglikelihood`` contract for the
``local-completions`` model adapter. For each row the solver:

1. Reads candidate continuations from ``task.metadata["_mc_choices"]``.
2. POSTs ``/completions`` with ``prompt = task.prompt + continuation``,
   ``max_tokens=0``, ``echo=true``, ``logprobs=1`` for each candidate.
3. Locates the continuation token span via OpenAI-compatible
   ``logprobs.text_offset`` (first token whose offset is ``>= len(prompt)``).
4. Sums ``token_logprobs`` over the continuation span, computes greedy
   eligibility against ``top_logprobs``, returns a ``SolveResult`` whose
   ``response`` is the argmax choice and whose ``scoring_details`` carries
   the per-choice logprobs/greedy flags for the verify stage.

The eval loop forwards ``solve_result.scoring_details`` to
:meth:`EvalEnvironment.verify` so the scoring step can place the payload
on ``MetricInput.candidate.metadata`` for the metric to consume.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import ErrorKind, SolveResult
from nemo_evaluator.solvers.trajectory_util import build_single_turn_atif

logger = logging.getLogger(__name__)


class LogprobRankingSolver:
    """Solver that ranks candidate continuations by sum-loglikelihood.

    Required ``task.metadata`` keys:

    - ``_mc_choices``: ``list[str]`` of candidate continuations.

    Optional ``task.metadata`` keys:

    - ``_mc_continuation_separator``: ``str`` inserted between the prompt
      and each continuation (default ``""``). lm-eval-harness conventionally
      uses ``" "`` for whitespace-sensitive tokenizers.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_concurrent_choices: int = 8,
    ) -> None:
        from nemo_evaluator.engine.model_client import ModelClient

        self._model_client = ModelClient(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
        )
        self._url = f"{base_url.rstrip('/')}/completions"
        self._model = model
        self._max_concurrent_choices = max_concurrent_choices

    async def solve(self, task: SeedResult) -> SolveResult:
        choices: list[str] = list(task.metadata.get("_mc_choices") or [])
        if not choices:
            return SolveResult(
                response="",
                error="LogprobRankingSolver requires task.metadata['_mc_choices'] to be a non-empty list",
                error_kind=ErrorKind.GRACEFUL,
            )

        separator = task.metadata.get("_mc_continuation_separator", "")
        prompt = task.prompt or ""
        full_context = (task.system + "\n" + prompt) if task.system else prompt

        sem = asyncio.Semaphore(self._max_concurrent_choices)

        async def _score_one(choice: str) -> tuple[float, bool]:
            async with sem:
                return await self._score_continuation(full_context, separator, choice)

        t0 = time.monotonic()
        results = await asyncio.gather(*[_score_one(c) for c in choices], return_exceptions=True)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        logprobs: list[float] = []
        is_greedy: list[bool] = []
        for r in results:
            if isinstance(r, BaseException):
                return SolveResult(
                    response="",
                    error=f"LogprobRankingSolver inference failed: {r}",
                    error_kind=ErrorKind.INFRA,
                )
            ll, greedy = r
            logprobs.append(ll)
            is_greedy.append(greedy)

        argmax_idx = max(range(len(choices)), key=lambda i: logprobs[i])
        response = choices[argmax_idx]

        model_resp = ModelResponse(
            content=response,
            model=self._model,
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=latency_ms,
            raw_response={"choices_logprobs": logprobs, "choices_is_greedy": is_greedy, "n_choices": len(choices)},
        )
        trajectory = build_single_turn_atif(prompt, response, model_name=self._model)
        return SolveResult(
            response=response,
            model_response=model_resp,
            trajectory=trajectory,
            scoring_details={
                "_mc_choices": choices,
                "_mc_choices_logprobs": logprobs,
                "_mc_choices_is_greedy": is_greedy,
            },
        )

    async def close(self) -> None:
        await self._model_client.close()

    async def _score_continuation(self, context: str, separator: str, continuation: str) -> tuple[float, bool]:
        """Score one continuation by sum-loglikelihood.

        Returns ``(sum_logprob, is_greedy)``. ``is_greedy`` is True iff every
        continuation token equals the top-1 candidate at its position under
        the model's logprob distribution.
        """
        full_context = context + separator
        full_prompt = full_context + continuation
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": full_prompt,
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }
        data = await self._model_client._post_with_retry(self._url, payload)
        return _parse_loglikelihood_response(data, full_context)


def _parse_loglikelihood_response(body: dict[str, Any], context: str) -> tuple[float, bool]:
    """Parse an OpenAI-compatible /completions response into ``(sum_logprob, is_greedy)``.

    Returns ``(-inf, False)`` when the response is missing logprobs or the
    continuation span is empty (zero tokens).
    """
    try:
        choice = body["choices"][0]
        logprobs = choice.get("logprobs") or {}
    except (KeyError, IndexError, TypeError):
        return (float("-inf"), False)

    tokens: list[str] = logprobs.get("tokens") or []
    token_logprobs: list[float | None] = logprobs.get("token_logprobs") or []
    text_offset: list[int | None] = logprobs.get("text_offset") or []
    top_logprobs: list[dict[str, float] | None] = logprobs.get("top_logprobs") or []

    if not tokens or not token_logprobs:
        return (float("-inf"), False)

    ctx_len = len(context)
    start_idx: int | None = None
    for i, off in enumerate(text_offset):
        if off is not None and off >= ctx_len:
            start_idx = i
            break

    # Fallback: tokenizer merged context-end with continuation-start. Walk
    # backwards from the end and pick the first token whose ``text_offset``
    # is < ``ctx_len`` — the token AFTER it is the first that contains
    # continuation chars only. This is an approximation; lm-eval-harness
    # tokenizes context separately for exact accounting.
    if start_idx is None:
        for i in range(len(text_offset) - 1, -1, -1):
            off = text_offset[i]
            if off is not None and off < ctx_len:
                start_idx = i + 1 if i + 1 < len(tokens) else i
                break
        if start_idx is None:
            start_idx = max(len(tokens) - 1, 0)

    cont_token_logprobs = [lp for lp in token_logprobs[start_idx:] if lp is not None]
    if not cont_token_logprobs:
        return (float("-inf"), False)

    sum_logprob = float(sum(cont_token_logprobs))

    is_greedy = True
    for i in range(start_idx, len(tokens)):
        top = top_logprobs[i] if i < len(top_logprobs) else None
        if not top:
            is_greedy = False
            break
        try:
            best_token = max(top.items(), key=lambda kv: kv[1])[0]
        except (AttributeError, ValueError):
            is_greedy = False
            break
        if best_token != tokens[i]:
            is_greedy = False
            break

    return (sum_logprob, is_greedy)


__all__ = ["LogprobRankingSolver"]
