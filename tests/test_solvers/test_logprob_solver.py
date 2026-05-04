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
"""Tests for :class:`LogprobRankingSolver` and the response parser."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.solvers.base import ErrorKind
from nemo_evaluator.solvers.logprob import (
    LogprobRankingSolver,
    _parse_loglikelihood_response,
)


# ── _parse_loglikelihood_response ───────────────────────────────────────────


def _build_logprobs_response(
    *,
    tokens: list[str],
    token_logprobs: list[float | None],
    text_offset: list[int | None],
    top_logprobs: list[dict[str, float] | None] | None = None,
) -> dict:
    """Shape an OpenAI-compatible /completions response body."""
    return {
        "choices": [
            {
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "text_offset": text_offset,
                    "top_logprobs": top_logprobs or [{tok: lp or 0.0} for tok, lp in zip(tokens, token_logprobs)],
                }
            }
        ]
    }


def test_parse_response_sums_continuation_logprobs() -> None:
    body = _build_logprobs_response(
        tokens=["The", " cat", " sat"],
        token_logprobs=[None, -0.5, -0.3],
        text_offset=[0, 3, 7],
    )
    sum_lp, _ = _parse_loglikelihood_response(body, context="The")
    assert sum_lp == pytest.approx(-0.8)


def test_parse_response_returns_minus_inf_when_logprobs_missing() -> None:
    body: dict = {"choices": [{"text": "x"}]}  # no logprobs field
    sum_lp, greedy = _parse_loglikelihood_response(body, context="ctx")
    assert sum_lp == float("-inf")
    assert greedy is False


def test_parse_response_is_greedy_true_when_top1_matches_tokens() -> None:
    body = _build_logprobs_response(
        tokens=["The", " cat"],
        token_logprobs=[None, -0.5],
        text_offset=[0, 3],
        top_logprobs=[{"The": -0.1, "An": -1.5}, {" cat": -0.5, " dog": -2.0}],
    )
    _, greedy = _parse_loglikelihood_response(body, context="The")
    assert greedy is True


def test_parse_response_is_greedy_false_when_top1_differs() -> None:
    body = _build_logprobs_response(
        tokens=["The", " cat"],
        token_logprobs=[None, -0.5],
        text_offset=[0, 3],
        top_logprobs=[{"The": -0.1}, {" dog": -0.1, " cat": -0.5}],
    )
    _, greedy = _parse_loglikelihood_response(body, context="The")
    assert greedy is False


def test_parse_response_handles_token_straddling_with_recursive_fallback() -> None:
    """If text_offset never crosses ctx_len, walk back from end."""
    # All tokens have offset < 3 (i.e. text_offset never reaches ctx_len),
    # falls back to last token containing some continuation.
    body = _build_logprobs_response(
        tokens=["The", " cat"],
        token_logprobs=[None, -0.4],
        text_offset=[0, 1],  # offsets all < 5 (ctx_len)
    )
    sum_lp, _ = _parse_loglikelihood_response(body, context="The c")
    # Fallback: only the last token's logprob is summed
    assert sum_lp == pytest.approx(-0.4)


def test_parse_response_empty_continuation_returns_minus_inf() -> None:
    body = _build_logprobs_response(
        tokens=["a", "b"],
        token_logprobs=[None, None],
        text_offset=[0, 1],
    )
    sum_lp, _ = _parse_loglikelihood_response(body, context="ab")
    assert sum_lp == float("-inf")


# ── LogprobRankingSolver.solve ──────────────────────────────────────────────


def _seed_with_choices(choices: list[str]) -> SeedResult:
    return SeedResult(
        prompt="Q: 2+2 = ",
        expected_answer="4",
        metadata={"_mc_choices": choices, "source": "byob"},
    )


@pytest.mark.asyncio
async def test_solve_ranks_choices_by_sum_logprob() -> None:
    # Three choices; the second one wins by raw logprob sum.
    responses = [
        _build_logprobs_response(tokens=["Q: 2+2 = ", "3"], token_logprobs=[None, -2.5], text_offset=[0, 9]),
        _build_logprobs_response(tokens=["Q: 2+2 = ", "4"], token_logprobs=[None, -0.5], text_offset=[0, 9]),
        _build_logprobs_response(tokens=["Q: 2+2 = ", "5"], token_logprobs=[None, -3.0], text_offset=[0, 9]),
    ]
    seed = _seed_with_choices(["3", "4", "5"])
    response_iter = iter(responses)

    solver = LogprobRankingSolver(base_url="http://fake", model="fake-model")

    async def fake_post(url, payload):
        return next(response_iter)

    with patch.object(solver._model_client, "_post_with_retry", side_effect=fake_post):
        result = await solver.solve(seed)

    assert result.response == "4"
    assert result.error is None
    assert result.scoring_details["_mc_choices"] == ["3", "4", "5"]
    assert result.scoring_details["_mc_choices_logprobs"] == pytest.approx([-2.5, -0.5, -3.0])
    assert len(result.scoring_details["_mc_choices_is_greedy"]) == 3


@pytest.mark.asyncio
async def test_solve_returns_graceful_error_when_choices_missing() -> None:
    seed = SeedResult(prompt="anything", expected_answer="?")
    solver = LogprobRankingSolver(base_url="http://fake", model="fake-model")
    result = await solver.solve(seed)
    assert result.response == ""
    assert result.error_kind == ErrorKind.GRACEFUL
    assert "_mc_choices" in (result.error or "")


@pytest.mark.asyncio
async def test_solve_propagates_inference_failures_as_infra_error() -> None:
    seed = _seed_with_choices(["a", "b"])
    solver = LogprobRankingSolver(base_url="http://fake", model="fake-model")

    async def boom(url, payload):
        raise RuntimeError("connection refused")

    with patch.object(solver._model_client, "_post_with_retry", side_effect=boom):
        result = await solver.solve(seed)

    assert result.response == ""
    assert result.error_kind == ErrorKind.INFRA
    assert "connection refused" in (result.error or "")
