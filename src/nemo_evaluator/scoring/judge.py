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
"""LLM-as-judge scoring for open-ended evaluation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)

DEFAULT_RUBRIC = """You are an expert evaluator. Score the following response on a scale of 1-5.

{instruction}

Response to evaluate:
{response}

{reference_section}

Provide your score as a JSON object: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

REFERENCE_SECTION = """Reference answer:
{expected}

Compare the response against the reference answer."""


@dataclass
class JudgeScoringConfig:
    rubric_template: str = DEFAULT_RUBRIC
    max_score: int = 5
    threshold: float = 0.6
    swap_check: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)


def build_judge_prompt(
    instruction: str,
    response: str,
    expected: str | None = None,
    config: JudgeScoringConfig | None = None,
) -> str:
    cfg = config or JudgeScoringConfig()
    ref_section = ""
    if expected:
        ref_section = REFERENCE_SECTION.format(expected=expected)

    return cfg.rubric_template.format(
        instruction=instruction,
        response=response,
        reference_section=ref_section,
        expected=expected or "",
    )


_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+(?:\.\d+)?)')
_BARE_SCORE_RE = re.compile(r"\b(\d+)\s*/\s*\d+\b")


def parse_judge_response(text: str, max_score: int = 5) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            score = float(data["score"])
            return {
                "score": min(score, max_score),
                "normalized": min(score / max_score, 1.0),
                "reasoning": data.get("reasoning", ""),
                "raw": text,
            }
    except (json.JSONDecodeError, ValueError):
        pass

    match = _SCORE_RE.search(text)
    if match:
        score = float(match.group(1))
        return {
            "score": min(score, max_score),
            "normalized": min(score / max_score, 1.0),
            "reasoning": text,
            "raw": text,
        }

    match = _BARE_SCORE_RE.search(text)
    if match:
        score = float(match.group(1))
        return {
            "score": min(score, max_score),
            "normalized": min(score / max_score, 1.0),
            "reasoning": text,
            "raw": text,
        }

    logger.warning("Could not parse judge score from: %s", text[:200])
    return {"score": 0, "normalized": 0.0, "reasoning": text, "raw": text, "parse_error": True}


def needs_judge(sample: ScorerInput) -> dict:
    """Signals that this sample needs LLM-as-judge post-processing in the eval loop."""
    return {"correct": False, "needs_judge": True, "extracted": sample.response[:500]}


async def judge_score(
    instruction: str,
    response: str,
    expected: str | None = None,
    client: Any = None,
    config: JudgeScoringConfig | None = None,
) -> dict[str, Any]:
    """End-to-end LLM-as-judge: build prompt, call model, parse result."""
    if client is None:
        raise ValueError("judge_score requires a ModelClient instance")

    cfg = config or JudgeScoringConfig()
    prompt = build_judge_prompt(instruction, response, expected, cfg)
    model_resp = await client.chat(
        prompt=prompt,
        system="You are a strict evaluation judge. Always respond with valid JSON.",
    )
    result = parse_judge_response(model_resp.content, cfg.max_score)
    result["judge_model"] = model_resp.model
    result["judge_tokens"] = model_resp.total_tokens
    result["judge_latency_ms"] = model_resp.latency_ms

    if cfg.swap_check and expected:
        swap_prompt = build_judge_prompt(instruction, expected, response, cfg)
        swap_resp = await client.chat(
            prompt=swap_prompt,
            system="You are a strict evaluation judge. Always respond with valid JSON.",
        )
        swap_result = parse_judge_response(swap_resp.content, cfg.max_score)
        result["swap_score"] = swap_result["score"]
        result["swap_normalized"] = swap_result["normalized"]
        result["position_bias"] = abs(result["normalized"] - swap_result["normalized"])

    return result
