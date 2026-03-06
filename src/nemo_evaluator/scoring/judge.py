"""LLM-as-judge scoring for open-ended evaluation.

Supports reference-based and reference-free judging with configurable rubrics.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

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
class JudgeConfig:
    rubric_template: str = DEFAULT_RUBRIC
    max_score: int = 5
    threshold: float = 0.6
    swap_check: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)


def build_judge_prompt(
    instruction: str,
    response: str,
    expected: str | None = None,
    config: JudgeConfig | None = None,
) -> str:
    cfg = config or JudgeConfig()
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
_BARE_SCORE_RE = re.compile(r'\b(\d+)\s*/\s*\d+\b')


def parse_judge_response(text: str, max_score: int = 5) -> dict[str, Any]:
    """Parse a judge model's response into a score and reasoning."""
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


async def judge_score(
    instruction: str,
    response: str,
    expected: str | None = None,
    client: Any = None,
    config: JudgeConfig | None = None,
) -> dict[str, Any]:
    """End-to-end LLM-as-judge scoring: build prompt, call model, parse result.

    Requires a ModelClient instance. Returns parsed score with full audit trail.
    Optionally performs swap-check (position bias detection) when config.swap_check=True.
    """
    if client is None:
        raise ValueError("judge_score requires a ModelClient instance")

    cfg = config or JudgeConfig()
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
