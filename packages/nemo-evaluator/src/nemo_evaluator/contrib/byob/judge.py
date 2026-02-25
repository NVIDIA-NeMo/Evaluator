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

"""LLM-as-Judge support for BYOB benchmarks.

Provides helpers that make it easy to write judge-based scorers without
boilerplate HTTP calls and response parsing.

The judge configuration schema aligns with the nemo-skills ``extra.judge``
convention used across NeMo Evaluator containers (simple-evals, mtbench,
nemo-skills).

Quick start::

    from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput
    from nemo_evaluator.contrib.byob.judge import judge_score

    @benchmark(
        name="qa-judge",
        dataset="qa.jsonl",
        prompt="Answer: {question}",
        judge={
            "url": "https://inference-api.nvidia.com/v1",
            "model_id": "openai/gpt-4.1",
            "api_key": "JUDGE_API_KEY",
        },
    )
    @scorer
    def qa_judge(sample: ScorerInput) -> dict:
        return judge_score(sample, template="binary_qa", criteria="Factual accuracy")

Multi-judge example::

    @benchmark(
        name="multi-judge",
        dataset="qa.jsonl",
        prompt="Answer: {question}",
        judge={"url": "http://judge1:8000/v1", "model_id": "judge-a"},
        judge_1={"url": "http://judge2:8000/v1", "model_id": "judge-b"},
    )
    @scorer
    def multi_judge(sample: ScorerInput) -> dict:
        scores_a = judge_score(sample, template="binary_qa")
        scores_b = judge_score(sample, template="likert_5", judge_key="judge_1")
        return {**scores_a, "judge_1_score": scores_b["judge_score"]}
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nemo_evaluator.contrib.byob.judge_templates import (
    DEFAULT_PATTERNS,
    DEFAULT_SCORE_MAPPINGS,
    TEMPLATES,
)
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JudgeConfig — aligned with nemo-skills extra.judge schema
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    """Configuration for a judge model endpoint.

    Field names align with the nemo-skills ``extra.judge`` convention used
    across NeMo Evaluator containers (simple-evals, mtbench, nemo-skills).

    Attributes:
        url: Base URL of the judge model endpoint.
        model_id: Model identifier for the judge.
        api_key: Environment variable name containing the API key.
        temperature: Sampling temperature (default 0.0 for determinism).
        top_p: Nucleus sampling parameter.
        max_new_tokens: Maximum tokens for judge response.
        parallelism: Max concurrent judge requests (informational).
        max_retries: Maximum retry attempts for judge calls.
        request_timeout: Request timeout in seconds.
    """

    url: str
    model_id: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    parallelism: Optional[int] = None
    max_retries: int = 16
    request_timeout: int = 600

    def resolve_api_key(self) -> Optional[str]:
        """Resolve the actual API key value from the environment variable."""
        if self.api_key:
            return os.environ.get(self.api_key)
        return None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgeConfig":
        """Create JudgeConfig from a dictionary.

        Accepts the nemo-skills field names. Also supports legacy BYOB
        field names (``api_key_name`` -> ``api_key``, ``max_tokens`` ->
        ``max_new_tokens``) for backward compatibility.
        """
        return cls(
            url=d["url"],
            model_id=d["model_id"],
            api_key=d.get("api_key") or d.get("api_key_name"),
            temperature=d.get("temperature", 0.0),
            top_p=d.get("top_p", 1.0),
            max_new_tokens=d.get("max_new_tokens") or d.get("max_tokens", 4096),
            parallelism=d.get("parallelism"),
            max_retries=d.get("max_retries", 16),
            request_timeout=d.get("request_timeout", 600),
        )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _create_judge_session(
    max_retries: int = 16,
    backoff_factor: float = 0.5,
) -> requests.Session:
    """Create a requests.Session for judge calls with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def judge_call(
    config: JudgeConfig,
    prompt: str,
    session: Optional[requests.Session] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the judge model and return the response text.

    Args:
        config: Judge endpoint configuration.
        prompt: The full judge prompt to send.
        session: Optional requests.Session for connection pooling.
        response_format: Optional response format dict for constrained decoding
            (e.g. ``{"type": "json_object"}``).  When set, it is included in
            the HTTP payload so that NIM endpoints apply constrained decoding.

    Returns:
        The judge model's response text.

    Raises:
        requests.HTTPError: On non-2xx response.
        requests.Timeout: On timeout.
    """
    endpoint = f"{config.url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key_value = config.resolve_api_key()
    if api_key_value:
        headers["Authorization"] = f"Bearer {api_key_value}"

    payload: Dict[str, Any] = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    http = session or requests
    response = http.post(
        endpoint, json=payload, headers=headers,
        timeout=config.request_timeout,
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def parse_grade(
    response: str,
    grade_pattern: str,
    structured: bool = False,
) -> Optional[str]:
    """Extract a grade from the judge response using regex or JSON parsing.

    When ``structured=True`` the response is assumed to be a JSON object
    produced by constrained decoding.  The function tries ``json.loads()``
    first and looks for common keys (``grade``, ``score``, ``rating``,
    ``verdict``).  If JSON parsing fails it falls back to regex.

    Args:
        response: Full judge response text.
        grade_pattern: Regex pattern with one capture group for the grade.
        structured: If True, try full JSON parse before regex.

    Returns:
        The extracted grade string, or None if no match found.
    """
    # When structured, try full JSON parse first
    if structured:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                for key in ("grade", "score", "rating", "verdict"):
                    if key in parsed:
                        return str(parsed[key])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try regex
    match = re.search(grade_pattern, response)
    if match:
        return match.group(1)

    # Fallback: try JSON extraction from embedded JSON
    try:
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            parsed = json.loads(json_match.group())
            for key in ("grade", "score", "rating", "verdict"):
                if key in parsed:
                    return str(parsed[key])
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def render_judge_prompt(template: str, **kwargs: Any) -> str:
    """Render a judge prompt template with the given variables.

    Args:
        template: Template string with ``{variable}`` placeholders.
        **kwargs: Variable values to substitute.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If a required placeholder is missing from kwargs.
    """
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# High-level judge_score entry point
# ---------------------------------------------------------------------------


# Module-level sessions keyed by judge URL for connection reuse
_judge_sessions: Dict[str, requests.Session] = {}


def _get_judge_session(config: JudgeConfig) -> requests.Session:
    """Return a session for the given judge config (created on first call)."""
    key = config.url
    if key not in _judge_sessions:
        _judge_sessions[key] = _create_judge_session(
            max_retries=config.max_retries,
        )
    return _judge_sessions[key]


def judge_score(
    sample: Any,
    template: str = "binary_qa",
    criteria: str = "",
    grade_pattern: Optional[str] = None,
    score_mapping: Optional[Dict[str, float]] = None,
    judge_key: str = "judge",
    response_format: Optional[Dict[str, Any]] = None,
    **template_kwargs: Any,
) -> dict:
    """Score a sample using an LLM judge.

    This is the primary entry point for judge-based scoring from inside a
    ``@scorer`` function.

    Args:
        sample: A ``ScorerInput`` instance.  Must have ``config[judge_key]``
            containing judge endpoint configuration (dict or JudgeConfig).
        template: Built-in template name (e.g. ``"binary_qa"``) or a custom
            template string with ``{question}``, ``{response}``,
            ``{reference}``, ``{criteria}`` placeholders.
        criteria: Evaluation criteria injected into the template's
            ``{criteria}`` placeholder.
        grade_pattern: Regex pattern to extract the grade.  If None, uses
            the default pattern for the named template.
        score_mapping: Dict mapping grade strings to numeric scores.  If None,
            uses the default mapping for the named template.
        judge_key: Key in ``sample.config`` containing the judge config dict.
            Defaults to ``"judge"`` (single judge).  For multi-judge setups
            use ``"judge_1"``, ``"judge_2"``, etc.
        response_format: Optional response format dict for constrained
            decoding (e.g. ``{"type": "json_object"}``).  When set, it is
            passed to ``judge_call()`` and ``parse_grade()`` uses structured
            JSON parsing.
        **template_kwargs: Extra template variables passed through to
            ``render_judge_prompt()``.  These override the default variables
            (``question``, ``response``, ``reference``, ``criteria``),
            allowing custom templates with arbitrary placeholders.

    Returns:
        Dict with ``judge_score`` (float) and ``judge_grade`` (str) keys.
        If the judge call fails or grade cannot be parsed, returns
        ``{"judge_score": 0.0, "judge_grade": "PARSE_ERROR"}``.
    """
    # Resolve judge config from sample
    judge_config_raw = sample.config.get(judge_key)
    if judge_config_raw is None:
        raise ValueError(
            f"judge_score() requires sample.config['{judge_key}']. "
            f"Pass {judge_key}={{...}} to @benchmark()."
        )

    if isinstance(judge_config_raw, JudgeConfig):
        config = judge_config_raw
    else:
        config = JudgeConfig.from_dict(judge_config_raw)

    # Resolve template
    if template in TEMPLATES:
        template_str = TEMPLATES[template]
        resolved_pattern = grade_pattern or DEFAULT_PATTERNS.get(template, r"GRADE:\s*(\S+)")
        resolved_mapping = score_mapping or DEFAULT_SCORE_MAPPINGS.get(template, {})
    else:
        template_str = template
        resolved_pattern = grade_pattern or r"GRADE:\s*(\S+)"
        resolved_mapping = score_mapping or {}

    # Render prompt — defaults can be overridden by template_kwargs
    prompt_vars = {
        "question": sample.metadata.get("question", sample.metadata.get("prompt", "")),
        "response": sample.response,
        "reference": sample.target,
        "criteria": f"Evaluation criteria: {criteria}" if criteria else "",
    }
    prompt_vars.update(template_kwargs)
    prompt = render_judge_prompt(template_str, **prompt_vars)

    # Get session (prefer internal session from config, fall back to per-URL session)
    session = sample.config.get("_judge_session") or _get_judge_session(config)

    # Call judge
    try:
        judge_response = judge_call(
            config, prompt, session=session, response_format=response_format,
        )
    except Exception:
        logger.warning("Judge call failed", judge_key=judge_key, url=config.url)
        return {"judge_score": 0.0, "judge_grade": "CALL_ERROR"}

    # Parse grade
    grade = parse_grade(
        judge_response, resolved_pattern, structured=response_format is not None,
    )
    if grade is None:
        logger.warning(
            "Could not parse grade from judge response",
            pattern=resolved_pattern, response_preview=judge_response[:200],
        )
        return {"judge_score": 0.0, "judge_grade": "PARSE_ERROR"}

    # Map grade to numeric score
    numeric_score = resolved_mapping.get(grade)
    if numeric_score is None:
        try:
            numeric_score = float(grade)
        except (ValueError, TypeError):
            logger.warning("Grade not in score_mapping and not numeric", grade=grade)
            numeric_score = 0.0

    return {"judge_score": numeric_score, "judge_grade": grade}
