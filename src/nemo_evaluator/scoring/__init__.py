"""Scoring: all scorer implementations and utilities.

Scorers are functions ``(ScorerInput) -> dict`` used by benchmarks.
"""
from nemo_evaluator.scoring.judge import (
    JudgeScoringConfig,
    build_judge_prompt,
    judge_score,
    needs_judge,
    parse_judge_response,
)
from nemo_evaluator.scoring.json_schema import extract_json, validate_json_schema
from nemo_evaluator.scoring.pattern import answer_line, multichoice_regex, numeric_match
from nemo_evaluator.scoring.sandbox import code_sandbox, code_sandbox_async
from nemo_evaluator.scoring.text import exact_match, fuzzy_match
from nemo_evaluator.scoring.types import ScorerInput

__all__ = [
    "ScorerInput",
    # Text
    "exact_match",
    "fuzzy_match",
    # Pattern
    "multichoice_regex",
    "answer_line",
    "numeric_match",
    # Sandbox
    "code_sandbox",
    "code_sandbox_async",
    # Judge
    "needs_judge",
    "JudgeScoringConfig",
    "build_judge_prompt",
    "parse_judge_response",
    "judge_score",
    # JSON
    "extract_json",
    "validate_json_schema",
]
