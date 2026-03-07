"""Scoring utilities: judge + JSON schema validation.

Environment-specific scoring (math, MCQ, exact match) lives in each
environment's verify() method -- not here. These are NEL-native capabilities
that add value beyond what external harnesses provide.
"""
from nemo_evaluator.scoring.judge import (
    JudgeConfig, build_judge_prompt, judge_score, parse_judge_response,
)
from nemo_evaluator.scoring.json_schema import extract_json, validate_json_schema

__all__ = [
    "JudgeConfig", "build_judge_prompt", "judge_score", "parse_judge_response",
    "extract_json", "validate_json_schema",
]
