"""GSM8K benchmark – grade-school math with numerical answer extraction.

BYOB reference implementation. Demonstrates the EvalEnvironment pattern with
math_equal scoring. Also usable via `nel serve --benchmark gsm8k` for Gym.
For production evaluation, prefer: nel harness run --harness lm-eval --tasks gsm8k
"""

from __future__ import annotations

import logging
import re

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register
from nemo_evaluator.scoring.extraction import extract_answer
from nemo_evaluator.scoring.math_equal import math_equal

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"####\s*(.+)")


def _extract_gsm8k_answer(solution: str) -> str:
    """Extract the final numeric answer after #### in GSM8K ground-truth solutions."""
    match = _ANSWER_RE.search(solution)
    if match:
        return match.group(1).strip().replace(",", "")
    lines = solution.strip().splitlines()
    return lines[-1].strip() if lines else ""


@register("gsm8k")
class GSM8KEnvironment(EvalEnvironment):
    """GSM8K: 8.5K grade-school math problems with step-by-step solutions.

    Downloads from HuggingFace datasets on first use. Each problem has a
    question and a solution ending with #### <answer>.
    """

    def __init__(self) -> None:
        super().__init__()
        self._load_dataset()

    def _load_dataset(self) -> None:
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main", split="test")
            self._dataset = [
                {
                    "question": row["question"],
                    "answer": _extract_gsm8k_answer(row["answer"]),
                    "full_solution": row["answer"],
                }
                for row in ds
            ]
            logger.info("Loaded GSM8K test set: %d problems", len(self._dataset))
        except Exception as e:
            logger.warning("Could not load GSM8K from HuggingFace: %s. Using empty dataset.", e)
            self._dataset = []

    def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        prompt = (
            f"Solve the following math problem step by step. "
            f"Put your final numerical answer after 'The answer is'.\n\n"
            f"Problem: {row['question']}"
        )
        return SeedResult(
            prompt=prompt,
            expected_answer=row["answer"],
            metadata={"category": "math", "dataset": "gsm8k"},
        )

    def verify(self, response: str, expected: str, **metadata) -> VerifyResult:
        extracted = extract_answer(response)
        correct = math_equal(extracted, expected)
        return VerifyResult(
            reward=1.0 if correct else 0.0,
            extracted_answer=extracted,
            scoring_details={
                "method": "math_equal",
                "correct": correct,
                "expected": expected,
                "extracted": extracted,
            },
            metadata=metadata,
        )
