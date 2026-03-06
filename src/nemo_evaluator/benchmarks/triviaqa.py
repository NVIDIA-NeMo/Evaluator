"""TriviaQA benchmark – trivia questions with exact-match scoring.

BYOB reference implementation. Demonstrates the EvalEnvironment pattern with
exact_match scoring. Also usable via `nel serve --benchmark triviaqa` for Gym.
For production evaluation, prefer: nel harness run --harness lm-eval --tasks triviaqa
"""

from __future__ import annotations

import logging

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult
from nemo_evaluator.environments.registry import register
from nemo_evaluator.scoring.exact_match import exact_match

logger = logging.getLogger(__name__)


def _get_aliases(answer_obj: dict) -> list[str]:
    """Extract all acceptable answer aliases from a TriviaQA answer object."""
    aliases = set()
    aliases.add(answer_obj.get("value", ""))
    for alias in answer_obj.get("aliases", []):
        aliases.add(alias)
    for ne in answer_obj.get("normalized_aliases", []):
        aliases.add(ne)
    aliases.discard("")
    return list(aliases)


@register("triviaqa")
class TriviaQAEnvironment(EvalEnvironment):
    """TriviaQA: trivia questions from Wikipedia and the web.

    Downloads from HuggingFace datasets on first use (rc split, validation set
    because the test set has no ground-truth labels).
    """

    def __init__(self) -> None:
        super().__init__()
        self._load_dataset()

    def _load_dataset(self) -> None:
        try:
            from datasets import load_dataset

            ds = load_dataset("trivia_qa", "rc", split="validation")
            self._dataset = [
                {
                    "question": row["question"],
                    "answer": row["answer"]["value"],
                    "aliases": _get_aliases(row["answer"]),
                    "category": row.get("entity_pages", {}).get("title", ["general"])[0]
                    if row.get("entity_pages", {}).get("title")
                    else "general",
                }
                for row in ds
            ]
            logger.info("Loaded TriviaQA validation set: %d problems", len(self._dataset))
        except Exception as e:
            logger.warning("Could not load TriviaQA from HuggingFace: %s. Using empty dataset.", e)
            self._dataset = []

    def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        prompt = (
            f"Answer the following trivia question with a short factual answer.\n\n"
            f"Question: {row['question']}\n"
            f"Answer:"
        )
        return SeedResult(
            prompt=prompt,
            expected_answer=row["answer"],
            metadata={
                "category": row.get("category", "general"),
                "dataset": "triviaqa",
                "_aliases": row.get("aliases", [row["answer"]]),
            },
        )

    def verify(self, response: str, expected: str, **metadata) -> VerifyResult:
        response_clean = response.strip().split("\n")[0].strip()

        aliases = metadata.get("_aliases", [expected])
        if not isinstance(aliases, list):
            aliases = [expected]

        correct = any(exact_match(response_clean, alias) for alias in aliases)

        return VerifyResult(
            reward=1.0 if correct else 0.0,
            extracted_answer=response_clean,
            scoring_details={
                "method": "exact_match",
                "correct": correct,
                "expected": expected,
                "response_first_line": response_clean,
            },
            metadata=metadata,
        )
