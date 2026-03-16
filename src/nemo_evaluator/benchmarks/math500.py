"""MATH-500 -- competition math from HuggingFaceH4."""
from nemo_evaluator.environments.byob import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, answer_line

_PROMPT = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer "
    "to the problem.\n\n{problem}"
)


def _prepare(row, idx, rng):
    return {**row, "category": row.get("type", ""), "level": row.get("level", "")}


@benchmark(name="math500", dataset="hf://HuggingFaceH4/MATH-500?split=test",
           prompt=_PROMPT, target_field="answer", prepare_row=_prepare)
@scorer
def math500_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample)
