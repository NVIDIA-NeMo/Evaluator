"""SimpleQA -- short-form factuality (requires LLM-as-judge)."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, needs_judge


@benchmark(
    name="simpleqa",
    dataset="hf://basicv8vc/SimpleQA?split=test",
    prompt="{problem}",
    target_field="answer",
    extra={"judge": True},
)
@scorer
def simpleqa_scorer(sample: ScorerInput) -> dict:
    return needs_judge(sample)
