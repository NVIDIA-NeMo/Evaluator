"""SimpleQA -- short-form factuality (requires LLM-as-judge)."""
from nemo_evaluator.environments.definitions import ScorerInput, benchmark, needs_judge, scorer


@benchmark(name="simpleqa", dataset="hf://basicv8vc/SimpleQA?split=test",
           prompt="{problem}", target_field="answer",
           extra={"judge": True})
@scorer
def simpleqa_scorer(sample: ScorerInput) -> dict:
    return needs_judge(sample)
