"""SWE-bench Verified -- software engineering benchmark with human-verified instances."""
from nemo_evaluator.benchmarks._swebench_base import (
    SWEBENCH_PROMPT,
    swebench_prepare_row,
    swebench_score,
    swebench_seed_fn,
)
from nemo_evaluator.environments.byob import benchmark, scorer
from nemo_evaluator.scoring.types import ScorerInput


@benchmark(
    name="swebench-verified",
    dataset="hf://princeton-nlp/SWE-bench_Verified?split=test",
    prompt=SWEBENCH_PROMPT,
    target_field="instance_id",
    seed_fn=swebench_seed_fn,
    prepare_row=swebench_prepare_row,
)
@scorer
async def swebench_verified_scorer(sample: ScorerInput) -> dict:
    return await swebench_score(sample)
