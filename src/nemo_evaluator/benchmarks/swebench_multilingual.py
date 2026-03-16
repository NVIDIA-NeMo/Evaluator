"""SWE-bench Multilingual -- software engineering benchmark across multiple languages."""
from nemo_evaluator.benchmarks._swebench_base import (
    SWEBENCH_PROMPT,
    swebench_prepare_row,
    swebench_score,
    swebench_seed_fn,
)
from nemo_evaluator.environments.byob import benchmark, scorer
from nemo_evaluator.scoring.types import ScorerInput

MULTILINGUAL_IMAGE_TEMPLATE = "swebench/sweb.eval.x86_64.{instance_id}:latest"


def _multilingual_seed_fn(row: dict, idx: int):
    return swebench_seed_fn(row, idx, image_template=MULTILINGUAL_IMAGE_TEMPLATE)


@benchmark(
    name="swebench-multilingual",
    dataset="hf://princeton-nlp/SWE-bench_Multilingual?split=test",
    prompt=SWEBENCH_PROMPT,
    target_field="instance_id",
    seed_fn=_multilingual_seed_fn,
    prepare_row=swebench_prepare_row,
)
@scorer
async def swebench_multilingual_scorer(sample: ScorerInput) -> dict:
    return await swebench_score(sample)
