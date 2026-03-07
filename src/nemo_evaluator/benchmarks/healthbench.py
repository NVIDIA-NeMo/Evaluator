"""HealthBench -- health-related QA (requires LLM-as-judge)."""
from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.define import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, needs_judge


def _load_healthbench():
    from datasets import load_dataset
    ds = load_dataset("openai/HealthBench", split="test")
    return [dict(row) for row in ds]


def _seed_healthbench(row, idx):
    conv = row.get("conversation", [])
    prompt = conv[-1].get("content", "") if conv else ""
    messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in conv]
    return SeedResult(
        prompt=prompt, expected_answer="",
        messages=messages,
        metadata={"category": row.get("category", ""), "criteria": row.get("criteria", [])},
    )


@benchmark(name="healthbench", dataset=_load_healthbench, target_field="",
           prompt="", seed_fn=_seed_healthbench, extra={"judge": True})
@scorer
def healthbench_scorer(sample: ScorerInput) -> dict:
    return needs_judge(sample)
