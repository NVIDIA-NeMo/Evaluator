"""HumanEval -- code generation with Docker-sandboxed test execution."""
from nemo_evaluator.environments.definitions import ScorerInput, benchmark, code_sandbox, scorer

_PROMPT = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n\n{prompt}"
)


def _prepare(row, idx, rng):
    return {**row, "_prompt": row["prompt"], "_test": row["test"],
            "entry_point": row["entry_point"]}


@benchmark(name="humaneval", dataset="hf://openai/openai_humaneval?split=test",
           prompt=_PROMPT, target_field="entry_point", prepare_row=_prepare)
@scorer
def humaneval_scorer(sample: ScorerInput) -> dict:
    return code_sandbox(sample)
