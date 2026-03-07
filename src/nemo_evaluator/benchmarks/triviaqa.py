"""TriviaQA -- trivia questions with multi-alias exact match."""
from nemo_evaluator.environments.define import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput


def _prepare(row, idx, rng):
    aliases = list({row["answer"]["value"]}
                   | set(row["answer"].get("aliases", []))
                   | set(row["answer"].get("normalized_aliases", []))
                   - {""})
    return {**row, "question": row["question"], "answer_text": row["answer"]["value"],
            "_aliases": aliases}


def _load_triviaqa():
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc", split="validation")
    return [dict(row) for row in ds]


@benchmark(name="triviaqa", dataset=_load_triviaqa,
           prompt="Answer with a short factual answer.\n\nQuestion: {question}\nAnswer:",
           target_field="answer_text", prepare_row=_prepare)
@scorer
def triviaqa_scorer(sample: ScorerInput) -> dict:
    first_line = sample.response.strip().split("\n")[0].strip()
    aliases = sample.metadata.get("_aliases", [str(sample.target)])
    from nemo_evaluator.scoring.text import _normalize
    pred = _normalize(first_line)
    correct = any(_normalize(a) == pred for a in aliases)
    return {"correct": correct, "extracted": first_line}
