"""MMLU -- Massive Multitask Language Understanding (14K 4-choice QA)."""
from nemo_evaluator.environments.byob import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, multichoice_regex

_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def _prepare(row, idx, rng):
    c = row["choices"]
    return {**row, "Question": row["question"], "A": c[0], "B": c[1], "C": c[2], "D": c[3],
            "answer": "ABCD"[row["answer"]], "category": row.get("subject", "")}


@benchmark(name="mmlu", dataset="hf://cais/mmlu?config=all&split=test",
           prompt=_PROMPT, target_field="answer", prepare_row=_prepare)
@scorer
def mmlu_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample)
