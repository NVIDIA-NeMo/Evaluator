"""GPQA Diamond -- graduate-level science QA with shuffled choices."""
from nemo_evaluator.environments.definitions import ScorerInput, benchmark, multichoice_regex, scorer

_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def _prepare(row, idx, rng):
    choices = [row["Correct Answer"], row["Incorrect Answer 1"],
               row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    perm = list(range(4))
    rng.shuffle(perm)
    shuffled = [choices[i] for i in perm]
    correct_letter = "ABCD"[shuffled.index(row["Correct Answer"])]
    return {**row, "Question": row["Question"],
            "A": shuffled[0], "B": shuffled[1], "C": shuffled[2], "D": shuffled[3],
            "answer": correct_letter, "category": row.get("High-level domain", "")}


@benchmark(name="gpqa", dataset="hf://Idavidrein/gpqa?config=gpqa_diamond&split=train",
           prompt=_PROMPT, target_field="answer", prepare_row=_prepare)
@scorer
def gpqa_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample)
