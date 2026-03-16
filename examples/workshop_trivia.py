from nemo_evaluator import benchmark, scorer, ScorerInput, fuzzy_match

DATASET = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "In what year did World War II end?", "answer": "1945"},
]

@benchmark(
    name="workshop_trivia",
    dataset=lambda: DATASET,
    prompt="Answer concisely.\n\nQuestion: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def score(sample: ScorerInput) -> dict:
    return fuzzy_match(sample)
