"""GSM8K -- grade school math (1.3K test problems)."""
import re
from nemo_evaluator.environments.define import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, answer_line

_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer after 'The answer is'.\n\n"
    "Problem: {question}"
)


def _prepare(row, idx, rng):
    m = re.search(r"####\s*(.+)", row.get("answer", ""))
    answer = m.group(1).strip().replace(",", "") if m else row.get("answer", "")
    return {**row, "answer": answer}


@benchmark(name="gsm8k", dataset="hf://openai/gsm8k?config=main&split=test",
           prompt=_PROMPT, target_field="answer", prepare_row=_prepare)
@scorer
def gsm8k_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample, pattern=r"(?i)(?:the answer is|answer\s*:)\s*([^\n]+)")
