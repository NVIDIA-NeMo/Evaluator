# Built-in Benchmarks

## Available Benchmarks

| Benchmark | Dataset | Scoring | Problems | Category |
|-----------|---------|---------|----------|----------|
| `gsm8k` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) test split | `math_equal` (sympy) | 1,319 | Math reasoning |
| `triviaqa` | [TriviaQA](https://huggingface.co/datasets/trivia_qa) RC validation | `exact_match` with aliases | ~11,000 | Trivia / factual |

## GSM8K

Grade-school math problems requiring multi-step reasoning.

**Prompt format:**
```
Solve the following math problem step by step. Put your final answer in \boxed{}.

Question: Natalia sold clips to 48 of her friends in April, and then she sold half
as many clips in May. How many clips did Natalia sell altogether in April and May?
```

**Answer extraction:** Looks for `\boxed{72}` first, then "the answer is 72", then the last number on the last line.

**Scoring:** Symbolic comparison via sympy. Handles: `72`, `72.0`, `\frac{72}{1}`, etc.

## TriviaQA

Trivia questions with multiple acceptable answer aliases.

**Prompt format:**
```
Answer the following trivia question with a short factual answer.

Question: What is the capital of Australia?
Answer:
```

**Scoring:** Normalized exact match against the primary answer and all aliases (e.g., "Canberra", "canberra", "CANBERRA").

## Adding a New Benchmark

See the {doc}`../tutorials/byob` tutorial for a complete walkthrough.
