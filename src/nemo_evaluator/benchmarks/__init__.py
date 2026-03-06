"""Built-in BYOB benchmark reference implementations.

These exist as templates showing how to write an EvalEnvironment with seed/verify.
For standard benchmarks (MMLU, MATH, HumanEval, IFEval, GPQA, DROP, ...),
use the harness adapters instead:

    nel harness run --harness simple-evals --eval mmlu
    nel harness run --harness lm-eval --tasks arc_easy,hellaswag
"""
from nemo_evaluator.benchmarks.gsm8k import GSM8KEnvironment
from nemo_evaluator.benchmarks.triviaqa import TriviaQAEnvironment

__all__ = ["GSM8KEnvironment", "TriviaQAEnvironment"]
