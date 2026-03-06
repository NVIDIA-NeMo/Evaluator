"""Harness adapters: wrap external eval harnesses without modifying them.

Lazy imports keep heavy dependencies (lm_eval, simple_evals) optional.
"""


def __getattr__(name: str):
    if name in ("SimpleEvalsAdapter", "NelSampler", "list_evals", "run_simple_eval"):
        from nemo_evaluator.harnesses.simple_evals import (  # noqa: F401
            SimpleEvalsAdapter, NelSampler, list_evals, run_simple_eval,
        )
        return locals()[name]
    if name in ("LMEvalAdapter", "NelLM", "list_tasks", "list_generate_tasks",
                 "run_lm_eval", "UnsupportedTaskTypeError"):
        from nemo_evaluator.harnesses.lm_eval import (  # noqa: F401
            LMEvalAdapter, NelLM, list_tasks, list_generate_tasks,
            run_lm_eval, UnsupportedTaskTypeError,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimpleEvalsAdapter", "NelSampler", "list_evals", "run_simple_eval",
    "LMEvalAdapter", "NelLM", "list_tasks", "list_generate_tasks",
    "run_lm_eval", "UnsupportedTaskTypeError",
]
