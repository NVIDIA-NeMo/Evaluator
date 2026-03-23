"""Shared exception types for the nemo-evaluator framework."""


class GracefulError(Exception):
    """Raised when a task fails in an expected way.

    The eval loop scores ``reward=0.0``, skips verification, and does
    **not** retry.  Everything that is *not* a ``GracefulError`` is
    treated as a system / infrastructure failure and will be retried.

    Typical causes:
    - Turn budget exhausted (interceptor hard-limit)
    - Agent produced no output (0 tokens, empty response)
    - Agent-internal crash caught by the solver
    - Tool infrastructure error during a ReAct loop
    - OpenClaw non-zero exit / no JSON output / timeout
    """
