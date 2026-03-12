"""Backward-compatibility shim -- NatSolver now lives in nemo_evaluator.solvers."""
from nemo_evaluator.solvers.nat import (  # noqa: F401
    NatSolver,
    _convert_trajectory,
    _parse_sse_lines,
)
