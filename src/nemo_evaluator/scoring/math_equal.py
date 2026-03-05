"""Sympy-based mathematical equality comparison."""

from __future__ import annotations

import re


def _normalize_number(s: str) -> str:
    """Strip whitespace, commas, dollar signs, percent signs, trailing periods."""
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.rstrip(".")
    return s


def _try_parse_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _sympy_equal(a: str, b: str) -> bool | None:
    """Try symbolic comparison via sympy. Returns None if sympy unavailable or parse fails."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return None

    for parser in [sympy.sympify, parse_latex]:
        try:
            expr_a = parser(a)
            expr_b = parser(b)
            if expr_a is not None and expr_b is not None:
                diff = sympy.simplify(expr_a - expr_b)
                if diff == 0:
                    return True
        except Exception:
            continue
    return None


def math_equal(predicted: str, expected: str) -> bool:
    """Compare two math answers. Tries numeric comparison first, then sympy symbolic."""
    pred_norm = _normalize_number(predicted)
    exp_norm = _normalize_number(expected)

    if pred_norm == exp_norm:
        return True

    pred_f = _try_parse_float(pred_norm)
    exp_f = _try_parse_float(exp_norm)
    if pred_f is not None and exp_f is not None:
        return abs(pred_f - exp_f) < 1e-6

    sym_result = _sympy_equal(pred_norm, exp_norm)
    if sym_result is not None:
        return sym_result

    return pred_norm.lower() == exp_norm.lower()
