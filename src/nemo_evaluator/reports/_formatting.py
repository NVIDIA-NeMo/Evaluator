"""Shared terminal formatting utilities for text renderers.

Provides ANSI-color helpers via ``click.style`` with ``NO_COLOR`` support,
verdict-color mapping, and a simple progress bar builder.
"""

from __future__ import annotations

import os

import click

NO_COLOR = os.environ.get("NO_COLOR") is not None


def style(text: str, **kwargs) -> str:
    """Wrap *text* in ANSI color/bold via ``click.style``, respecting ``NO_COLOR``."""
    if NO_COLOR:
        return text
    return click.style(text, **kwargs)


def bar(value: float, width: int = 20) -> str:
    """Return a block-char progress bar representing *value* (0-1)."""
    filled = round(value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def verdict_color(verdict: str) -> str:
    """Map a verdict string to a Click color name."""
    if verdict in ("BLOCK", "NO-GO"):
        return "red"
    if verdict in ("WARN", "INCONCLUSIVE"):
        return "yellow"
    return "green"
