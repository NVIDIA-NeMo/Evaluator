"""Shared trajectory helpers for single-turn solvers."""
from __future__ import annotations

from typing import Any


def _single_turn_trajectory(
    prompt: str, response: str, system: str | None = None,
) -> list[dict[str, Any]]:
    """Build a minimal trajectory for a single model call.

    Provides uniform trajectory output across all solver types so that
    downstream consumers (logs, UI, scorers) always see a consistent format.
    """
    entries: list[dict[str, Any]] = []
    if system:
        entries.append({
            "type": "message",
            "message": {"role": "system", "content": [{"type": "text", "text": system}]},
        })
    entries.append({
        "type": "message",
        "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
    })
    entries.append({
        "type": "message",
        "message": {"role": "assistant", "content": [{"type": "text", "text": response}]},
    })
    return entries
