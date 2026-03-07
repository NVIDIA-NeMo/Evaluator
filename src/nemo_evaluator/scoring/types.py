"""Scoring data types shared across all scorer implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScorerInput:
    """Input passed to scorer functions."""

    response: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
