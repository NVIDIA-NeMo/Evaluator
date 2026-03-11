"""Scoring data types shared across all scorer implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox


@dataclass
class ScorerInput:
    """Input passed to scorer functions.

    The ``sandbox`` field is available for scorers that need to inspect or
    execute commands inside a per-problem sandbox (e.g., running test suites
    after an agent modifies code).  Existing scorers that don't use it are
    unaffected -- the field defaults to ``None``.
    """

    response: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    sandbox: Sandbox | None = None
