"""ExportPlugin protocol."""
from __future__ import annotations

from typing import Any, Protocol


class ExportPlugin(Protocol):
    def export(self, bundles: list[dict[str, Any]], config: dict[str, Any] | None = None) -> None:
        """Push evaluation results to an external tracking system."""
        ...
