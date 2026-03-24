"""Export plugins: push evaluation results to experiment trackers."""

from __future__ import annotations

from typing import Any

from nemo_evaluator.engine.exporters.base import ExportPlugin

_REGISTRY: dict[str, type[ExportPlugin]] = {}


def register_exporter(name: str, cls: type[ExportPlugin]) -> None:
    _REGISTRY[name] = cls


def get_exporter(name: str, **kwargs: Any) -> ExportPlugin:
    if name not in _REGISTRY:
        _lazy_load()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown exporter {name!r}. Available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def _lazy_load() -> None:
    from nemo_evaluator.engine.exporters.wandb_export import WandBExporter

    register_exporter("wandb", WandBExporter)

    from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

    register_exporter("mlflow", MLflowExporter)

    from nemo_evaluator.engine.exporters.inspect_export import InspectExporter

    register_exporter("inspect", InspectExporter)
