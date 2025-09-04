"""Exporter registration and factory."""

from nemo_evaluator_launcher.exporters.gsheets import GSheetsExporter
from nemo_evaluator_launcher.exporters.local import LocalExporter
from nemo_evaluator_launcher.exporters.mlflow import MLflowExporter
from nemo_evaluator_launcher.exporters.registry import available_exporters, get_exporter
from nemo_evaluator_launcher.exporters.wandb import WandBExporter


def create_exporter(name: str, config: dict = None):
    return get_exporter(name)(config or {})
