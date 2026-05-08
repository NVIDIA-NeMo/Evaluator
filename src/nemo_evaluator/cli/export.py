# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""``nel export`` — post-hoc export of a completed run to a registered exporter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _parse_override(token: str) -> tuple[str, Any]:
    if "=" not in token:
        raise click.BadParameter(f"Override '{token}' must be in 'key=value' form (e.g. -o tracking_uri=http://...).")
    key, _, value = token.partition("=")
    key = key.strip()
    if not key:
        raise click.BadParameter(f"Override '{token}' has empty key.")
    return key, _parse_value(value)


def _load_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise click.ClickException(f"PyYAML is required to load {path}: pip install pyyaml") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


@click.command("export")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--dest",
    "-d",
    required=True,
    help="Destination exporter name (e.g. mlflow, inspect, wandb).",
)
@click.option(
    "--override",
    "-o",
    "overrides",
    multiple=True,
    help="Override exporter kwargs: 'key=value' (repeatable, values parsed as JSON when possible).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Load exporter kwargs from YAML or JSON file (merged before --override).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output_dir passed to the exporter (defaults to the first bundle's parent).",
)
def export_cmd(
    paths: tuple[Path, ...],
    dest: str,
    overrides: tuple[str, ...],
    config_path: Path | None,
    output_dir: Path | None,
) -> None:
    """Export bundles at PATHS to a registered exporter (mlflow / inspect / wandb)."""
    from nemo_evaluator.engine.exporters import get_exporter
    from nemo_evaluator.reports.eval import (
        discover_bundle_paths,
        load_bundles_for_export,
    )

    bundle_paths = discover_bundle_paths(list(paths))
    if not bundle_paths:
        raise click.ClickException(
            "No eval-*.json bundle files found under the given PATHS. Pass a run directory or bundle file directly."
        )

    bundles = load_bundles_for_export(bundle_paths)
    if not bundles:
        raise click.ClickException("No valid bundles could be loaded.")

    exporter_kwargs: dict[str, Any] = {}
    if config_path is not None:
        loaded = _load_config_file(config_path)
        if not isinstance(loaded, dict):
            raise click.ClickException(f"Config file {config_path} must hold a mapping at the top level.")
        if dest in loaded and isinstance(loaded[dest], dict):
            exporter_kwargs.update(loaded[dest])
        else:
            exporter_kwargs.update(loaded)

    for token in overrides:
        key, value = _parse_override(token)
        exporter_kwargs[key] = value

    if output_dir is not None:
        effective_output_dir = output_dir
    else:
        bench_dir = Path(bundles[0].get("_output_path", "."))
        effective_output_dir = bench_dir.parent.resolve()

    try:
        exporter = get_exporter(dest, **exporter_kwargs)
    except TypeError as exc:
        raise click.ClickException(f"Exporter '{dest}' rejected the provided kwargs: {exc}") from exc
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Failed to construct exporter '{dest}': {exc}") from exc

    click.echo(f"Exporting {len(bundles)} bundle(s) from {effective_output_dir} via '{dest}'...")
    try:
        exporter.export(bundles, config={"output_dir": str(effective_output_dir)})
    except Exception as exc:
        raise click.ClickException(f"Export to {dest} failed: {exc}") from exc
    click.echo(f"Export to {dest} complete.")
