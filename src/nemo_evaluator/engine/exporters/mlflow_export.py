# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""MLflow exporter: one run per bundle, artifact policy, idempotent re-export."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import mlflow  # noqa: F401

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "nemo-evaluator"

EXCLUDED_PATTERNS = ["*cache*", "*.db", "*.lock", "synthetic", "debug.json"]
REQUIRED_ARTIFACT_GLOBS = ("eval-*.json", "results.jsonl")


_MLFLOW_KEY_MAX = 250
_MLFLOW_PARAM_VAL_MAX = 250
_MLFLOW_TAG_VAL_MAX = 5000
_INVALID_KEY_CHARS = re.compile(r"[^/\w.\- ]")
_MULTI_UNDERSCORE = re.compile(r"_+")


def mlflow_sanitize(value: Any, kind: str = "key") -> str:
    """Sanitize ``value`` for MLflow's character / length rules."""
    s = "" if value is None else str(value)

    if kind in ("key", "metric", "tag_key", "param_key"):
        s = s.replace("pass@", "pass_at_")
        s = _INVALID_KEY_CHARS.sub("_", s)
        s = _MULTI_UNDERSCORE.sub("_", s).strip()
        return s[:_MLFLOW_KEY_MAX] or "key"

    s = s.replace("\n", " ").replace("\r", " ").strip()
    max_len = _MLFLOW_TAG_VAL_MAX if kind == "tag_value" else _MLFLOW_PARAM_VAL_MAX
    return s[:max_len]


def flatten_config(
    config: Any,
    parent_key: str = "",
    sep: str = ".",
    max_depth: int = 10,
) -> dict[str, str]:
    """Flatten a nested config into dot-notation keys, bounded by ``max_depth``."""
    if max_depth <= 0:
        return {parent_key: str(config)} if parent_key else {}

    if isinstance(config, dict):
        items: dict[str, str] = {}
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            items.update(flatten_config(value, new_key, sep, max_depth - 1))
        return items

    if isinstance(config, list):
        items = {}
        for idx, item in enumerate(config):
            item_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            items.update(flatten_config(item, item_key, sep, max_depth - 1))
        return items

    if not parent_key:
        return {}
    if config is None:
        return {parent_key: "null"}
    return {parent_key: str(config)}


def should_exclude_artifact(name: str) -> bool:
    """True if ``name`` matches ``EXCLUDED_PATTERNS`` (case-insensitive)."""
    n = name.lower()
    for pattern in EXCLUDED_PATTERNS:
        p = pattern.lower()
        if p.startswith("*") and p.endswith("*"):
            if p[1:-1] in n:
                return True
        elif p.startswith("*"):
            if n.endswith(p[1:]):
                return True
        elif n == p:
            return True
    return False


def get_copytree_ignore() -> Callable[[str, list[str]], list[str]]:
    def ignore_func(_directory: str, contents: list[str]) -> list[str]:
        return [name for name in contents if should_exclude_artifact(name)]

    return ignore_func


class MLflowExportSettings(BaseModel):
    """Configuration for :class:`MLflowExporter`."""

    model_config = ConfigDict(extra="forbid")

    tracking_uri: str | None = Field(default=None)
    experiment_name: str = Field(default=DEFAULT_EXPERIMENT_NAME)
    log_config_params: bool = Field(default=False)
    log_config_params_max_depth: int = Field(default=10)
    skip_existing: bool = Field(default=False)
    run_name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    tags: dict[str, str] | None = Field(default=None)
    extra_metadata: dict[str, Any] | None = Field(default=None)

    copy_artifacts: bool = Field(default=True)
    only_required: bool = Field(default=True)
    copy_logs: bool = Field(default=False)

    emit_traces: bool = Field(default=True)
    emit_traces_max_samples: int | None = Field(default=None)
    emit_traces_content_max: int = Field(default=4000)

    @model_validator(mode="before")
    @classmethod
    def _validate_tracking_uri(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        tracking_uri = data.get("tracking_uri")
        if not tracking_uri:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri and "://" not in tracking_uri:
            tracking_uri = os.getenv(tracking_uri, tracking_uri)
        if not tracking_uri:
            raise ValueError(
                "MLflow requires 'tracking_uri' to be configured. "
                "Set output.export_config.mlflow.tracking_uri or the "
                "MLFLOW_TRACKING_URI environment variable."
            )
        data["tracking_uri"] = tracking_uri
        return data


class MLflowExporter:
    """Push nel-next evaluation bundles to an MLflow tracking server."""

    def __init__(self, **kwargs: Any) -> None:
        self._settings = MLflowExportSettings.model_validate(kwargs)

    @property
    def _experiment_name(self) -> str:
        return self._settings.experiment_name

    @property
    def _tracking_uri(self) -> str | None:
        return self._settings.tracking_uri

    def export(
        self,
        bundles: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> None:
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow is required to use MLflowExporter: pip install mlflow")

        output_dir = Path((config or {}).get("output_dir", "."))

        mlflow.set_tracking_uri(self._settings.tracking_uri.rstrip("/"))
        mlflow.set_experiment(self._settings.experiment_name)

        successful, failed, skipped = 0, 0, 0
        for bundle in bundles:
            try:
                result = self._export_one_bundle(bundle, output_dir)
            except Exception as exc:
                bname = bundle.get("benchmark", {}).get("name", "?")
                logger.error("MLflow export failed for %s: %s", bname, exc, exc_info=True)
                failed += 1
                continue
            if result == "skipped":
                skipped += 1
            else:
                successful += 1

        logger.info(
            "MLflow export summary: experiment=%s successful=%d failed=%d skipped=%d",
            self._settings.experiment_name,
            successful,
            failed,
            skipped,
        )

    def _get_existing_run_id(self, job_id: str, experiment_name: str) -> str | None:
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return None
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.job_id = '{job_id}'",
            )
            if runs is None or getattr(runs, "empty", True):
                return None
            existing = runs.iloc[0]
            if len(runs) > 1:
                logger.warning(
                    "Multiple MLflow runs for job %s; using first: %s",
                    job_id,
                    existing.run_id,
                )
            else:
                logger.info(
                    "Found existing MLflow run for job %s: %s",
                    job_id,
                    existing.run_id,
                )
            return existing.run_id
        except Exception as exc:
            logger.debug(
                "Error searching for existing MLflow run for job %s: %s",
                job_id,
                exc,
            )
            return None

    def _build_params(self, bundle: dict[str, Any]) -> dict[str, str]:
        benchmark = bundle.get("benchmark", {}) or {}
        bundle_config = bundle.get("config", {}) or {}

        all_params: dict[str, Any] = {
            "run_id": bundle.get("run_id", ""),
            "benchmark": benchmark.get("name", ""),
            "timestamp": bundle.get("timestamp", ""),
            "sdk_version": bundle.get("sdk_version", ""),
            "config_hash": bundle.get("config_hash", ""),
            "n_results": bundle.get("n_results", ""),
        }
        output_path = bundle.get("_output_path")
        if output_path:
            all_params["results_dir"] = str(output_path)
        if self._settings.extra_metadata:
            all_params.update(self._settings.extra_metadata)
        if self._settings.log_config_params:
            all_params.update(
                flatten_config(
                    bundle_config,
                    parent_key="config",
                    max_depth=self._settings.log_config_params_max_depth,
                )
            )
        return {
            mlflow_sanitize(k, "param_key"): mlflow_sanitize(v, "param_value")
            for k, v in all_params.items()
            if v not in (None, "")
        }

    def _build_tags(self, bundle: dict[str, Any]) -> dict[str, str]:
        benchmark = bundle.get("benchmark", {}) or {}
        tags = {
            "job_id": bundle.get("run_id", ""),
            "benchmark": benchmark.get("name", ""),
            "config_hash": bundle.get("config_hash", ""),
            "sdk_version": bundle.get("sdk_version", ""),
        }
        if self._settings.tags:
            tags.update({k: v for k, v in self._settings.tags.items() if v})
        return {mlflow_sanitize(k, "tag_key"): mlflow_sanitize(v, "tag_value") for k, v in tags.items() if v}

    def _build_metrics(self, bundle: dict[str, Any]) -> dict[str, float]:
        scores = (bundle.get("benchmark", {}) or {}).get("scores", {}) or {}
        safe_metrics: dict[str, float] = {}
        for metric, val in scores.items():
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            if isinstance(val, (int, float)):
                safe_metrics[mlflow_sanitize(metric, "metric")] = float(val)
        return safe_metrics

    def _export_one_bundle(self, bundle: dict[str, Any], output_dir: Path) -> str:
        benchmark = bundle.get("benchmark", {}) or {}
        job_id = bundle.get("run_id") or f"unknown-{benchmark.get('name', '?')}"

        safe_params = self._build_params(bundle)
        safe_tags = self._build_tags(bundle)
        safe_metrics = self._build_metrics(bundle)

        existing_run_id = self._get_existing_run_id(job_id, self._settings.experiment_name)
        if existing_run_id and self._settings.skip_existing:
            logger.info(
                "Skipping MLflow export for job %s (run %s exists, skip_existing=True)",
                job_id,
                existing_run_id,
            )
            return "skipped"

        with mlflow.start_run(run_id=existing_run_id):
            if safe_tags:
                mlflow.set_tags(safe_tags)

            run_name = self._settings.run_name or f"eval-{job_id}"
            mlflow.set_tag("mlflow.runName", mlflow_sanitize(run_name, "tag_value"))

            if self._settings.description:
                mlflow.set_tag(
                    "mlflow.note.content",
                    mlflow_sanitize(self._settings.description, "tag_value"),
                )

            if not existing_run_id:
                mlflow.log_params(safe_params)
            else:
                logger.info(
                    "Skipping params for existing MLflow run %s (immutable by MLflow)",
                    existing_run_id,
                )

            if safe_metrics:
                mlflow.log_metrics(safe_metrics)

            self._log_artifacts(bundle, output_dir)

            if self._settings.emit_traces and not existing_run_id:
                self._emit_sample_traces(bundle)

        return "success"

    def _emit_sample_traces(self, bundle: dict[str, Any]) -> None:
        """Emit one MLflow Trace per sample for the Traces tab."""
        from nemo_evaluator.engine.exporters._trace_emit import emit_sample_trace

        results = bundle.get("_results") or []
        if not results:
            logger.debug("No _results on bundle; skipping trace emission")
            return

        bench_name = (bundle.get("benchmark", {}) or {}).get("name", "unknown")
        model_name = (bundle.get("config", {}) or {}).get("model", "unknown")
        content_max = self._settings.emit_traces_content_max
        cap = self._settings.emit_traces_max_samples

        emitted = 0
        skipped = 0
        tiers: dict[str, int] = {}
        for sample in results:
            if cap is not None and emitted >= cap:
                skipped += 1
                continue
            try:
                tier = emit_sample_trace(
                    sample,
                    model_name=model_name,
                    bench_name=bench_name,
                    content_max=content_max,
                )
                tiers[tier] = tiers.get(tier, 0) + 1
                emitted += 1
            except Exception as exc:
                logger.warning(
                    "Failed to emit MLflow trace for sample %s: %s",
                    sample.get("problem_idx", "?"),
                    exc,
                )
        logger.info(
            "MLflow traces: emitted=%d skipped=%d tiers=%s",
            emitted,
            skipped,
            tiers,
        )

    def _log_artifacts(self, bundle: dict[str, Any], output_dir: Path) -> None:
        if not self._settings.copy_artifacts:
            return

        benchmark = bundle.get("benchmark", {}) or {}
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", benchmark.get("name", "unknown"))
        task_dir = output_dir / safe_name

        if not task_dir.exists():
            # Fallback for in-process flows that export before materializing.
            with tempfile.TemporaryDirectory() as tmp:
                public = {k: v for k, v in bundle.items() if not k.startswith("_")}
                bp = Path(tmp) / f"{bundle.get('run_id', 'bundle')}.json"
                bp.write_text(json.dumps(public, indent=2, default=str))
                mlflow.log_artifact(str(bp), artifact_path=safe_name)
            return

        if self._settings.only_required:
            for pat in REQUIRED_ARTIFACT_GLOBS:
                for p in sorted(task_dir.glob(pat)):
                    if p.is_file() and not should_exclude_artifact(p.name):
                        mlflow.log_artifact(str(p), artifact_path=safe_name)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                staged = Path(tmp) / "artifacts"
                shutil.copytree(
                    task_dir,
                    staged,
                    ignore=get_copytree_ignore(),
                    dirs_exist_ok=True,
                )
                mlflow.log_artifacts(str(staged), artifact_path=safe_name)

        if self._settings.copy_logs:
            logs_dir = output_dir / "logs"
            if logs_dir.exists():
                for p in sorted(logs_dir.iterdir()):
                    if p.is_file() and not should_exclude_artifact(p.name):
                        mlflow.log_artifact(str(p), artifact_path=f"{safe_name}/logs")
