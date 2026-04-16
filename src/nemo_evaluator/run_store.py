# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unified run store: track evaluations across all executors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.config import EvalConfig

logger = logging.getLogger(__name__)

_RUN_META = "nel_run.json"


def _runs_store() -> Path:
    """Central index of all tracked runs, regardless of executor."""
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "nel" / "runs"


@dataclass
class RunMeta:
    run_id: str
    executor: str
    output_dir: str
    started_at: str
    config_summary: str = ""
    details: dict = field(default_factory=dict)

    def save(self) -> None:
        """Write to both the output dir and the central store.

        The output-dir write is best-effort: for remote executors (e.g.
        SLURM) the path may not be locally accessible.
        """
        data = json.dumps(asdict(self), indent=2)

        try:
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / _RUN_META).write_text(data, encoding="utf-8")
        except OSError:
            logger.debug("Could not write %s to output dir %s (remote path?)", _RUN_META, self.output_dir)

        store_dir = _runs_store() / self.run_id
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / _RUN_META).write_text(data, encoding="utf-8")

    def update_details(self, **kwargs) -> None:
        """Merge new keys into details and re-save."""
        self.details.update(kwargs)
        self.save()


def generate_run_id(config: "EvalConfig") -> str:
    """Generate a timestamped run ID: YYYYMMDD_HHMMSS_{hash8}.

    The hash is derived from benchmark names and model identifiers to
    avoid collisions when multiple runs are submitted in the same second.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts: list[str] = []
    for b in config.benchmarks:
        parts.append(b.name)
        svc_name = getattr(b.solver, "service", "")
        svc = config.services.get(svc_name) if svc_name else None
        if svc:
            parts.append(getattr(svc, "model", "") or "")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]
    return f"{ts}_{digest}"


def config_summary(config: "EvalConfig") -> str:
    """One-line summary of benchmarks in a config."""
    names = [b.name for b in config.benchmarks]
    if len(names) == 1:
        return names[0]
    return f"{names[0]} (+{len(names) - 1} more)"


def load_run_meta(output_dir: str | Path) -> RunMeta | None:
    """Load RunMeta from an output directory."""
    p = Path(output_dir) / _RUN_META
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return RunMeta(**data)
    except (json.JSONDecodeError, TypeError, OSError):
        return None


def resolve_run(
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    job_id: str | None = None,
    host: str | None = None,
) -> RunMeta | None:
    """Resolve a RunMeta from any combination of identifiers.

    Resolution order:
    1. run_id → central store lookup
    2. output_dir → nel_run.json in directory
    3. job_id → scan store for matching SLURM job_id in details
    """
    if run_id:
        p = _runs_store() / run_id / _RUN_META
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return RunMeta(**data)
            except (json.JSONDecodeError, TypeError, OSError):
                pass

    if output_dir:
        meta = load_run_meta(output_dir)
        if meta:
            return meta

    if job_id:
        store = _runs_store()
        if store.is_dir():
            for meta_path in store.glob(f"*/{_RUN_META}"):
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    details = data.get("details", {})
                    if details.get("job_id") == job_id or job_id in details.get("job_ids", []):
                        if host and details.get("hostname") != host:
                            continue
                        return RunMeta(**data)
                except (json.JSONDecodeError, TypeError, OSError):
                    continue

    return None


def list_runs() -> list[RunMeta]:
    """List all tracked runs from the central store, newest first."""
    store = _runs_store()
    if not store.is_dir():
        return []

    runs: list[RunMeta] = []
    for meta_path in store.glob(f"*/{_RUN_META}"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(RunMeta(**data))
        except (json.JSONDecodeError, TypeError, OSError):
            continue

    runs.sort(key=lambda r: r.started_at, reverse=True)
    return runs


def remove_run(run_id: str) -> bool:
    """Remove a run entry from the central store."""
    import shutil

    store_dir = _runs_store() / run_id
    if store_dir.is_dir():
        shutil.rmtree(store_dir, ignore_errors=True)
        return True
    return False
