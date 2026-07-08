# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write-once snapshot of the composed run config for reproducibility.

The per-shard ``config_*.yaml`` files are transformed for execution
(model URL runtime-assigned, cluster forced to ``local``) and cannot
serve as the run record. This module persists the composed config
instead, with ``${ENV_VAR}`` references NOT expanded — so secrets never
reach disk by construction — plus a provenance header. Configs without a
pre-expansion source (quick mode, programmatic) get no snapshot: the
validated config holds resolved secrets and must not be persisted.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "full_config.yaml"


def package_version() -> str:
    """Best-effort nemo-evaluator package version."""
    try:
        from importlib.metadata import version

        return version("nemo-evaluator")
    except Exception:  # pragma: no cover - metadata missing in odd installs
        return "unknown"


def build_provenance(
    source_config: str | None = None,
    overrides: tuple[str, ...] | list[str] = (),
    run_id: str | None = None,
) -> dict[str, str]:
    """Provenance metadata recorded in the snapshot header."""
    prov: dict[str, str] = {
        "nemo-evaluator version": package_version(),
        "created": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
    }
    if source_config:
        prov["source config"] = source_config
    if overrides:
        prov["overrides"] = " ".join(overrides)
    if run_id:
        prov["run_id"] = run_id
    return prov


def build_snapshot_text(raw: dict[str, Any], provenance: dict[str, str]) -> str:
    """Render the snapshot file: provenance header + config YAML."""
    lines = [
        "# Re-run with:  nel eval run <this file>",
    ]
    for k, v in provenance.items():
        lines.append(f"# {k}: {v}")
    header = "\n".join(lines) + "\n"
    body = yaml.dump(raw, default_flow_style=False, sort_keys=False)
    return header + body


def record_output_dir_override(config: Any, output_dir: str) -> None:
    """Sync a CLI ``-o`` override into the captured dict so the snapshot
    records where the run actually wrote."""
    raw = getattr(config, "_composed_raw", None)
    if isinstance(raw, dict):
        raw.setdefault("output", {})["dir"] = output_dir


def write_config_snapshot(config: Any, output_dir: str | Path | None = None, *, force: bool = False) -> Path | None:
    """Write the composed-config snapshot into *output_dir*.

    Only the pre-expansion composed dict is persisted (env refs intact,
    no secrets by construction). Existing snapshots are kept unless
    *force*: resumes preserve the original record, fresh runs into a
    reused dir overwrite it. Skipped in inner shard executions. Never
    raises — must not break a run.
    """
    if os.environ.get("NEL_INNER_EXECUTION") == "1":
        return None
    try:
        raw = getattr(config, "_composed_raw", None)
        if raw is None:
            # No pre-expansion source (quick mode / programmatic): the
            # validated config holds resolved secrets, so write nothing.
            return None
        out_dir = Path(output_dir or config.output.dir)
        path = out_dir / SNAPSHOT_FILENAME
        if path.exists() and not force:
            logger.debug("Config snapshot already exists (kept, force=False): %s", path)
            return path

        provenance = dict(getattr(config, "_snapshot_provenance", None) or {})
        if not provenance:
            provenance = build_provenance()

        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(build_snapshot_text(raw, provenance), encoding="utf-8")
        logger.info("Saved run config snapshot: %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - never break a run over the snapshot
        logger.warning("Could not write run config snapshot: %s", exc)
        return None
