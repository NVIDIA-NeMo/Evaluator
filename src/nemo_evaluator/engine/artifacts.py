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
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemo_evaluator import __version__
from nemo_evaluator.observability.collector import ArtifactCollector
from nemo_evaluator.observability.types import RunArtifacts


def _config_hash(config: dict[str, Any]) -> str:
    """Deterministic config hash covering model, temperature, system prompt, and dataset."""
    keys_to_hash = [
        "model",
        "model_url",
        "temperature",
        "max_tokens",
        "top_p",
        "seed",
        "system_prompt",
        "benchmark",
        "repeats",
        "max_problems",
        "shuffle_seed",
    ]
    subset = {k: config.get(k) for k in keys_to_hash if config.get(k) is not None}
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()


def build_artifact_bundle(
    benchmark_name: str,
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    config: dict[str, Any],
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ch = _config_hash(config)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", benchmark_name)
    bundle: dict[str, Any] = {
        "run_id": f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe_name}",
        "config_hash": f"sha256:{ch}",
        "sdk_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "benchmark": {
            "name": benchmark_name,
            "samples": len({r["problem_idx"] for r in results}),
            "repeats": config.get("repeats", 1),
            "scores": metrics,
        },
        "n_results": len(results),
    }
    if categories:
        bundle["benchmark"]["categories"] = {
            c["category"]: {"mean_reward": c["mean_reward"], "n": c["n_samples"]} for c in categories
        }
    return bundle


def write_all(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if "run_id" not in bundle:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", bundle.get("benchmark", {}).get("name", "unknown"))
        bundle["run_id"] = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe_name}"
        bundle.setdefault("sdk_version", __version__)
        bundle.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        bundle.setdefault("config_hash", "")

    # Bundle JSON (public keys only)
    public = {k: v for k, v in bundle.items() if not k.startswith("_")}
    bp = out / f"{bundle['run_id']}.json"
    bp.write_text(json.dumps(public, indent=2, default=str))
    paths["bundle"] = bp

    # Results JSONL
    results = bundle.get("_results", [])
    rp = out / "results.jsonl"
    with rp.open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    paths["results"] = rp

    # Observability artifacts (trajectories, runtime stats, failure analysis)
    artifacts: RunArtifacts | None = bundle.get("_artifacts")
    if artifacts:
        obs_paths = ArtifactCollector.write(artifacts, out, bundle["run_id"])
        paths.update(obs_paths)

    return paths
