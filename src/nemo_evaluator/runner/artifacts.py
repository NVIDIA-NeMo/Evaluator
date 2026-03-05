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


def build_artifact_bundle(
    benchmark_name: str,
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    config: dict[str, Any],
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", benchmark_name)
    bundle: dict[str, Any] = {
        "run_id": f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe_name}",
        "config_hash": f"sha256:{config_hash}",
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
            c["category"]: {"mean_reward": c["mean_reward"], "n": c["n_samples"]}
            for c in categories
        }
    return bundle


def write_all(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

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
