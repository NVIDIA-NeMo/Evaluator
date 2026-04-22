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
"""Ray-based distributed evaluation launcher.

Shards a benchmark across Ray workers, merges results.
Compatible with Ray clusters used by NeMo Gym for training.

Usage:
    ray job submit --working-dir . -- python -m nemo_evaluator.engine.ray_launcher \
        --benchmark gsm8k --shards 8 --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import ray

logger = logging.getLogger(__name__)


@ray.remote
def run_shard(
    benchmark: str,
    shard_idx: int,
    total_shards: int,
    model_url: str,
    model_id: str,
    api_key: str | None,
    n_repeats: int,
    max_problems: int | None,
    system_prompt: str | None,
    shuffle_seed: int | None = 42,
) -> dict:
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.observability.progress import NoOpProgress
    from nemo_evaluator.engine.eval_loop import run_evaluation
    from nemo_evaluator.engine.model_client import ModelClient
    from nemo_evaluator.solvers import ChatSolver

    env = get_environment(benchmark)
    client = ModelClient(base_url=model_url, model=model_id, api_key=api_key, temperature=0.0)
    solver = ChatSolver(client, system_prompt=system_prompt)

    config = {
        "benchmark": benchmark,
        "model": model_id,
        "repeats": n_repeats,
    }

    # Pass shard_info (not problem_range) so shuffle_seed applies uniformly.
    bundle = asyncio.run(
        run_evaluation(
            env,
            solver,
            n_repeats=n_repeats,
            max_problems=max_problems,
            config=config,
            progress=NoOpProgress(),
            shard_info=(shard_idx, total_shards),
            shuffle_seed=shuffle_seed,
        )
    )

    bundle.pop("_artifacts", None)
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", required=True)
    parser.add_argument("--shards", "-s", type=int, default=4)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument(
        "--model-url", default=os.environ.get("NEMO_MODEL_URL"), required="NEMO_MODEL_URL" not in os.environ
    )
    parser.add_argument(
        "--model-id", default=os.environ.get("NEMO_MODEL_ID"), required="NEMO_MODEL_ID" not in os.environ
    )
    parser.add_argument("--api-key", default=os.environ.get("NEMO_API_KEY"))
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--output-dir", "-o", default="./eval_results/ray")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed for deterministic shuffling of sample order across shards (pass -1 to disable).",
    )
    args = parser.parse_args()

    shuffle_seed: int | None = None if args.shuffle_seed < 0 else args.shuffle_seed

    ray.init()

    futures = [
        run_shard.remote(
            args.benchmark,
            i,
            args.shards,
            args.model_url,
            args.model_id,
            args.api_key,
            args.repeats,
            args.max_problems,
            args.system_prompt,
            shuffle_seed,
        )
        for i in range(args.shards)
    ]

    logger.info("Launched %d Ray tasks for %s", args.shards, args.benchmark)
    shard_bundles = ray.get(futures)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, bundle in enumerate(shard_bundles):
            shard_dir = Path(tmpdir) / f"shard_{i}"
            shard_dir.mkdir()
            results = bundle.pop("_results", [])
            (shard_dir / "results.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in results) + "\n")
            bundle_path = shard_dir / f"{bundle.get('run_id', f'shard_{i}')}.json"
            bundle_path.write_text(json.dumps(bundle, indent=2, default=str))

        from nemo_evaluator.engine.sharding import merge_results

        shard_dirs = sorted(Path(tmpdir).glob("shard_*"))
        merged = merge_results(shard_dirs, out, n_repeats=args.repeats)

    logger.info("Merged %d shards:", args.shards)
    for k, v in merged.get("benchmark", {}).get("scores", {}).items():
        if isinstance(v, dict) and "value" in v:
            lo = v.get("ci_lower")
            hi = v.get("ci_upper")
            ci_str = f"  [{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else ""
            logger.info("  %s: %.4f%s", k, v["value"], ci_str)
    logger.info("  Output: %s/", out)


if __name__ == "__main__":
    main()
