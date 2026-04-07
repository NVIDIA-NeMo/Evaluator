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
import os
import tempfile
from pathlib import Path

import ray


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
) -> dict:
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.observability.progress import NoOpProgress
    from nemo_evaluator.engine.eval_loop import run_evaluation
    from nemo_evaluator.engine.model_client import ModelClient
    from nemo_evaluator.engine.sharding import get_shard_range
    from nemo_evaluator.solvers import ChatSolver

    env = get_environment(benchmark)
    client = ModelClient(base_url=model_url, model=model_id, api_key=api_key, temperature=0.0)
    solver = ChatSolver(client, system_prompt=system_prompt)

    total = len(env)
    if max_problems:
        total = min(total, max_problems)
    problem_range = get_shard_range(total, shard_idx, total_shards)

    config = {
        "benchmark": benchmark,
        "model": model_id,
        "repeats": n_repeats,
        "shard": {"idx": shard_idx, "total": total_shards, "range": list(problem_range)},
    }

    bundle = asyncio.run(
        run_evaluation(
            env,
            solver,
            n_repeats=n_repeats,
            max_problems=max_problems,
            config=config,
            progress=NoOpProgress(),
            problem_range=problem_range,
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
    args = parser.parse_args()

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
        )
        for i in range(args.shards)
    ]

    print(f"Launched {args.shards} Ray tasks for {args.benchmark}")
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

    print(f"\nMerged {args.shards} shards:")
    for k, v in merged.get("benchmark", {}).get("scores", {}).items():
        if isinstance(v, dict) and "value" in v:
            print(f"  {k}: {v['value']:.4f}  [{v.get('ci_lower', '?'):.4f}, {v.get('ci_upper', '?'):.4f}]")
    print(f"  Output: {out}/")


if __name__ == "__main__":
    main()
