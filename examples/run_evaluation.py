#!/usr/bin/env python3
"""Programmatic evaluation with the Python API (no CLI).

Shows how to use nemo_evaluator as a library: load a benchmark,
create a model client, run evaluation, and write artifacts.

Usage:
    pip install -e ".[scoring]"
    python examples/run_evaluation.py
    NEMO_MODEL_URL=https://api.example.com/v1 NEMO_MODEL_ID=my-model python examples/run_evaluation.py
"""

import argparse
import asyncio
import logging
import os

from nemo_evaluator.environments.registry import get_environment
from nemo_evaluator.observability.progress import ConsoleProgress
from nemo_evaluator.runner.artifacts import write_all
from nemo_evaluator.runner.eval_loop import run_evaluation
from nemo_evaluator.runner.model_client import ModelClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def main(benchmark: str, repeats: int, max_problems: int, output_dir: str):
    env = get_environment(benchmark)
    client = ModelClient(
        base_url=os.environ["NEMO_MODEL_URL"],
        model=os.environ["NEMO_MODEL_ID"],
        api_key=os.getenv("NEMO_API_KEY"),
        temperature=0.0,
        max_concurrent=4,
    )

    bundle = await run_evaluation(
        env, client,
        n_repeats=repeats,
        max_problems=max_problems,
        progress=ConsoleProgress(),
    )

    print(f"\n{'='*60}")
    print(f"{benchmark} – {bundle['benchmark']['samples']} problems x {repeats} repeats")
    print(f"{'='*60}")
    for metric, val in bundle["benchmark"]["scores"].items():
        if isinstance(val, dict) and "value" in val:
            print(f"  {metric}: {val['value']:.4f}  [{val['ci_lower']:.4f}, {val['ci_upper']:.4f}]")

    paths = write_all(bundle, output_dir)
    print(f"\nArtifacts:")
    for name, p in paths.items():
        print(f"  {name}: {p}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", "-b", default="gsm8k")
    p.add_argument("--repeats", "-n", type=int, default=2)
    p.add_argument("--max-problems", type=int, default=10)
    p.add_argument("--output-dir", "-o", default="./eval_results")
    args = p.parse_args()
    asyncio.run(main(args.benchmark, args.repeats, args.max_problems, args.output_dir))
