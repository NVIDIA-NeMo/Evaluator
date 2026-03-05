#!/usr/bin/env python3
"""PI integration: decompose a verifiers SingleTurnEnv into Evaluator's
seed/verify loop with full observability."""

from __future__ import annotations

import asyncio
import os


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", "-e", required=True, help="verifiers environment name")
    p.add_argument("--repeats", "-n", type=int, default=1)
    p.add_argument("--max-problems", type=int, default=10)
    p.add_argument("--output-dir", "-o", default="./eval_results/pi")
    args = p.parse_args()

    from nemo_evaluator.adapters.pi import PIAdapter
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient

    adapter = PIAdapter(args.env)
    client = ModelClient(
        base_url=os.getenv("MODEL_URL", "https://inference-api.nvidia.com/v1"),
        model=os.getenv("MODEL_ID", "azure/openai/gpt-5.2"),
        api_key=os.getenv("NEMO_API_KEY", ""),
    )

    bundle = asyncio.run(run_evaluation(
        adapter, client, n_repeats=args.repeats, max_problems=args.max_problems,
        system_prompt=adapter.system_prompt, progress=ConsoleProgress()))

    paths = write_all(bundle, args.output_dir)
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
