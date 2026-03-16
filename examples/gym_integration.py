#!/usr/bin/env python3
"""Gym integration examples: serve, consume, and export.

Usage:
    pip install -e ".[scoring]"

    # 1. Serve gsm8k for Gym training (equivalent to: nel serve -b gsm8k --gym-compat)
    python examples/gym_integration.py serve

    # 2. Consume a running server with full Evaluator metrics
    python examples/gym_integration.py consume --endpoint http://localhost:9090

    # 3. Export a benchmark as JSONL for ng_collect_rollouts
    python examples/gym_integration.py export
"""

from __future__ import annotations

import argparse
import asyncio
import os


def serve(args):
    import uvicorn

    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.serving.app import generate_app

    env = get_environment(args.benchmark)
    app = generate_app(env, gym_compat=True)
    print(f"Serving {env.name} (gym-compat) on :{args.port}, size={len(env)}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def consume(args):
    from nemo_evaluator.environments.gym import GymEnvironment
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient
    from nemo_evaluator.solvers import ChatSolver

    env = GymEnvironment(args.endpoint)
    client = ModelClient(
        base_url=os.environ["NEMO_MODEL_URL"],
        model=os.environ["NEMO_MODEL_ID"],
        api_key=os.getenv("NEMO_API_KEY"),
    )
    solver = ChatSolver(client)
    bundle = asyncio.run(run_evaluation(
        env, solver, n_repeats=args.repeats, max_problems=args.max_problems,
        progress=ConsoleProgress()))
    paths = write_all(bundle, args.output_dir)
    for k, p in paths.items():
        print(f"  {k}: {p}")


def export(args):
    """Export a benchmark as JSONL by querying a served environment."""
    import json

    from nemo_evaluator.environments.registry import get_environment

    env = get_environment(args.benchmark)
    size = asyncio.run(env.dataset_size()) if hasattr(env, "dataset_size") else len(env)
    output_path = f"{args.output_dir}/{args.benchmark}.jsonl"
    os.makedirs(args.output_dir, exist_ok=True)

    async def _export():
        with open(output_path, "w") as f:
            for i in range(size):
                seed = await env.seed(i)
                row = {
                    "responses_create_params": {"input": [{"role": "user", "content": seed.prompt}]},
                    "expected_answer": seed.expected_answer,
                    "uuid": f"{args.benchmark}-{i}",
                    "metadata": seed.metadata,
                }
                f.write(json.dumps(row) + "\n")

    asyncio.run(_export())
    print(f"Exported {size} rows -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gym integration examples")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Serve a benchmark for Gym training")
    s.add_argument("--benchmark", "-b", default="gsm8k")
    s.add_argument("--port", "-p", type=int, default=9090)
    s.set_defaults(func=serve)

    c = sub.add_parser("consume", help="Run eval against a remote environment server")
    c.add_argument("--endpoint", required=True)
    c.add_argument("--repeats", "-n", type=int, default=2)
    c.add_argument("--max-problems", type=int, default=10)
    c.add_argument("--output-dir", "-o", default="./eval_results/gym")
    c.set_defaults(func=consume)

    e = sub.add_parser("export", help="Export benchmark as JSONL for ng_collect_rollouts")
    e.add_argument("--benchmark", "-b", default="gsm8k")
    e.add_argument("--output-dir", "-o", default="/tmp/evaluator_data")
    e.set_defaults(func=export)

    args = p.parse_args()
    args.func(args)
