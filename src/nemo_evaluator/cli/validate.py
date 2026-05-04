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

import asyncio
import logging

import click


@click.command("validate")
@click.option("--benchmark", "-b", required=True)
@click.option("--samples", "-s", default=5, type=int)
@click.option("--model-url", required=True, envvar="NEMO_MODEL_URL", help="Model endpoint URL (or set NEMO_MODEL_URL)")
@click.option("--model-id", required=True, envvar="NEMO_MODEL_ID", help="Model identifier (or set NEMO_MODEL_ID)")
@click.option("--api-key", envvar="NEMO_API_KEY")
@click.option("--verbose", "-v", is_flag=True)
def validate_cmd(benchmark, samples, model_url, model_id, api_key, verbose):
    """Quick sanity check: run a few samples and report."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from nemo_evaluator.engine.eval_loop import run_evaluation
    from nemo_evaluator.engine.model_client import ModelClient
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.solvers import ChatSolver

    try:
        env = get_environment(benchmark)
    except KeyError:
        raise click.ClickException(f"Unknown benchmark: {benchmark!r}. Run 'nel list'.")

    client = ModelClient(base_url=model_url, model=model_id, api_key=api_key)
    solver = ChatSolver(client)

    bundle = asyncio.run(run_evaluation(env, solver, n_repeats=1, max_problems=samples, progress=ConsoleProgress()))
    results = bundle.get("_results", [])

    correct = sum(1 for r in results if r["reward"] > 0)
    click.echo(f"\n{correct}/{len(results)} correct")
    for r in results:
        s = "PASS" if r["reward"] > 0 else "FAIL"
        click.echo(
            f"  [{s}] p{r['problem_idx']}: "
            f"expected={r['expected_answer']!r} got={r.get('extracted_answer', '?')!r} "
            f"({r.get('latency_ms', 0):.0f}ms {r.get('tokens', 0)}tok)"
        )
