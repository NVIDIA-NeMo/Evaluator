"""CLI commands for listing and running lm-eval harness tasks."""
from __future__ import annotations

import logging
from pathlib import Path

import click


@click.group("harness")
def harness_group():
    """Run evaluations through lm-eval harness."""


@harness_group.command("list")
@click.option("--generate-only", is_flag=True,
              help="Show only tasks compatible with chat APIs (no loglikelihood)")
def list_evals(generate_only):
    """List available lm-eval tasks."""
    try:
        from nemo_evaluator.environments.lm_eval import list_tasks, list_groups

        if generate_only:
            from nemo_evaluator.environments.lm_eval import list_generate_tasks
            gen_tasks = list_generate_tasks()
            click.echo(f"\nlm-eval ({len(gen_tasks)} generate-only tasks, chat API compatible):")
            for t in gen_tasks[:50]:
                click.echo(f"  lm-eval/{t}")
            if len(gen_tasks) > 50:
                click.echo(f"  ... and {len(gen_tasks) - 50} more")
        else:
            all_tasks = list_tasks()
            groups = list_groups()
            click.echo(f"\nlm-eval ({len(all_tasks)} tasks total):")
            click.echo("  Groups:")
            for g in groups[:30]:
                click.echo(f"    {g}")
            if len(groups) > 30:
                click.echo(f"    ... and {len(groups) - 30} more")
            click.echo(
                "\n  NOTE: Many tasks require loglikelihood (not supported via chat API)."
                "\n  Use --generate-only to see compatible tasks."
            )
        click.echo("\n  Use: nel run --benchmark lm-eval/<task>")
    except ImportError as exc:
        click.echo(f"\nlm-eval: not available ({exc})")


@harness_group.command("run")
@click.option("--tasks", "-t", required=True, help="Comma-separated lm-eval task names")
@click.option("--model-url", envvar="NEMO_MODEL_URL", required=True)
@click.option("--model-id", envvar="NEMO_MODEL_ID", required=True)
@click.option("--api-key", envvar="NEMO_API_KEY", default=None)
@click.option("--examples", type=int, default=None, help="Limit number of examples")
@click.option("--fewshot", type=int, default=None, help="Number of few-shot examples")
@click.option("--repeats", "-n", type=int, default=1)
@click.option("--temperature", type=float, default=0.0)
@click.option("--max-tokens", type=int, default=2048)
@click.option("--cache-dir", envvar="NEL_CACHE_DIR", default=None)
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--verbose", "-v", is_flag=True)
def run_harness(tasks, model_url, model_id, api_key,
                examples, fewshot, repeats, temperature, max_tokens,
                cache_dir, output_dir, verbose):
    """Run lm-eval tasks (convenience wrapper for nel run)."""
    import asyncio

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from nemo_evaluator.environments.lm_eval import LMEvalEnvironment
    from nemo_evaluator.observability.progress import ConsoleProgress
    from nemo_evaluator.runner.artifacts import write_all
    from nemo_evaluator.runner.eval_loop import run_evaluation
    from nemo_evaluator.runner.model_client import ModelClient
    from nemo_evaluator.runner.solver import ChatSolver

    client = ModelClient(
        base_url=model_url, model=model_id, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens, cache_dir=cache_dir,
    )
    solver = ChatSolver(client)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    task_list = [t.strip() for t in tasks.split(",")]
    for task_name in task_list:
        env = LMEvalEnvironment(task_name=task_name, num_fewshot=fewshot, limit=examples)
        click.echo(f"\n{'='*60}\n  lm-eval/{task_name} ({len(env)} items)\n{'='*60}")
        bundle = asyncio.run(run_evaluation(
            env, solver, n_repeats=repeats,
            config={"benchmark": f"lm-eval/{task_name}", "model": model_id},
            progress=ConsoleProgress(),
        ))
        task_dir = str(out / task_name) if len(task_list) > 1 else str(out)
        write_all(bundle, task_dir)
