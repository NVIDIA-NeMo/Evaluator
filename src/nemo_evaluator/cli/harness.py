"""CLI commands for running external harnesses: simple-evals, lm-eval."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click


@click.group("harness")
def harness_group():
    """Run evaluations through external harnesses (simple-evals, lm-eval)."""


# --- nel harness list ---

@harness_group.command("list")
@click.option("--harness", "-h", type=click.Choice(["simple-evals", "lm-eval", "all"]), default="all")
@click.option("--repo-path", default=None, help="Path to simple-evals repo (if not installed)")
@click.option("--generate-only", is_flag=True,
              help="lm-eval: show only tasks compatible with chat APIs (no loglikelihood)")
def list_evals(harness, repo_path, generate_only):
    """List available evaluations from external harnesses."""
    if harness in ("simple-evals", "all"):
        try:
            from nemo_evaluator.harnesses.simple_evals import list_evals as se_list
            evals = se_list(repo_path)
            click.echo("\nsimple-evals (all generation-based, fully compatible):")
            for e in evals:
                click.echo(f"  {e['name']:<20s} ({e['class']})")
        except ImportError as exc:
            click.echo(f"\nsimple-evals: not available ({exc})")

    if harness in ("lm-eval", "all"):
        try:
            from nemo_evaluator.harnesses.lm_eval import list_tasks, list_groups

            if generate_only:
                from nemo_evaluator.harnesses.lm_eval import list_generate_tasks
                gen_tasks = list_generate_tasks()
                click.echo(f"\nlm-eval ({len(gen_tasks)} generate-only tasks, chat API compatible):")
                for t in gen_tasks[:50]:
                    click.echo(f"  {t}")
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
            click.echo("\n  Use: nel harness run --harness lm-eval --tasks <name>")
        except ImportError as exc:
            click.echo(f"\nlm-eval: not available ({exc})")


# --- nel harness run ---

@harness_group.command("run")
@click.option("--harness", "-h", required=True, type=click.Choice(["simple-evals", "lm-eval"]))
@click.option("--eval", "-e", "eval_names", default=None, help="simple-evals: comma-separated eval names")
@click.option("--tasks", "-t", default=None, help="lm-eval: comma-separated task names")
@click.option("--model-url", envvar="NEMO_MODEL_URL", required=True)
@click.option("--model-id", envvar="NEMO_MODEL_ID", required=True)
@click.option("--api-key", envvar="NEMO_API_KEY", default=None)
@click.option("--examples", type=int, default=None, help="Limit number of examples")
@click.option("--fewshot", type=int, default=None, help="lm-eval: number of few-shot examples")
@click.option("--repeats", "-n", type=int, default=1)
@click.option("--temperature", type=float, default=0.0)
@click.option("--max-tokens", type=int, default=2048)
@click.option("--cache-dir", envvar="NEL_CACHE_DIR", default=None)
@click.option("--repo-path", default=None, help="Path to simple-evals repo")
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--verbose", "-v", is_flag=True)
def run_harness(harness, eval_names, tasks, model_url, model_id, api_key,
                examples, fewshot, repeats, temperature, max_tokens,
                cache_dir, repo_path, output_dir, verbose):
    """Run evaluation through an external harness."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from nemo_evaluator.runner.model_client import ModelClient
    client = ModelClient(
        base_url=model_url, model=model_id, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens, cache_dir=cache_dir,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if harness == "simple-evals":
        _run_simple_evals(client, eval_names, examples, repeats, repo_path, out)
    elif harness == "lm-eval":
        _run_lm_eval(client, tasks, fewshot, examples, out)


def _run_simple_evals(client, eval_names, examples, repeats, repo_path, out):
    from nemo_evaluator.harnesses.simple_evals import run_simple_eval
    from nemo_evaluator.metrics.confidence import bootstrap_ci

    if not eval_names:
        raise click.ClickException("--eval is required for simple-evals (e.g. --eval mmlu,math)")

    names = [n.strip() for n in eval_names.split(",")]

    for name in names:
        click.echo(f"\n{'='*60}\n  simple-evals/{name}\n{'='*60}")

        all_scores: list[float] = []
        all_steps = []

        for rep in range(repeats):
            if repeats > 1:
                click.echo(f"  repeat {rep+1}/{repeats}")

            result = run_simple_eval(
                eval_name=name,
                client=client,
                repo_path=repo_path,
                num_examples=examples,
            )
            if result.score is not None:
                all_scores.append(result.score)
            all_steps.extend(result.step_records)

        score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        click.echo(f"\n  Score: {score:.4f}")

        if len(all_scores) > 1:
            ci = bootstrap_ci(all_scores)
            click.echo(f"  95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")

        click.echo(f"  Model calls: {len(all_steps)}")

        bundle = {
            "harness": "simple-evals",
            "eval": name,
            "score": score,
            "scores": all_scores,
            "metrics": result.metrics if result else {},
            "n_samples": result.n_samples if result else 0,
            "n_model_calls": len(all_steps),
            "repeats": repeats,
        }
        bp = out / f"simple-evals-{name}.json"
        bp.write_text(json.dumps(bundle, indent=2, default=str))
        click.echo(f"  Output: {bp}")


def _run_lm_eval(client, tasks_str, fewshot, limit, out):
    from nemo_evaluator.harnesses.lm_eval import run_lm_eval

    if not tasks_str:
        raise click.ClickException("--tasks is required for lm-eval (e.g. --tasks arc_easy,mmlu)")

    task_list = [t.strip() for t in tasks_str.split(",")]

    click.echo(f"\n{'='*60}\n  lm-eval: {', '.join(task_list)}\n{'='*60}")

    result = run_lm_eval(
        tasks=task_list,
        client=client,
        num_fewshot=fewshot,
        limit=limit,
    )

    for task_name, metrics in result.results.items():
        click.echo(f"\n  {task_name}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                click.echo(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    click.echo(f"\n  Model calls: {len(result.step_records)}")

    bundle = {
        "harness": "lm-eval",
        "tasks": task_list,
        "results": result.results,
        "n_samples": result.n_samples,
        "n_model_calls": len(result.step_records),
    }
    bp = out / "lm-eval-results.json"
    bp.write_text(json.dumps(bundle, indent=2, default=str))
    click.echo(f"  Output: {bp}")
