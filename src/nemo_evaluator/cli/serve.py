from __future__ import annotations

import click


@click.command("serve")
@click.option("--benchmark", "-b", required=True)
@click.option("--host", default="0.0.0.0")
@click.option("--port", "-p", default=9090, type=int)
@click.option("--gym-compat", is_flag=True, help="Accept NeMoGymResponse in /verify")
@click.option("--export-data", type=click.Path(), default=None,
              help="Export JSONL for ng_collect_rollouts")
def serve_cmd(benchmark, host, port, gym_compat, export_data):
    """Serve a benchmark as HTTP environment."""
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.environments.server import generate_app

    env = get_environment(benchmark)
    app = generate_app(env, gym_compat=gym_compat, data_export_dir=export_data)

    mode = "gym-compat" if gym_compat else "evaluator"
    click.echo(f"{env.name} on http://{host}:{port} ({mode})")
    click.echo(f"  size={len(env)}  endpoints: /seed_session /verify /health")

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
