from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import click


SBATCH_EVAL_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=nel-eval-{benchmark}
    #SBATCH --output={output_dir}/slurm-%A_%a.log
    #SBATCH --error={output_dir}/slurm-%A_%a.log
    #SBATCH --partition={partition}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={cpus}
    #SBATCH --mem={mem}
    #SBATCH --time={time_limit}
    {gpu_line}
    {array_line}

    set -euo pipefail

    echo "=== SLURM eval job ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Array Task: ${{SLURM_ARRAY_TASK_ID:-0}} / ${{SLURM_ARRAY_TASK_COUNT:-1}}"
    echo "Node: $(hostname)"
    echo "Start: $(date -Iseconds)"

    {conda_activate}

    export NEL_SHARD_IDX="${{SLURM_ARRAY_TASK_ID:-0}}"
    export NEL_TOTAL_SHARDS="${{SLURM_ARRAY_TASK_COUNT:-1}}"

    nel run {run_args} \\
        --output-dir "{output_dir}/shard_${{NEL_SHARD_IDX}}" \\
        --no-progress

    echo "Done: $(date -Iseconds)"
""")

SBATCH_SERVE_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=nel-serve-{benchmark}
    #SBATCH --output={output_dir}/slurm-serve-%j.log
    #SBATCH --error={output_dir}/slurm-serve-%j.log
    #SBATCH --partition={partition}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={cpus}
    #SBATCH --mem={mem}
    #SBATCH --time={time_limit}
    {gpu_line}

    set -euo pipefail

    echo "=== SLURM serve job ==="
    echo "Node: $(hostname)"

    {conda_activate}

    SERVE_HOST=$(hostname -f)
    SERVE_PORT={port}

    echo "Serving on $SERVE_HOST:$SERVE_PORT"
    echo "$SERVE_HOST:$SERVE_PORT" > "{output_dir}/endpoint.txt"

    nel serve --benchmark {benchmark} --host 0.0.0.0 --port $SERVE_PORT {serve_flags}
""")

SBATCH_MERGE_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=nel-merge-{benchmark}
    #SBATCH --output={output_dir}/slurm-merge-%j.log
    #SBATCH --dependency=afterok:{array_job_id}
    #SBATCH --partition={partition}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=8G
    #SBATCH --time=00:10:00

    set -euo pipefail
    {conda_activate}

    python -c "
from nemo_evaluator.runner.sharding import merge_results
from pathlib import Path
import glob

shard_dirs = sorted(glob.glob('{output_dir}/shard_*'))
print(f'Merging {{len(shard_dirs)}} shards...')
bundle = merge_results(shard_dirs, '{output_dir}/merged', n_repeats={repeats})
print(f'Merged: {{bundle.get(\"n_results\", 0)}} results')
for k, v in bundle.get('benchmark', {{}}).get('scores', {{}}).items():
    if isinstance(v, dict) and 'value' in v:
        print(f'  {{k}}: {{v[\"value\"]:.4f}}')
"
""")


@click.group("slurm")
def slurm_cmd():
    """Generate and submit SLURM jobs for distributed evaluation."""


@slurm_cmd.command("eval")
@click.argument("config_or_benchmark")
@click.option("--shards", "-s", type=int, default=1, help="Number of SLURM array tasks")
@click.option("--partition", "-p", default="batch")
@click.option("--cpus", type=int, default=4)
@click.option("--mem", default="32G")
@click.option("--time-limit", default="04:00:00")
@click.option("--gpus", type=int, default=0)
@click.option("--conda-env", default=None, help="Conda environment to activate")
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--model-url", default=None)
@click.option("--model-id", default=None)
@click.option("--repeats", "-n", type=int, default=1)
@click.option("--max-problems", type=int, default=None)
@click.option("--submit", is_flag=True, help="Submit immediately (otherwise just generates scripts)")
def slurm_eval(config_or_benchmark, shards, partition, cpus, mem, time_limit,
               gpus, conda_env, output_dir, model_url, model_id, repeats,
               max_problems, submit):
    """Generate sbatch script for distributed evaluation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    is_config = config_or_benchmark.endswith((".yaml", ".yml"))
    if is_config:
        run_args = config_or_benchmark
        benchmark = Path(config_or_benchmark).stem
    else:
        run_args = f"--benchmark {config_or_benchmark}"
        benchmark = config_or_benchmark

    if model_url:
        run_args += f" --model-url {model_url}"
    if model_id:
        run_args += f" --model-id {model_id}"
    if repeats > 1:
        run_args += f" --repeats {repeats}"
    if max_problems:
        run_args += f" --max-problems {max_problems}"

    conda_activate = f"source /opt/anaconda3/bin/activate {conda_env}" if conda_env else ""
    gpu_line = f"#SBATCH --gres=gpu:{gpus}" if gpus > 0 else ""
    array_line = f"#SBATCH --array=0-{shards-1}" if shards > 1 else ""

    script = SBATCH_EVAL_TEMPLATE.format(
        benchmark=benchmark, output_dir=out, partition=partition,
        cpus=cpus, mem=mem, time_limit=time_limit, gpu_line=gpu_line,
        array_line=array_line, conda_activate=conda_activate, run_args=run_args,
    )

    script_path = out / "eval.sbatch"
    script_path.write_text(script)
    click.echo(f"Generated: {script_path}")

    if shards > 1:
        merge_script = SBATCH_MERGE_TEMPLATE.format(
            benchmark=benchmark, output_dir=out, partition=partition,
            conda_activate=conda_activate, repeats=repeats,
            array_job_id="${EVAL_JOB_ID}",
        )
        merge_path = out / "merge.sbatch"
        merge_path.write_text(merge_script)
        click.echo(f"Generated: {merge_path}")

    if submit:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        click.echo(result.stdout.strip())
        if shards > 1 and result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            merge_script_final = merge_script.replace("${EVAL_JOB_ID}", job_id)
            merge_path.write_text(merge_script_final)
            merge_result = subprocess.run(["sbatch", str(merge_path)], capture_output=True, text=True)
            click.echo(f"Merge job: {merge_result.stdout.strip()}")
    else:
        click.echo(f"\nTo submit:  sbatch {script_path}")
        if shards > 1:
            click.echo(f"Then merge: sbatch {out}/merge.sbatch  (update dependency job ID first)")


@slurm_cmd.command("serve")
@click.option("--benchmark", "-b", required=True)
@click.option("--port", type=int, default=9090)
@click.option("--partition", "-p", default="batch")
@click.option("--cpus", type=int, default=4)
@click.option("--mem", default="16G")
@click.option("--time-limit", default="24:00:00")
@click.option("--gpus", type=int, default=0)
@click.option("--conda-env", default=None)
@click.option("--gym-compat", is_flag=True)
@click.option("--export-data", type=click.Path(), default=None)
@click.option("--output-dir", "-o", default="./eval_results")
@click.option("--submit", is_flag=True)
def slurm_serve(benchmark, port, partition, cpus, mem, time_limit,
                gpus, conda_env, gym_compat, export_data, output_dir, submit):
    """Generate sbatch script for serving an environment."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conda_activate = f"source /opt/anaconda3/bin/activate {conda_env}" if conda_env else ""
    gpu_line = f"#SBATCH --gres=gpu:{gpus}" if gpus > 0 else ""
    serve_flags = ""
    if gym_compat:
        serve_flags += " --gym-compat"
    if export_data:
        serve_flags += f" --export-data {export_data}"

    script = SBATCH_SERVE_TEMPLATE.format(
        benchmark=benchmark, output_dir=out, partition=partition,
        cpus=cpus, mem=mem, time_limit=time_limit, gpu_line=gpu_line,
        conda_activate=conda_activate, port=port, serve_flags=serve_flags,
    )

    script_path = out / "serve.sbatch"
    script_path.write_text(script)
    click.echo(f"Generated: {script_path}")
    click.echo(f"Endpoint will be written to: {out}/endpoint.txt")

    if submit:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        click.echo(result.stdout.strip())
    else:
        click.echo(f"\nTo submit: sbatch {script_path}")


@slurm_cmd.command("merge")
@click.option("--shard-dir", "-d", required=True, help="Parent dir containing shard_* subdirs")
@click.option("--output-dir", "-o", required=True)
@click.option("--repeats", "-n", type=int, default=1)
def slurm_merge(shard_dir, output_dir, repeats):
    """Merge sharded evaluation results."""
    import glob
    from nemo_evaluator.runner.sharding import merge_results

    shard_dirs = sorted(glob.glob(f"{shard_dir}/shard_*"))
    if not shard_dirs:
        click.echo(f"No shard directories found in {shard_dir}", err=True)
        return

    click.echo(f"Merging {len(shard_dirs)} shards...")
    bundle = merge_results(shard_dirs, output_dir, n_repeats=repeats)

    click.echo(f"  results: {bundle.get('n_results', 0)}")
    for k, v in bundle.get("benchmark", {}).get("scores", {}).items():
        if isinstance(v, dict) and "value" in v:
            click.echo(f"  {k}: {v['value']:.4f}  [{v.get('ci_lower','?'):.4f}, {v.get('ci_upper','?'):.4f}]")
    click.echo(f"  Output: {output_dir}/")
