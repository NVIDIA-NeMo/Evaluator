"""Generate self-contained sbatch scripts from EvalConfig."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from nemo_evaluator.eval.config import ClusterConfig, EvalConfig, ServiceConfig
from nemo_evaluator.eval.containers import default_base_image, resolve_deployment_image, resolve_eval_image

_HEADER = """\
#!/bin/bash
#SBATCH --job-name=nel-eval-{job_name}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err
#SBATCH --time={walltime}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
{gres_line}
{partition_line}
{account_line}

set -uo pipefail

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
mkdir -p "$OUTPUT_DIR"

echo "=== NeMo Evaluator ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -Iseconds)"
"""

_CONDA_ACTIVATE = """\
source /opt/anaconda3/bin/activate {conda_env}
"""

_CONTAINER_COMMON = """\
# Container mode (Pyxis/Enroot)
CONTAINER_MOUNTS="{mount_flags}"
CONTAINER_ENV="{env_flags}"
"""

_MODEL_SERVICE = """\
# Service: {name} ({svc_type})
echo "Starting {svc_type} server: {name}..."
{srun_prefix}{cuda_prefix}{cmd} \\
    {model_flag} {model} \\
    --port {port} \\
    {tp_flag}\\
    {extra_args}&
{name_upper}_PID=$!
{name_upper}_URL="http://localhost:{port}/v1"
{name_upper}_MODEL="{model}"
"""

_GYM_SERVICE = """\
# Service: {name} (gym)
echo "Starting benchmark server: {name}..."
nel serve -b {benchmark} --host 0.0.0.0 -p {port} &
{name_upper}_PID=$!
"""

_HEALTH_WAIT = """\
# Wait for {name}
echo "Waiting for {name} at {url}..."
{name_upper}_READY=0
for _i in $(seq 1 {max_attempts}); do
    if curl -sf "{url}{health_path}" > /dev/null 2>&1; then
        echo "  {name} ready."
        {name_upper}_READY=1
        break
    fi
    if [ -n "${{{name_upper}_PID:-}}" ] && ! kill -0 ${name_upper}_PID 2>/dev/null; then
        echo "  {name} died during startup."
        exit 1
    fi
    sleep 5
done
if [ ${name_upper}_READY -eq 0 ]; then
    echo "ERROR: {name} did not become healthy after {max_attempts} attempts."
    exit 1
fi
"""

_TASK = """\
# Benchmark {idx}/{total}: {bench_name}
echo ""
echo "============================================================"
echo "  Benchmark {idx}/{total}: {bench_name} (repeats={repeats})"
echo "============================================================"
{run_prefix}nel eval run \\
    --bench "{bench_name}" \\
    --model-url "${model_url_var}" \\
    --model-id "${model_id_var}" \\
    --repeats {repeats} \\
    {extra_flags}\\
    -o "$OUTPUT_DIR/{safe_name}" || {{ echo "  FAILED: {bench_name}"; NEL_EXIT_CODE=1; }}
"""

_TASK_SEPARATED = """\
# Benchmark {idx}/{total}: {bench_name} (separated mode)
echo ""
echo "============================================================"
echo "  Benchmark {idx}/{total}: {bench_name} (repeats={repeats})"
echo "  Env container: {env_image} -> port {env_port}"
echo "============================================================"
srun --overlap --nodes 1 --ntasks 1 \\
    --container-image {env_image} $CONTAINER_MOUNTS $CONTAINER_ENV \\
    nel serve -b "{bench_name}" --host 0.0.0.0 -p {env_port} &
BENCH_{safe_name}_PID=$!
"""

_REPORT = """\
# Generate reports
echo ""
echo "=== Generating reports ==="
{report_commands}
"""

_CLEANUP = """\
# Cleanup
echo ""
echo "Shutting down services..."
{kill_commands}
echo "=== Evaluation complete ==="
echo "End: $(date -Iseconds)"
echo "Results: $OUTPUT_DIR"
exit $NEL_EXIT_CODE
"""

_SANDBOX_NODES = """\
# Sandbox node allocation
SANDBOX_NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | tail -n +{sandbox_start_node} | head -n {sandbox_node_count})
export NEL_SANDBOX_NODES=$(echo $SANDBOX_NODES | tr ' ' ',')
echo "Sandbox nodes (${{NEL_SANDBOX_NODES}}): {sandbox_node_count} nodes x {slots_per_node} slots = {total_slots} max concurrent"
"""

_SANDBOX_PRE_PULL = """\
# Pre-pull sandbox images on sandbox nodes
echo "Pre-pulling sandbox images..."
for node in $SANDBOX_NODES; do
    for img in {images}; do
        srun --overlap --nodelist=$node --ntasks=1 enroot import "docker://$img" &
    done
done
wait
echo "Pre-pull complete."
"""

_AUTO_RESUME = """\
# Auto-resume on walltime expiry
ATTEMPT_FILE="$OUTPUT_DIR/.nel_attempt"
if [ -f "$ATTEMPT_FILE" ]; then
    ATTEMPT=$(cat "$ATTEMPT_FILE")
else
    ATTEMPT=0
fi
ATTEMPT=$((ATTEMPT + 1))
echo $ATTEMPT > "$ATTEMPT_FILE"

if [ $ATTEMPT -ge {max_attempts} ]; then
    echo "Max resume attempts ({max_attempts}) reached."
else
    # Check if all benchmarks completed successfully via checkpoint
    if python3 -c "
import json, sys, os
cp_path = '$OUTPUT_DIR/checkpoint.json'
if not os.path.exists(cp_path):
    sys.exit(0)  # no checkpoint = nothing completed, resubmit
cp = json.load(open(cp_path))
failed = len(cp.get('failed_benchmarks', {{}}))
completed = len(cp.get('completed_benchmarks', {{}}))
sys.exit(0 if failed > 0 or completed == 0 else 1)
" 2>/dev/null; then
        echo "Evaluation incomplete, resubmitting (attempt $ATTEMPT/{max_attempts})..."
        NEXT_JOB=$(sbatch --dependency=afternotok:$SLURM_JOB_ID "{script_path}")
        echo "Resubmitted: $NEXT_JOB"
    else
        echo "All benchmarks completed, no resubmit needed."
    fi
fi
"""


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


_MODEL_CMD = {
    "vllm": ("python -m vllm.entrypoints.openai.api_server", "--model", "--tensor-parallel-size"),
    "sglang": ("python -m sglang.launch_server", "--model-path", "--tp-size"),
}


def _service_block(name: str, svc: ServiceConfig, use_containers: bool = False) -> str:
    upper = _safe(name).upper()

    if svc.type in _MODEL_CMD:
        cmd, model_flag, tp_flag_name = _MODEL_CMD[svc.type]
        deploy_image = resolve_deployment_image(svc.type)
        srun_prefix = ""
        if use_containers and deploy_image:
            srun_prefix = (
                f"srun --overlap --nodes {svc.num_nodes} --ntasks 1 "
                f"--container-image {deploy_image} $CONTAINER_MOUNTS $CONTAINER_ENV "
            )
        tp_flag = f"{tp_flag_name} {svc.tensor_parallel_size} " if svc.tensor_parallel_size else ""
        extra = " ".join(svc.extra_args)
        cuda = ""
        if svc.gpus and isinstance(svc.gpus, list):
            cuda = f'CUDA_VISIBLE_DEVICES={",".join(str(g) for g in svc.gpus)} '
        return _MODEL_SERVICE.format(
            name=name, name_upper=upper, svc_type=svc.type,
            cmd=cmd, model_flag=model_flag, model=svc.model or "",
            port=svc.port, tp_flag=tp_flag, extra_args=extra,
            cuda_prefix=cuda, srun_prefix=srun_prefix,
        )

    if svc.type == "gym":
        return _GYM_SERVICE.format(
            name=name, name_upper=upper,
            benchmark=svc.benchmark or "", port=svc.port,
        )

    if svc.type == "api":
        return (
            f'# Service: {name} (external API)\n'
            f'{upper}_URL="{svc.url or ""}"\n'
            f'{upper}_MODEL="{svc.model or ""}"\n'
            f'{upper}_PID=""\n'
        )

    return (
        f'# Service: {name} ({svc.type})\n'
        f'{upper}_URL="http://localhost:{svc.port}/v1"\n'
        f'{upper}_MODEL="{svc.model or ""}"\n'
        f'{upper}_PID=""\n'
    )


def _health_block(name: str, svc: ServiceConfig) -> str:
    if svc.type == "api":
        return ""
    upper = _safe(name).upper()
    url = f"http://localhost:{svc.port}"
    health = svc.health_path if svc.is_model_server else "/health"
    max_attempts = int(svc.startup_timeout / 5) or 120
    return _HEALTH_WAIT.format(
        name=name, name_upper=upper, url=url,
        health_path=health, max_attempts=max_attempts,
    )


def _resolve_eval_image_for_bench(
    bench_name: str,
    base_override: str | None,
    use_containers: bool,
) -> str:
    """Return the eval container image for a benchmark, or empty string if not containerized."""
    if not use_containers:
        return ""
    return resolve_eval_image(bench_name, base_override=base_override) or default_base_image(base_override)


def _sandbox_node_count(config: EvalConfig) -> int:
    """Total extra nodes needed for sandbox across all benchmarks."""
    return max(
        (b.sandbox.sandbox_nodes for b in config.benchmarks
         if b.sandbox and b.sandbox.backend == "slurm" and b.sandbox.sandbox_nodes > 0),
        default=0,
    )


def generate_sbatch(config: EvalConfig) -> str:
    """Generate a complete sbatch script from an EvalConfig."""
    cluster = config.cluster
    output_dir = config.output.dir

    extra_sandbox_nodes = _sandbox_node_count(config)
    total_nodes = cluster.nodes + extra_sandbox_nodes

    job_name = _safe(config.benchmarks[0].name) if len(config.benchmarks) == 1 else "multi"

    gres_line = f"#SBATCH --gres={cluster.gres}" if cluster.gres else ""
    partition_line = f"#SBATCH --partition={cluster.partition}" if cluster.partition else ""
    account_line = f"#SBATCH --account={cluster.account}" if cluster.account else ""

    parts: list[str] = []

    # Header
    parts.append(_HEADER.format(
        job_name=job_name, output_dir=output_dir,
        walltime=cluster.walltime, nodes=total_nodes,
        ntasks_per_node=cluster.ntasks_per_node,
        gres_line=gres_line, partition_line=partition_line,
        account_line=account_line,
    ))

    use_containers = cluster.container_image is not None or cluster.type == "slurm"
    if use_containers:
        mount_parts = []
        for m in cluster.container_mounts:
            mount_parts.append(f"--container-mounts={m}")
        if cluster.mount_home:
            mount_parts.append("--container-mounts=$HOME:$HOME")
        else:
            mount_parts.append("--no-container-mount-home")
        mount_flags = " ".join(mount_parts)
        env_flags = " ".join(
            f"--container-env={k}" for k in cluster.container_env
        )
        parts.append(_CONTAINER_COMMON.format(
            mount_flags=mount_flags,
            env_flags=env_flags,
        ))
        for k, v in cluster.container_env.items():
            parts.append(f'export {k}="{v}"')
        if cluster.container_env:
            parts.append("")

    if cluster.conda_env:
        parts.append(_CONDA_ACTIVATE.format(conda_env=cluster.conda_env))

    # Services
    services = config.resolved_services()
    for name, svc in services.items():
        parts.append(_service_block(name, svc, use_containers=use_containers))

    parts.append("")

    # Health waits
    for name, svc in services.items():
        block = _health_block(name, svc)
        if block:
            parts.append(block)

    # Sandbox node allocation and pre-pull
    if extra_sandbox_nodes > 0:
        sandbox_start_node = cluster.nodes + 1
        sandbox_bench = next(
            (b for b in config.benchmarks
             if b.sandbox and b.sandbox.backend == "slurm" and b.sandbox.sandbox_nodes > 0),
            None,
        )
        slots = sandbox_bench.sandbox.slots_per_node if sandbox_bench and sandbox_bench.sandbox else 4
        total_slots = extra_sandbox_nodes * slots
        parts.append(_SANDBOX_NODES.format(
            sandbox_start_node=sandbox_start_node,
            sandbox_node_count=extra_sandbox_nodes,
            slots_per_node=slots,
            total_slots=total_slots,
        ))

        if sandbox_bench and sandbox_bench.sandbox and sandbox_bench.sandbox.image:
            parts.append(_SANDBOX_PRE_PULL.format(
                images=shlex.quote(sandbox_bench.sandbox.image),
            ))

    # Tasks
    base_override = cluster.container_image

    separated = cluster.env_mode == "separated" and use_containers
    base_env_port = 9100

    total = len(config.benchmarks)
    env_kill_cmds: list[str] = []

    for i, bench in enumerate(config.benchmarks, 1):
        svc_name = bench.model
        upper = _safe(svc_name).upper()
        model_url_var = f"{upper}_URL"
        model_id_var = f"{upper}_MODEL"

        extra_flags = ""
        if bench.system_prompt:
            extra_flags += f"--system-prompt {shlex.quote(bench.system_prompt)} "
        if bench.temperature is not None:
            extra_flags += f"--temperature {bench.temperature} "
        if bench.max_tokens is not None:
            extra_flags += f"--max-tokens {bench.max_tokens} "
        if bench.max_problems is not None:
            extra_flags += f"--max-problems {bench.max_problems} "
        if cluster.auto_resume:
            extra_flags += "--resume "

        safe_name = _safe(bench.name)

        eval_image = _resolve_eval_image_for_bench(
            bench.name, base_override, use_containers,
        )

        if separated:
            env_port = base_env_port + i - 1

            parts.append(_TASK_SEPARATED.format(
                idx=i, total=total, bench_name=bench.name,
                repeats=bench.repeats, env_image=eval_image,
                env_port=env_port, safe_name=safe_name,
            ))

            parts.append(_HEALTH_WAIT.format(
                name=f"env-{safe_name}", name_upper=f"BENCH_{safe_name}",
                url=f"http://localhost:{env_port}",
                health_path="/health", max_attempts=60,
            ))

            base_image = default_base_image(base_override)
            orch_prefix = (
                f"srun --overlap --nodes 1 --ntasks 1 "
                f"--container-image {base_image} $CONTAINER_MOUNTS $CONTAINER_ENV \\\n    "
            )
            parts.append(_TASK.format(
                idx=i, total=total, bench_name=f"gym://localhost:{env_port}",
                model_url_var=model_url_var, model_id_var=model_id_var,
                repeats=bench.repeats, extra_flags=extra_flags,
                safe_name=safe_name, run_prefix=orch_prefix,
            ))
            env_kill_cmds.append(f'kill $BENCH_{safe_name}_PID 2>/dev/null || true')
        else:
            run_prefix = ""
            if use_containers and eval_image:
                run_prefix = (
                    f"srun --overlap --nodes 1 --ntasks 1 "
                    f"--container-image {eval_image} $CONTAINER_MOUNTS $CONTAINER_ENV \\\n    "
                )
            parts.append(_TASK.format(
                idx=i, total=total, bench_name=bench.name,
                model_url_var=model_url_var, model_id_var=model_id_var,
                repeats=bench.repeats, extra_flags=extra_flags,
                safe_name=safe_name, run_prefix=run_prefix,
            ))

    # Reports
    report_cmds = []
    ext_map = {"markdown": "md", "html": "html", "csv": "csv", "json": "json", "latex": "tex"}
    for fmt in config.output.report:
        ext = ext_map.get(fmt, fmt)
        report_cmds.append(f'nel eval report "$OUTPUT_DIR" -f {fmt} -o "$OUTPUT_DIR/report.{ext}"')
    if report_cmds:
        parts.append(_REPORT.format(report_commands="\n".join(report_cmds)))

    # Auto-resume
    if cluster.auto_resume:
        parts.append(_AUTO_RESUME.format(
            max_attempts=cluster.max_resume_attempts,
            script_path=f"$OUTPUT_DIR/nel_eval.sbatch",
        ))

    # Cleanup
    kill_cmds = list(env_kill_cmds)
    for name, svc in services.items():
        if svc.type != "api":
            upper = _safe(name).upper()
            kill_cmds.append(f'kill ${upper}_PID 2>/dev/null || true')
    for name, svc in services.items():
        if svc.type != "api":
            upper = _safe(name).upper()
            kill_cmds.append(f'wait ${upper}_PID 2>/dev/null || true')

    parts.append(_CLEANUP.format(kill_commands="\n".join(kill_cmds) if kill_cmds else "echo 'No managed services.'"))

    return "\n".join(parts)


def write_sbatch(config: EvalConfig, output_dir: str | Path | None = None) -> Path:
    """Generate and write the sbatch script to disk."""
    out = Path(output_dir or config.output.dir)
    out.mkdir(parents=True, exist_ok=True)

    script = generate_sbatch(config)
    path = out / "nel_eval.sbatch"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path
