"""Generate self-contained sbatch scripts from EvalConfig."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from nemo_evaluator.eval.config import EvalConfig, ServiceConfig

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

set -euo pipefail

OUTPUT_DIR="{output_dir}"
mkdir -p "$OUTPUT_DIR"

echo "=== NeMo Evaluator ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -Iseconds)"
"""

_CONDA_ACTIVATE = """\
source /opt/anaconda3/bin/activate {conda_env}
"""

_VLLM_SERVICE = """\
# Service: {name} (vllm)
echo "Starting vLLM server: {name}..."
{cuda_prefix}python -m vllm.entrypoints.openai.api_server \\
    --model {model} \\
    --port {port} \\
    {tp_flag}\\
    {extra_args}&
{name_upper}_PID=$!
{name_upper}_URL="http://localhost:{port}/v1"
{name_upper}_MODEL="{model}"
"""

_SGLANG_SERVICE = """\
# Service: {name} (sglang)
echo "Starting SGLang server: {name}..."
{cuda_prefix}python -m sglang.launch_server \\
    --model-path {model} \\
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
for _i in $(seq 1 {max_attempts}); do
    if curl -sf "{url}{health_path}" > /dev/null 2>&1; then
        echo "  {name} ready."
        break
    fi
    if [ -n "${{{name_upper}_PID:-}}" ] && ! kill -0 ${name_upper}_PID 2>/dev/null; then
        echo "  {name} died during startup."
        exit 1
    fi
    sleep 5
done
"""

_TASK = """\
# Benchmark {idx}/{total}: {bench_name}
echo ""
echo "============================================================"
echo "  Benchmark {idx}/{total}: {bench_name} (repeats={repeats})"
echo "============================================================"
nel eval run \\
    --bench "{bench_name}" \\
    --model-url "${model_url_var}" \\
    --model-id "${model_id_var}" \\
    --repeats {repeats} \\
    {extra_flags}\\
    -o "$OUTPUT_DIR/{safe_name}" \\
    --no-progress
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
"""


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


def _service_block(name: str, svc: ServiceConfig) -> str:
    upper = _safe(name).upper()

    if svc.type == "vllm":
        tp_flag = f"--tensor-parallel-size {svc.tensor_parallel_size} " if svc.tensor_parallel_size else ""
        extra = " ".join(svc.extra_args)
        cuda = ""
        if svc.gpus and isinstance(svc.gpus, list):
            cuda = f'CUDA_VISIBLE_DEVICES={",".join(str(g) for g in svc.gpus)} '
        return _VLLM_SERVICE.format(
            name=name, name_upper=upper, model=svc.model or "",
            port=svc.port, tp_flag=tp_flag, extra_args=extra,
            cuda_prefix=cuda,
        )

    if svc.type == "sglang":
        tp_flag = f"--tp-size {svc.tensor_parallel_size} " if svc.tensor_parallel_size else ""
        extra = " ".join(svc.extra_args)
        cuda = ""
        if svc.gpus and isinstance(svc.gpus, list):
            cuda = f'CUDA_VISIBLE_DEVICES={",".join(str(g) for g in svc.gpus)} '
        return _SGLANG_SERVICE.format(
            name=name, name_upper=upper, model=svc.model or "",
            port=svc.port, tp_flag=tp_flag, extra_args=extra,
            cuda_prefix=cuda,
        )

    if svc.type == "gym":
        return _GYM_SERVICE.format(
            name=name, name_upper=upper,
            benchmark=svc.benchmark or "", port=svc.port,
        )

    if svc.type == "api":
        upper = _safe(name).upper()
        return (
            f'# Service: {name} (external API)\n'
            f'{upper}_URL="{svc.url or ""}"\n'
            f'{upper}_MODEL="{svc.model or ""}"\n'
            f'{upper}_PID=""\n'
        )

    # Docker / NIM
    return (
        f'# Service: {name} ({svc.type}) — TODO: add container support\n'
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


def generate_sbatch(config: EvalConfig) -> str:
    """Generate a complete sbatch script from an EvalConfig."""
    cluster = config.cluster
    output_dir = config.output.dir

    job_name = _safe(config.benchmarks[0].name) if len(config.benchmarks) == 1 else "multi"

    gres_line = f"#SBATCH --gres={cluster.gres}" if cluster.gres else ""
    partition_line = f"#SBATCH --partition={cluster.partition}" if cluster.partition else ""
    account_line = f"#SBATCH --account={cluster.account}" if cluster.account else ""

    parts: list[str] = []

    # Header
    parts.append(_HEADER.format(
        job_name=job_name, output_dir=output_dir,
        walltime=cluster.walltime, nodes=cluster.nodes,
        ntasks_per_node=cluster.ntasks_per_node,
        gres_line=gres_line, partition_line=partition_line,
        account_line=account_line,
    ))

    # Conda
    if cluster.conda_env:
        parts.append(_CONDA_ACTIVATE.format(conda_env=cluster.conda_env))

    # Services
    services = _resolve_services(config)
    for name, svc in services.items():
        parts.append(_service_block(name, svc))

    parts.append("")

    # Health waits
    for name, svc in services.items():
        block = _health_block(name, svc)
        if block:
            parts.append(block)

    # Tasks
    total = len(config.benchmarks)
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

        safe_name = _safe(bench.name)
        parts.append(_TASK.format(
            idx=i, total=total, bench_name=bench.name,
            model_url_var=model_url_var, model_id_var=model_id_var,
            repeats=bench.repeats, extra_flags=extra_flags,
            safe_name=safe_name,
        ))

    # Reports
    report_cmds = []
    ext_map = {"markdown": "md", "html": "html", "csv": "csv", "json": "json", "latex": "tex"}
    for fmt in config.output.report:
        ext = ext_map.get(fmt, fmt)
        report_cmds.append(f'nel eval report "$OUTPUT_DIR" -f {fmt} -o "$OUTPUT_DIR/report.{ext}"')
    if report_cmds:
        parts.append(_REPORT.format(report_commands="\n".join(report_cmds)))

    # Cleanup
    kill_cmds = []
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


def _resolve_services(config: EvalConfig) -> dict[str, ServiceConfig]:
    """Build the services dict, handling simple mode auto-service."""
    if config.services:
        return dict(config.services)

    if config.is_simple and config.model:
        m = config.model
        if m.deploy:
            svc = ServiceConfig(
                type=m.deploy,
                model=m.name or m.id,
                port=m.port,
                tensor_parallel_size=m.tensor_parallel_size,
                extra_env=m.extra_env,
                extra_args=list(m.extra_args),
            )
            return {"default": svc}
        else:
            svc = ServiceConfig(
                type="api",
                url=m.url,
                model=m.id,
                api_key=m.api_key,
            )
            return {"default": svc}

    return {}


def write_sbatch(config: EvalConfig, output_dir: str | Path | None = None) -> Path:
    """Generate and write the sbatch script to disk."""
    out = Path(output_dir or config.output.dir)
    out.mkdir(parents=True, exist_ok=True)

    script = generate_sbatch(config)
    path = out / "nel_eval.sbatch"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path
