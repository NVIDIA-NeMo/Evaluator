"""Generate self-contained sbatch scripts from EvalConfig."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from nemo_evaluator.eval.config import (
    ApptainerSandbox,
    EvalConfig,
    ExternalApiService,
    GymResourceService,
    NatAgentService,
    SlurmCluster,
    SlurmSandbox,
)
from nemo_evaluator.eval.containers import (
    default_base_image,
    resolve_deployment_image,
    resolve_eval_image,
)

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
{array_line}

set -uo pipefail
# Prevent nel from re-submitting to SLURM/Docker from inside the job.
export NEL_INNER_EXECUTION=1

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
mkdir -p "$OUTPUT_DIR"

echo "=== NeMo Evaluator ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -Iseconds)"
"""

_SHARD_SETUP = """\
# Sharded execution (array task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT)
export NEL_SHARD_IDX=$SLURM_ARRAY_TASK_ID
export NEL_TOTAL_SHARDS=$SLURM_ARRAY_TASK_COUNT
OUTPUT_DIR="$OUTPUT_DIR/shard_$SLURM_ARRAY_TASK_ID"
mkdir -p "$OUTPUT_DIR"
echo "Shard: $NEL_SHARD_IDX / $NEL_TOTAL_SHARDS  (output: $OUTPUT_DIR)"
"""

_SHARD_MERGE_HINT = """\
# All array tasks write to separate shard_N/ directories.
# After all tasks complete, merge results:
#   nel eval merge "{parent_output_dir}"
echo ""
echo "Shard $SLURM_ARRAY_TASK_ID complete. Merge after all tasks finish:"
echo "  nel eval merge {parent_output_dir}"
"""

_HEADER_HETJOB_PREAMBLE = """\
#!/bin/bash
"""

_HEADER_HETJOB_POOL = """\
# ── Het Group {het_idx}: {pool_name} ──
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
{gres_line}
{partition_line}
"""

_HEADER_HETJOB_SEPARATOR = """\
#SBATCH hetjob
"""

_HEADER_HETJOB_FOOTER = """\
#SBATCH --job-name=nel-eval-{job_name}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err
#SBATCH --time={walltime}
{account_line}

set -uo pipefail
export NEL_INNER_EXECUTION=1

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
mkdir -p "$OUTPUT_DIR"

echo "=== NeMo Evaluator (het-job) ==="
echo "Job ID: $SLURM_JOB_ID"
{het_echo_lines}
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

_GYM_CMD_SERVICE = """\
# Service: {name} (gym / custom server)
echo "Starting server: {name}..."
{server_cmd} &
{name_upper}_PID=$!
"""

_NAT_SERVICE = """\
# Service: {name} (nat agent)
echo "Starting NAT agent server: {name}..."
{srun_prefix}nat serve --config_file {config_file} --port {port} --host 0.0.0.0 &
{name_upper}_PID=$!
{name_upper}_URL="http://localhost:{port}"
{name_upper}_MODEL="nat-agent"
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

_HEALTH_WAIT_MULTI = """\
# Wait for {name} (try multiple health endpoints)
echo "Waiting for {name} at {url}..."
{name_upper}_READY=0
for _i in $(seq 1 {max_attempts}); do
    if curl -sf "{url}/health" > /dev/null 2>&1 || curl -sf "{url}/openapi.json" > /dev/null 2>&1; then
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

_SANDBOX_NODES_HETJOB = """\
# Sandbox node allocation (het-group {het_group})
SANDBOX_NODES=$(scontrol show hostname $SLURM_JOB_NODELIST_HET_GROUP_{het_group})
export NEL_SANDBOX_NODES=$(echo $SANDBOX_NODES | tr ' ' ',')
export NEL_SANDBOX_HET_GROUP={het_group}
echo "Sandbox nodes (${{NEL_SANDBOX_NODES}}): {sandbox_node_count} nodes x {slots_per_node} slots = {total_slots} max concurrent"
"""

_SANDBOX_NODES_INLINE = """\
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

_APPTAINER_PRE_PROVISION = """\
# Pre-provision Apptainer SIF on shared filesystem
SIF_CACHE="{sif_cache_dir}"
mkdir -p "$SIF_CACHE"
echo "Building SIF images in $SIF_CACHE ..."
for img in {images}; do
    SAFE=$(echo "$img" | sed 's|/|_|g; s|:|__|g')
    SIF_PATH="$SIF_CACHE/${{SAFE}}.sif"
    if [ ! -f "$SIF_PATH" ]; then
        echo "  Building $SIF_PATH from docker://$img ..."
        apptainer build "$SIF_PATH" "docker://$img"
    else
        echo "  $SIF_PATH already exists."
    fi
done
# Verify SIF accessible from all sandbox nodes
echo "Verifying SIF accessibility on sandbox nodes..."
for node in $SANDBOX_NODES; do
    srun --overlap --nodelist=$node --ntasks=1{het_group_flag} test -d "$SIF_CACHE" || {{
        echo "ERROR: $SIF_CACHE not accessible on $node"; exit 1;
    }}
done
echo "SIF pre-provision complete."
"""

_APPTAINER_CLEANUP = """\
# Cleanup Apptainer instances on sandbox nodes
echo "Cleaning up Apptainer instances on sandbox nodes..."
for node in $SANDBOX_NODES; do
    srun --overlap --nodelist=$node --ntasks=1{het_group_flag} \\
        bash -c 'for inst in $(apptainer instance list -j 2>/dev/null | python3 -c "import sys,json; [print(i[\\"instance\\"]) for i in json.load(sys.stdin).get(\\"instances\\",[]) if i[\\"instance\\"].startswith(\\"nel-\\")]" 2>/dev/null); do apptainer instance stop "$inst" 2>/dev/null; done' &
done
wait
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
    if python3 -c "
import json, sys, os
cp_path = '$OUTPUT_DIR/checkpoint.json'
if not os.path.exists(cp_path):
    sys.exit(0)
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


def _collect_used_pools(config: EvalConfig) -> list[str]:
    """Collect node pool names actually referenced by services and sandboxes,
    preserving declaration order from node_pools dict."""
    if not isinstance(config.cluster, SlurmCluster):
        return []
    all_pools = list(config.cluster.node_pools.keys())
    used: set[str] = set()
    for svc in config.services.values():
        pool = getattr(svc, "node_pool", None)
        if pool:
            used.add(pool)
    for sb in config.sandboxes.values():
        pool = getattr(sb, "node_pool", None)
        if pool:
            used.add(pool)
    for bench in config.benchmarks:
        pool = getattr(bench.sandbox, "node_pool", None)
        if pool:
            used.add(pool)
    if not used:
        return all_pools[:1]
    return [p for p in all_pools if p in used]


def _resolve_service_image(svc) -> str:
    if getattr(svc, "image", None):
        return svc.image
    return resolve_deployment_image(svc.type)


def _service_block(name: str, svc, *, use_containers: bool = False,
                   pool_to_het: dict[str, int] | None = None) -> str:
    upper = _safe(name).upper()
    pool = getattr(svc, "node_pool", None)
    het_flag = ""
    if pool and pool_to_het and pool in pool_to_het:
        het_flag = f" --het-group={pool_to_het[pool]}"

    if svc.type in _MODEL_CMD:
        cmd, model_flag, tp_flag_name = _MODEL_CMD[svc.type]
        deploy_image = _resolve_service_image(svc)
        srun_prefix = ""
        if use_containers and deploy_image:
            srun_prefix = (
                f"srun --overlap --nodes {svc.num_nodes} --ntasks 1{het_flag} "
                f"--container-image {deploy_image} $CONTAINER_MOUNTS $CONTAINER_ENV "
            )
        tp_flag = (
            f"{tp_flag_name} {svc.tensor_parallel_size} "
            if svc.tensor_parallel_size else ""
        )
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

    if isinstance(svc, NatAgentService):
        deploy_image = _resolve_service_image(svc)
        srun_prefix = ""
        if use_containers and deploy_image:
            srun_prefix = (
                f"srun --overlap --nodes 1 --ntasks 1{het_flag} "
                f"--container-image {deploy_image} $CONTAINER_MOUNTS $CONTAINER_ENV "
            )
        config_file = svc.nat_config_file or "config.yml"
        return _NAT_SERVICE.format(
            name=name, name_upper=upper, port=svc.port,
            config_file=config_file, srun_prefix=srun_prefix,
        )

    if isinstance(svc, GymResourceService):
        if svc.server_cmd:
            return _GYM_CMD_SERVICE.format(
                name=name, name_upper=upper,
                server_cmd=svc.server_cmd,
            )
        return _GYM_SERVICE.format(
            name=name, name_upper=upper,
            benchmark=svc.benchmark or "", port=svc.port,
        )

    if isinstance(svc, ExternalApiService):
        return (
            f'# Service: {name} (external API)\n'
            f'{upper}_URL="{svc.url or ""}"\n'
            f'{upper}_MODEL="{svc.model or ""}"\n'
            f'{upper}_PID=""\n'
        )

    return (
        f'# Service: {name} ({svc.type})\n'
        f'{upper}_URL="http://localhost:{getattr(svc, "port", 8000)}/v1"\n'
        f'{upper}_MODEL="{getattr(svc, "model", "") or ""}"\n'
        f'{upper}_PID=""\n'
    )


def _health_block(name: str, svc) -> str:
    if isinstance(svc, ExternalApiService):
        return ""
    upper = _safe(name).upper()
    port = getattr(svc, "port", 8000)
    url = f"http://localhost:{port}"
    timeout = getattr(svc, "startup_timeout", 600.0)
    max_attempts = int(timeout / 5) or 120

    if isinstance(svc, GymResourceService) and svc.server_cmd:
        return _HEALTH_WAIT_MULTI.format(
            name=name, name_upper=upper, url=url,
            max_attempts=max_attempts,
        )

    health = getattr(svc, "health_path", "/health") or "/health"
    return _HEALTH_WAIT.format(
        name=name, name_upper=upper, url=url,
        health_path=health, max_attempts=max_attempts,
    )


def _get_solver_service(bench) -> str | None:
    return getattr(bench.solver, "service", None)


def _find_sandbox_bench(config: EvalConfig):
    """Return the first benchmark with a SLURM/Apptainer sandbox that has a node_pool."""
    for b in config.benchmarks:
        if isinstance(b.sandbox, (SlurmSandbox, ApptainerSandbox)):
            if getattr(b.sandbox, "node_pool", None):
                return b
    for b in config.benchmarks:
        if isinstance(b.sandbox, (SlurmSandbox, ApptainerSandbox)):
            return b
    return None


def generate_sbatch(config: EvalConfig) -> str:
    cluster = config.cluster
    if not isinstance(cluster, SlurmCluster):
        raise ValueError(f"generate_sbatch requires a SlurmCluster, got {type(cluster).__name__}")

    output_dir = config.output.dir
    job_name = (
        _safe(config.benchmarks[0].name) if len(config.benchmarks) == 1
        else "multi"
    )
    account_line = f"#SBATCH --account={cluster.account}" if cluster.account else ""

    used_pools = _collect_used_pools(config)
    use_het = len(used_pools) > 1
    pool_to_het = {name: i for i, name in enumerate(used_pools)} if use_het else {}

    parts: list[str] = []

    if use_het:
        parts.append(_HEADER_HETJOB_PREAMBLE)
        for i, pool_name in enumerate(used_pools):
            pool = cluster.node_pools[pool_name]
            gres_line = f"#SBATCH --gres={pool.gres}" if pool.gres else ""
            partition_line = f"#SBATCH --partition={pool.partition}" if pool.partition else ""
            parts.append(_HEADER_HETJOB_POOL.format(
                het_idx=i, pool_name=pool_name,
                nodes=pool.nodes, ntasks_per_node=pool.ntasks_per_node,
                gres_line=gres_line, partition_line=partition_line,
            ))
            if i < len(used_pools) - 1:
                parts.append(_HEADER_HETJOB_SEPARATOR)

        het_echo_lines = "\n".join(
            f'echo "Het-group {i} ({name}): $SLURM_JOB_NODELIST_HET_GROUP_{i}"'
            for i, name in enumerate(used_pools)
        )
        parts.append(_HEADER_HETJOB_FOOTER.format(
            job_name=job_name, output_dir=output_dir,
            walltime=cluster.walltime, account_line=account_line,
            het_echo_lines=het_echo_lines,
        ))
    else:
        pool = cluster.node_pools[used_pools[0]]
        gres_line = f"#SBATCH --gres={pool.gres}" if pool.gres else ""
        partition_line = f"#SBATCH --partition={pool.partition}" if pool.partition else ""
        array_line = (
            f"#SBATCH --array=0-{cluster.shards - 1}"
            if cluster.shards else ""
        )
        parts.append(_HEADER.format(
            job_name=job_name, output_dir=output_dir,
            walltime=cluster.walltime, nodes=pool.nodes,
            ntasks_per_node=pool.ntasks_per_node,
            gres_line=gres_line, partition_line=partition_line,
            account_line=account_line, array_line=array_line,
        ))

    is_sharded = cluster.shards is not None

    if is_sharded:
        parts.append(_SHARD_SETUP)

    use_containers = cluster.container_image is not None
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

    for name, svc in config.services.items():
        parts.append(_service_block(
            name, svc,
            use_containers=use_containers,
            pool_to_het=pool_to_het if use_het else None,
        ))

    parts.append("")

    for name, svc in config.services.items():
        block = _health_block(name, svc)
        if block:
            parts.append(block)

    sb_bench = _find_sandbox_bench(config)
    sandbox_pool_name = getattr(sb_bench.sandbox, "node_pool", None) if sb_bench else None

    if sb_bench and isinstance(sb_bench.sandbox, (SlurmSandbox, ApptainerSandbox)):
        sb = sb_bench.sandbox
        slots = sb.slots_per_node

        if sandbox_pool_name and sandbox_pool_name in pool_to_het:
            het_idx = pool_to_het[sandbox_pool_name]
            pool = cluster.node_pools[sandbox_pool_name]
            total_slots = pool.nodes * slots
            parts.append(_SANDBOX_NODES_HETJOB.format(
                het_group=het_idx,
                sandbox_node_count=pool.nodes,
                slots_per_node=slots,
                total_slots=total_slots,
            ))
        elif sandbox_pool_name and sandbox_pool_name in cluster.node_pools:
            pool = cluster.node_pools[sandbox_pool_name]
            total_slots = pool.nodes * slots
            sandbox_start = sum(
                cluster.node_pools[p].nodes for p in used_pools
                if p != sandbox_pool_name
            ) + 1
            parts.append(_SANDBOX_NODES_INLINE.format(
                sandbox_start_node=sandbox_start,
                sandbox_node_count=pool.nodes,
                slots_per_node=slots,
                total_slots=total_slots,
            ))

        het_group_flag = (
            f" --het-group={pool_to_het[sandbox_pool_name]}"
            if sandbox_pool_name and sandbox_pool_name in pool_to_het
            else ""
        )
        sb_image = sb.image

        if isinstance(sb, ApptainerSandbox) and sb_image:
            sif_cache = sb.sif_cache_dir or "/tmp/nel_sif_cache"
            parts.append(_APPTAINER_PRE_PROVISION.format(
                sif_cache_dir=sif_cache,
                images=shlex.quote(sb_image),
                het_group_flag=het_group_flag,
            ))
        elif isinstance(sb, SlurmSandbox) and sb_image:
            parts.append(_SANDBOX_PRE_PULL.format(
                images=shlex.quote(sb_image),
            ))

    base_override = cluster.container_image
    total = len(config.benchmarks)

    for i, bench in enumerate(config.benchmarks, 1):
        svc_name = _get_solver_service(bench) or ""
        upper = _safe(svc_name).upper() if svc_name else "MODEL"
        model_url_var = f"{upper}_URL"
        model_id_var = f"{upper}_MODEL"

        extra_flags = ""
        gen = getattr(bench.solver, "generation", None)
        system_prompt = getattr(bench.solver, "system_prompt", None)
        if system_prompt:
            extra_flags += f"--system-prompt {shlex.quote(system_prompt)} "
        if gen and gen.temperature is not None:
            extra_flags += f"--temperature {gen.temperature} "
        if gen and gen.max_tokens is not None:
            extra_flags += f"--max-tokens {gen.max_tokens} "
        if bench.max_problems is not None:
            extra_flags += f"--max-problems {bench.max_problems} "
        if cluster.auto_resume:
            extra_flags += "--resume "

        safe_name = _safe(bench.name)

        eval_image = ""
        if use_containers:
            eval_image = (
                resolve_eval_image(bench.name, base_override=base_override)
                or default_base_image(base_override)
            )

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

    if is_sharded:
        parts.append(_SHARD_MERGE_HINT.format(parent_output_dir=output_dir))
    else:
        report_cmds = []
        ext_map = {
            "markdown": "md", "html": "html", "csv": "csv",
            "json": "json", "latex": "tex",
        }
        for fmt in config.output.report:
            ext = ext_map.get(fmt, fmt)
            report_cmds.append(
                f'nel eval report "$OUTPUT_DIR" -f {fmt} -o "$OUTPUT_DIR/report.{ext}"'
            )
        if report_cmds:
            parts.append(_REPORT.format(report_commands="\n".join(report_cmds)))

        if cluster.auto_resume:
            parts.append(_AUTO_RESUME.format(
                max_attempts=cluster.max_resume_attempts,
                script_path="$OUTPUT_DIR/nel_eval.sbatch",
            ))

    if sb_bench and isinstance(sb_bench.sandbox, ApptainerSandbox):
        het_group_flag = (
            f" --het-group={pool_to_het[sandbox_pool_name]}"
            if sandbox_pool_name and sandbox_pool_name in pool_to_het
            else ""
        )
        parts.append(_APPTAINER_CLEANUP.format(het_group_flag=het_group_flag))

    kill_cmds = []
    for name, svc in config.services.items():
        if not isinstance(svc, ExternalApiService):
            upper = _safe(name).upper()
            kill_cmds.append(f'kill ${upper}_PID 2>/dev/null || true')
    for name, svc in config.services.items():
        if not isinstance(svc, ExternalApiService):
            upper = _safe(name).upper()
            kill_cmds.append(f'wait ${upper}_PID 2>/dev/null || true')

    parts.append(_CLEANUP.format(
        kill_commands="\n".join(kill_cmds) if kill_cmds else "echo 'No managed services.'"
    ))

    return "\n".join(parts)


def write_sbatch(config: EvalConfig, output_dir: str | Path | None = None) -> Path:
    out = Path(output_dir or config.output.dir)
    out.mkdir(parents=True, exist_ok=True)

    script = generate_sbatch(config)
    path = out / "nel_eval.sbatch"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path
