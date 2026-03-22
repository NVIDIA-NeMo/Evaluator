"""Generate self-contained sbatch scripts from EvalConfig."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nemo_evaluator.eval.config import (
    ApptainerSandbox,
    EvalConfig,
    ExternalApiService,
    GymResourceService,
    NatAgentService,
    NoSandbox,
    SimpleSolver,
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
#SBATCH --output={output_dir}/slurm-%j.log
#SBATCH --error={output_dir}/slurm-%j.log
#SBATCH --time={walltime}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
{gres_line}
{partition_line}
{account_line}
{array_line}

set -uo pipefail
export NEL_INNER_EXECUTION=1
export PYTHONUNBUFFERED=1

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
JOB_START_EPOCH=$(date +%s)
mkdir -p "$OUTPUT_DIR"

echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/.nel_job_chain"

{resume_tracking}
echo "=== NeMo Evaluator ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -Iseconds)"
{resume_attempt_info}
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
#SBATCH --output={output_dir}/slurm-%j.log
#SBATCH --error={output_dir}/slurm-%j.log
#SBATCH --time={walltime}
{account_line}

set -uo pipefail
export NEL_INNER_EXECUTION=1
export PYTHONUNBUFFERED=1

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
JOB_START_EPOCH=$(date +%s)
mkdir -p "$OUTPUT_DIR"

echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/.nel_job_chain"

{resume_tracking}
echo "=== NeMo Evaluator (het-job) ==="
echo "Job ID: $SLURM_JOB_ID"
{het_echo_lines}
echo "Start: $(date -Iseconds)"
{resume_attempt_info}
"""

_SECRETS_SOURCE = """\
# Load credentials from separate file (not embedded in this script).
# Restrict permissions: chmod 600 "$BASE_OUTPUT_DIR/.secrets.env"
BASE_OUTPUT_DIR="$OUTPUT_DIR"
if [ -f "$BASE_OUTPUT_DIR/.secrets.env" ]; then
    set -a
    source "$BASE_OUTPUT_DIR/.secrets.env"
    set +a
fi
"""

_CONDA_ACTIVATE = """\
source /opt/anaconda3/bin/activate {conda_env}
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

_TASK_CONFIG = """\
# Benchmark {idx}/{total}: {bench_name} (full config)
echo ""
echo "============================================================"
echo "  Benchmark {idx}/{total}: {bench_name} (repeats={repeats})"
echo "============================================================"
export {svc_url_var}="${{{model_url_bash}}}"
export {svc_model_var}="${{{model_id_bash}}}"
export NEL_OUTPUT_DIR="$OUTPUT_DIR/{safe_name}"
mkdir -p "$NEL_OUTPUT_DIR"
{run_prefix}nel eval run "$OUTPUT_DIR/config_{safe_name}.yaml" {extra_flags}|| {{ echo "  FAILED: {bench_name}"; NEL_EXIT_CODE=1; }}
"""

_REPORT = """\
# Generate reports
echo ""
echo "=== Generating reports ==="
{report_commands}
"""

_PREFLIGHT_IMAGE_CHECK = """\
# Pre-flight: verify container images are pullable
echo "Checking container images..."
{checks}
echo "  All images OK."
"""

_IMAGE_CHECK_LINE = """\
echo "  {label}: {image}"
srun --overlap --nodes 1 --ntasks 1 --container-image {image} true 2>&1 \\
    | head -5 || {{ echo "FATAL: cannot pull image for {label}: {image}"; exit 1; }}"""


def _preflight_image_checks(config: EvalConfig, cluster: SlurmCluster) -> str:
    """Generate bash that validates all container images before starting services."""
    images: dict[str, str] = {}
    for name, svc in config.services.items():
        img = getattr(svc, "image", None)
        if img:
            images[name] = img
    eval_img = getattr(cluster, "eval_image", None)
    if eval_img:
        images["eval-runner"] = eval_img

    if not images:
        return ""

    checks = [
        _IMAGE_CHECK_LINE.format(label=label, image=img)
        for label, img in images.items()
    ]
    return _PREFLIGHT_IMAGE_CHECK.format(checks="\n".join(checks))


_CLEANUP_FUNC = """\
# Cleanup function — called on EXIT (success, failure, or signal)
cleanup() {{
    echo ""
    echo "Shutting down services..."
    {kill_commands}
    echo "=== Evaluation complete ==="
    echo "End: $(date -Iseconds)"
    echo "Results: $OUTPUT_DIR"
}}
trap cleanup EXIT
"""

_FOOTER = """\
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

_RESUME_TRACKING = """\
ATTEMPT_FILE="$OUTPUT_DIR/.nel_attempt"
if [ -f "$ATTEMPT_FILE" ]; then
    NEL_ATTEMPT=$(cat "$ATTEMPT_FILE")
else
    NEL_ATTEMPT=0
fi
NEL_ATTEMPT=$((NEL_ATTEMPT + 1))
echo $NEL_ATTEMPT > "$ATTEMPT_FILE"
"""

_RESUME_ATTEMPT_INFO = """\
echo "Attempt: $NEL_ATTEMPT / {max_attempts}"
"""

_AUTO_RESUME = """\
# Auto-resume: check if evaluation needs to continue
if [ $NEL_ATTEMPT -ge {max_attempts} ]; then
    echo "Max resume attempts ({max_attempts}) reached. Not resubmitting."
else
    JOB_RUNTIME=$(( $(date +%s) - JOB_START_EPOCH ))
    if [ $JOB_RUNTIME -lt 600 ] && [ ! -f "$OUTPUT_DIR/checkpoint.json" ]; then
        echo "Job failed quickly (${{JOB_RUNTIME}}s) without progress. Not resubmitting."
    elif python3 -c "
import json, sys, os
cp_path = '$OUTPUT_DIR/checkpoint.json'
if not os.path.exists(cp_path):
    sys.exit(0)
cp = json.load(open(cp_path))
total = {total_benchmarks}
failed = len(cp.get('failed_benchmarks', {{}}))
completed = len(cp.get('completed_benchmarks', {{}}))
reports_ok = os.path.exists('$OUTPUT_DIR/report.md') or total == 0
sys.exit(0 if (failed > 0 or completed < total or not reports_ok) else 1)
" 2>/dev/null; then
        echo "Evaluation incomplete, resubmitting (attempt $NEL_ATTEMPT/{max_attempts})..."
        NEXT_JOB=$(sbatch --dependency=afternotok:$SLURM_JOB_ID "{script_path}")
        echo "Resubmitted: $NEXT_JOB"
    else
        echo "All benchmarks completed and reports generated. No resubmit needed."
    fi
fi
"""


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


_MODEL_CMD = {
    "vllm": ("vllm serve", "", "--tensor-parallel-size"),
    "sglang": ("sglang serve", "", "--tp-size"),
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


def _build_srun_prefix(
    svc, deploy_image: str, *, het_flag: str, cluster_mounts: list[str], cluster_env_keys: list[str], mount_home: bool
) -> str:
    """Build a per-service srun command with merged mounts and env."""
    if not deploy_image:
        return ""

    svc_mounts = getattr(svc, "container_mounts", [])
    all_mounts = cluster_mounts + svc_mounts
    mount_parts = [f"--container-mounts={m}" for m in all_mounts]
    if mount_home:
        mount_parts.append("--container-mounts=$HOME:$HOME")
    else:
        mount_parts.append("--no-container-mount-home")

    svc_env = getattr(svc, "extra_env", {})
    env_parts = [f"--container-env={k}" for k in cluster_env_keys]
    env_parts += [f"--container-env={k}" for k in svc_env]

    num_nodes = getattr(svc, "num_nodes", 1)
    return (
        f"srun --overlap --nodes {num_nodes} --ntasks 1{het_flag} "
        f"--container-image {deploy_image} "
        f"{' '.join(mount_parts)} {' '.join(env_parts)} "
    )


def _service_env_exports(svc) -> str:
    """Generate export lines for service-specific env vars."""
    svc_env = getattr(svc, "extra_env", {})
    if not svc_env:
        return ""
    lines = [f"export {k}={shlex.quote(v)}" for k, v in svc_env.items()]
    return "\n".join(lines) + "\n"


def _service_block(
    name: str,
    svc,
    *,
    use_containers: bool = False,
    cluster_mounts: list[str] | None = None,
    cluster_env_keys: list[str] | None = None,
    mount_home: bool = True,
    pool_to_het: dict[str, int] | None = None,
) -> str:
    upper = _safe(name).upper()
    pool = getattr(svc, "node_pool", None)
    het_flag = ""
    if pool and pool_to_het and pool in pool_to_het:
        het_flag = f" --het-group={pool_to_het[pool]}"

    if svc.type in _MODEL_CMD:
        cmd, model_flag, tp_flag_name = _MODEL_CMD[svc.type]
        deploy_image = _resolve_service_image(svc)
        srun_prefix = ""
        if use_containers:
            srun_prefix = _build_srun_prefix(
                svc,
                deploy_image,
                het_flag=het_flag,
                cluster_mounts=cluster_mounts or [],
                cluster_env_keys=cluster_env_keys or [],
                mount_home=mount_home,
            )
        env_exports = _service_env_exports(svc)
        tp_flag = f"{tp_flag_name} {svc.tensor_parallel_size} " if svc.tensor_parallel_size else ""
        extra = " ".join(svc.extra_args)
        cuda = ""
        if svc.gpus and isinstance(svc.gpus, list):
            cuda = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in svc.gpus)} "
        return env_exports + _MODEL_SERVICE.format(
            name=name,
            name_upper=upper,
            svc_type=svc.type,
            cmd=cmd,
            model_flag=model_flag,
            model=svc.model or "",
            port=svc.port,
            tp_flag=tp_flag,
            extra_args=extra,
            cuda_prefix=cuda,
            srun_prefix=srun_prefix,
        )

    if isinstance(svc, NatAgentService):
        deploy_image = _resolve_service_image(svc)
        srun_prefix = ""
        if use_containers:
            srun_prefix = _build_srun_prefix(
                svc,
                deploy_image,
                het_flag=het_flag,
                cluster_mounts=cluster_mounts or [],
                cluster_env_keys=cluster_env_keys or [],
                mount_home=mount_home,
            )
        config_file = svc.nat_config_file or "config.yml"
        return _NAT_SERVICE.format(
            name=name,
            name_upper=upper,
            port=svc.port,
            config_file=config_file,
            srun_prefix=srun_prefix,
        )

    if isinstance(svc, GymResourceService):
        if svc.server_cmd:
            return _GYM_CMD_SERVICE.format(
                name=name,
                name_upper=upper,
                server_cmd=svc.server_cmd,
            )
        return _GYM_SERVICE.format(
            name=name,
            name_upper=upper,
            benchmark=svc.benchmark or "",
            port=svc.port,
        )

    if isinstance(svc, ExternalApiService):
        return (
            f"# Service: {name} (external API)\n"
            f'{upper}_URL="{svc.url or ""}"\n'
            f'{upper}_MODEL="{svc.model or ""}"\n'
            f'{upper}_PID=""\n'
        )

    return (
        f"# Service: {name} ({svc.type})\n"
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
            name=name,
            name_upper=upper,
            url=url,
            max_attempts=max_attempts,
        )

    health = getattr(svc, "health_path", "/health") or "/health"
    return _HEALTH_WAIT.format(
        name=name,
        name_upper=upper,
        url=url,
        health_path=health,
        max_attempts=max_attempts,
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


def _needs_full_config(bench) -> bool:
    """True if the benchmark can't use --bench quick mode (needs the full config YAML)."""
    return not isinstance(bench.solver, SimpleSolver) or not isinstance(bench.sandbox, NoSandbox)


def _extract_bench_config(config: EvalConfig, bench_idx: int, svc_url_var: str, svc_model_var: str) -> dict:
    """Build a standalone config dict for one benchmark.

    Managed model services are replaced with external API services whose
    ``url`` and ``model`` reference shell env-vars (expanded by NEL at
    load time).  Cluster is forced to ``local`` since this runs inside
    the sbatch job.
    """
    bench = config.benchmarks[bench_idx]
    svc_name = _get_solver_service(bench) or ""
    svc = config.services.get(svc_name)

    _PROTO_SUFFIX = {
        "chat_completions": "/chat/completions",
        "completions": "/completions",
        "responses": "/responses",
    }

    services: dict = {}
    if svc:
        proto = getattr(svc, "protocol", "chat_completions")
        url_suffix = _PROTO_SUFFIX.get(proto, "/chat/completions")
        svc_dict: dict = {
            "type": "api",
            "url": f"${{{svc_url_var}}}{url_suffix}",
            "protocol": proto,
            "model": f"${{{svc_model_var}}}",
        }
        api_key = getattr(svc, "api_key", None)
        if api_key:
            svc_dict["api_key"] = api_key
        if hasattr(svc, "interceptors") and svc.interceptors:
            svc_dict["interceptors"] = [ic.model_dump(exclude_none=True) for ic in svc.interceptors]
        if hasattr(svc, "generation"):
            gen = svc.generation.model_dump(exclude_none=True)
            if gen:
                svc_dict["generation"] = gen
        svc_dict["proxy_verbose"] = getattr(svc, "proxy_verbose", False)
        services[svc_name] = svc_dict

    bench_dict = bench.model_dump(exclude_none=True)
    bench_dict.pop("scoring", None)

    return {
        "services": services,
        "benchmarks": [bench_dict],
        "output": {"dir": "${NEL_OUTPUT_DIR}"},
    }


def generate_sbatch(config: EvalConfig) -> tuple[str, dict[str, dict], dict[str, str]]:
    cluster = config.cluster
    if not isinstance(cluster, SlurmCluster):
        raise ValueError(f"generate_sbatch requires a SlurmCluster, got {type(cluster).__name__}")

    output_dir = config.output.dir
    job_name = _safe(config.benchmarks[0].name) if len(config.benchmarks) == 1 else "multi"
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
            parts.append(
                _HEADER_HETJOB_POOL.format(
                    het_idx=i,
                    pool_name=pool_name,
                    nodes=pool.nodes,
                    ntasks_per_node=pool.ntasks_per_node,
                    gres_line=gres_line,
                    partition_line=partition_line,
                )
            )
            if i < len(used_pools) - 1:
                parts.append(_HEADER_HETJOB_SEPARATOR)

        het_echo_lines = "\n".join(
            f'echo "Het-group {i} ({name}): $SLURM_JOB_NODELIST_HET_GROUP_{i}"' for i, name in enumerate(used_pools)
        )
        resume_tracking = ""
        resume_attempt_info = ""
        if cluster.auto_resume:
            resume_tracking = _RESUME_TRACKING
            resume_attempt_info = _RESUME_ATTEMPT_INFO.format(
                max_attempts=cluster.max_resume_attempts,
            )
        parts.append(
            _HEADER_HETJOB_FOOTER.format(
                job_name=job_name,
                output_dir=output_dir,
                walltime=cluster.walltime,
                account_line=account_line,
                het_echo_lines=het_echo_lines,
                resume_tracking=resume_tracking,
                resume_attempt_info=resume_attempt_info,
            )
        )
    else:
        pool = cluster.node_pools[used_pools[0]]
        gres_line = f"#SBATCH --gres={pool.gres}" if pool.gres else ""
        partition_line = f"#SBATCH --partition={pool.partition}" if pool.partition else ""
        array_line = f"#SBATCH --array=0-{cluster.shards - 1}" if cluster.shards else ""
        resume_tracking = ""
        resume_attempt_info = ""
        if cluster.auto_resume:
            resume_tracking = _RESUME_TRACKING
            resume_attempt_info = _RESUME_ATTEMPT_INFO.format(
                max_attempts=cluster.max_resume_attempts,
            )
        parts.append(
            _HEADER.format(
                job_name=job_name,
                output_dir=output_dir,
                walltime=cluster.walltime,
                nodes=pool.nodes,
                ntasks_per_node=pool.ntasks_per_node,
                gres_line=gres_line,
                partition_line=partition_line,
                account_line=account_line,
                array_line=array_line,
                resume_tracking=resume_tracking,
                resume_attempt_info=resume_attempt_info,
            )
        )

    is_sharded = cluster.shards is not None
    secrets_env: dict[str, str] = {}

    if is_sharded:
        for bench in config.benchmarks:
            if _needs_full_config(bench):
                raise ValueError(
                    f"Sharding (shards={cluster.shards}) is incompatible with "
                    f"benchmark '{bench.name}' which requires a sidecar config "
                    f"(solver type '{bench.solver.type}' or sandbox type "
                    f"'{bench.sandbox.type}'). Remove shards or use a simple solver."
                )

    if cluster.container_env:
        secrets_env.update(cluster.container_env)
        parts.append(_SECRETS_SOURCE)

    if is_sharded:
        parts.append(_SHARD_SETUP)

    _any_service_has_image = any(getattr(s, "image", None) for s in config.services.values())
    use_containers = (
        cluster.container_mounts
        or cluster.container_env
        or _any_service_has_image
        or any(getattr(s, "container_mounts", []) for s in config.services.values())
        or getattr(cluster, "eval_image", None)
    )

    cluster_env_keys = list(cluster.container_env.keys())
    for _implicit in ("PYTHONUNBUFFERED", "NEL_INNER_EXECUTION"):
        if _implicit not in cluster_env_keys:
            cluster_env_keys.append(_implicit)

    if use_containers:
        parts.append(_preflight_image_checks(config, cluster))

    if cluster.conda_env:
        parts.append(_CONDA_ACTIVATE.format(conda_env=cluster.conda_env))

    for name, svc in config.services.items():
        parts.append(
            _service_block(
                name,
                svc,
                use_containers=use_containers,
                cluster_mounts=cluster.container_mounts,
                cluster_env_keys=cluster_env_keys,
                mount_home=cluster.mount_home,
                pool_to_het=pool_to_het if use_het else None,
            )
        )

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
            parts.append(
                _SANDBOX_NODES_HETJOB.format(
                    het_group=het_idx,
                    sandbox_node_count=pool.nodes,
                    slots_per_node=slots,
                    total_slots=total_slots,
                )
            )
        elif sandbox_pool_name and sandbox_pool_name in cluster.node_pools:
            pool = cluster.node_pools[sandbox_pool_name]
            total_slots = pool.nodes * slots
            sandbox_start = sum(cluster.node_pools[p].nodes for p in used_pools if p != sandbox_pool_name) + 1
            parts.append(
                _SANDBOX_NODES_INLINE.format(
                    sandbox_start_node=sandbox_start,
                    sandbox_node_count=pool.nodes,
                    slots_per_node=slots,
                    total_slots=total_slots,
                )
            )

        het_group_flag = (
            f" --het-group={pool_to_het[sandbox_pool_name]}"
            if sandbox_pool_name and sandbox_pool_name in pool_to_het
            else ""
        )
        sb_image = sb.image

        if isinstance(sb, ApptainerSandbox) and sb_image:
            sif_cache = sb.sif_cache_dir or "/tmp/nel_sif_cache"
            parts.append(
                _APPTAINER_PRE_PROVISION.format(
                    sif_cache_dir=sif_cache,
                    images=shlex.quote(sb_image),
                    het_group_flag=het_group_flag,
                )
            )
        elif isinstance(sb, SlurmSandbox) and sb_image:
            parts.append(
                _SANDBOX_PRE_PULL.format(
                    images=shlex.quote(sb_image),
                )
            )

    base_override = getattr(cluster, "eval_image", None)
    total = len(config.benchmarks)
    sidecar_configs: dict[str, dict] = {}

    eval_run_prefix = ""
    if use_containers and base_override:
        _eval_mounts = [f"--container-mounts={m}" for m in cluster.container_mounts]
        _eval_mounts.append(f"--container-mounts={output_dir}:{output_dir}")
        if cluster.mount_home:
            _eval_mounts.append("--container-mounts=$HOME:$HOME")
        else:
            _eval_mounts.append("--no-container-mount-home")
        _eval_envs = [f"--container-env={k}" for k in cluster_env_keys]
        eval_run_prefix = (
            f"srun --overlap --nodes 1 --ntasks 1 "
            f"--container-image {base_override} "
            f"{' '.join(_eval_mounts)} {' '.join(_eval_envs)} \\\n    "
        )

    for i, bench in enumerate(config.benchmarks, 1):
        svc_name = _get_solver_service(bench) or ""
        upper = _safe(svc_name).upper() if svc_name else "MODEL"
        model_url_var = f"{upper}_URL"
        model_id_var = f"{upper}_MODEL"
        safe_name = _safe(bench.name)

        eval_image = ""
        if use_containers:
            eval_image = resolve_eval_image(bench.name, base_override=base_override) or default_base_image(
                base_override
            )

        run_prefix = ""
        if use_containers and eval_image:
            eval_mount_parts = [f"--container-mounts={m}" for m in cluster.container_mounts]
            eval_mount_parts.append(f"--container-mounts={output_dir}:{output_dir}")
            if cluster.mount_home:
                eval_mount_parts.append("--container-mounts=$HOME:$HOME")
            else:
                eval_mount_parts.append("--no-container-mount-home")
            eval_env_parts = [f"--container-env={k}" for k in cluster_env_keys]
            run_prefix = (
                f"srun --overlap --nodes 1 --ntasks 1 "
                f"--container-image {eval_image} "
                f"{' '.join(eval_mount_parts)} {' '.join(eval_env_parts)} \\\n    "
            )

        if _needs_full_config(bench):
            sidecar = _extract_bench_config(
                config,
                i - 1,
                svc_url_var=model_url_var,
                svc_model_var=model_id_var,
            )
            sidecar_configs[safe_name] = sidecar
            config_extra_flags = "--resume " if cluster.auto_resume else ""
            parts.append(
                _TASK_CONFIG.format(
                    idx=i,
                    total=total,
                    bench_name=bench.name,
                    svc_url_var=model_url_var,
                    svc_model_var=model_id_var,
                    model_url_bash=model_url_var,
                    model_id_bash=model_id_var,
                    repeats=bench.repeats,
                    safe_name=safe_name,
                    run_prefix=run_prefix,
                    extra_flags=config_extra_flags,
                )
            )
        else:
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

            parts.append(
                _TASK.format(
                    idx=i,
                    total=total,
                    bench_name=bench.name,
                    model_url_var=model_url_var,
                    model_id_var=model_id_var,
                    repeats=bench.repeats,
                    extra_flags=extra_flags,
                    safe_name=safe_name,
                    run_prefix=run_prefix,
                )
            )

    if is_sharded:
        parts.append(_SHARD_MERGE_HINT.format(parent_output_dir=output_dir))
    else:
        report_cmds = []
        ext_map = {
            "markdown": "md",
            "html": "html",
            "csv": "csv",
            "json": "json",
            "latex": "tex",
        }
        for fmt in config.output.report:
            ext = ext_map.get(fmt, fmt)
            report_cmds.append(f'{eval_run_prefix}nel eval report "$OUTPUT_DIR" -f {fmt} -o "$OUTPUT_DIR/report.{ext}"')
        if report_cmds:
            parts.append(_REPORT.format(report_commands="\n".join(report_cmds)))

        if cluster.auto_resume:
            parts.append(
                _AUTO_RESUME.format(
                    max_attempts=cluster.max_resume_attempts,
                    total_benchmarks=len(config.benchmarks),
                    script_path="$OUTPUT_DIR/nel_eval.sbatch",
                )
            )

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
            kill_cmds.append(f"    [ -n \"${{{upper}_PID:-}}\" ] && kill ${upper}_PID 2>/dev/null || true")
    for name, svc in config.services.items():
        if not isinstance(svc, ExternalApiService):
            upper = _safe(name).upper()
            kill_cmds.append(f"    [ -n \"${{{upper}_PID:-}}\" ] && wait ${upper}_PID 2>/dev/null || true")

    cleanup_body = "\n".join(kill_cmds) if kill_cmds else "    echo 'No managed services.'"
    parts.insert(1, _CLEANUP_FUNC.format(kill_commands=cleanup_body))

    parts.append(_FOOTER)

    return "\n".join(parts), sidecar_configs, secrets_env


def _redact(value: str) -> str:
    """Mask a secret value for display, showing only the last 4 chars.

    Matches the old evaluator's ``redact_secrets_env_content()`` behaviour:
    all values are redacted (no name-based heuristic).
    """
    if len(value) <= 4:
        return "***"
    return f"***{value[-4:]}"


def _run_id(config: EvalConfig) -> str:
    """Generate a timestamped run ID: YYYYMMDD_HHMMSSZ-{hash8}.

    The hash is derived from benchmark names and model identifiers to
    avoid collisions when multiple jobs are submitted in the same second.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts: list[str] = []
    for b in config.benchmarks:
        parts.append(b.name)
        svc_name = getattr(b.solver, "service", "")
        svc = config.services.get(svc_name) if svc_name else None
        if svc:
            parts.append(getattr(svc, "model", "") or "")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]
    return f"{ts}_{digest}"


def stamp_output_dir(config: EvalConfig) -> None:
    """Append a timestamped run-ID subdirectory to config.output.dir.

    Called once before write_sbatch / generate_sbatch so the timestamped
    path is baked into the sbatch script and all metadata.  Skipped when
    running inside a SLURM job (NEL_INNER_EXECUTION=1) or when the user
    has opted out (output.timestamped=false).
    """
    if (
        not config.output.timestamped
        or os.environ.get("NEL_INNER_EXECUTION") == "1"
    ):
        return
    config.output.dir = str(Path(config.output.dir) / _run_id(config))


def write_sbatch(config: EvalConfig, output_dir: str | Path | None = None) -> tuple[Path, list[Path]]:
    """Write sbatch script + sidecar config YAMLs + .secrets.env.

    Returns (sbatch_path, list_of_extra_paths).
    The extra paths include sidecar configs and .secrets.env, all of
    which must be copied alongside the sbatch script for SSH submission.
    """
    out = Path(output_dir or config.output.dir)
    out.mkdir(parents=True, exist_ok=True)

    script, sidecar_configs, secrets_env = generate_sbatch(config)
    path = out / "nel_eval.sbatch"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)

    extra_paths: list[Path] = []

    if secrets_env:
        secrets_path = out / ".secrets.env"
        lines = [f"export {k}={shlex.quote(v)}" for k, v in secrets_env.items()]
        secrets_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        secrets_path.chmod(0o600)
        extra_paths.append(secrets_path)

    for safe_name, cfg_dict in sidecar_configs.items():
        cfg_path = out / f"config_{safe_name}.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False), encoding="utf-8")
        extra_paths.append(cfg_path)

    return path, extra_paths
