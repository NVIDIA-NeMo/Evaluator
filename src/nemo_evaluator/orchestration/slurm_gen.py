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
"""Generate self-contained sbatch scripts from EvalConfig."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

import yaml

from nemo_evaluator.config import (
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
from nemo_evaluator.config.clusters import _parse_walltime
from nemo_evaluator.orchestration.image_resolver import (
    default_base_image,
    resolve_deployment_image,
    resolve_eval_image,
)
from nemo_evaluator.orchestration.secrets_env import (
    SecretsEnvResult,
    build_reexport_commands,
    generate_secrets_env,
    redact_secrets_env_content,
    reexport_keys,
)

_HEADER = """\
#!/bin/bash
#SBATCH --job-name=nel-eval-{job_name}
#SBATCH --output={output_dir}/logs/slurm-%j.log
#SBATCH --error={output_dir}/logs/slurm-%j.log
#SBATCH --time={walltime}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --no-requeue
{gres_line}
{gpus_per_node_line}
{partition_line}
{account_line}
{sbatch_comment_line}
{sbatch_extra_lines}

set -uo pipefail
export NEL_INNER_EXECUTION=1
export PYTHONUNBUFFERED=1

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
JOB_START_EPOCH=$(date +%s)
mkdir -p "$OUTPUT_DIR/logs"

echo "=== NeMo Evaluator ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -Iseconds)"
"""

_SHARD_ENV = """\
# Shard {shard_idx} of {total_shards} (independent job)
export NEL_SHARD_IDX={shard_idx}
export NEL_TOTAL_SHARDS={total_shards}
rm -f "$OUTPUT_DIR/.shard_done"
echo "Shard: $NEL_SHARD_IDX / $NEL_TOTAL_SHARDS  (output: $OUTPUT_DIR)"
"""

_SHARD_FINISH = """\
# Mark this shard as complete and auto-merge if last
_PARENT_DIR="{parent_output_dir}"
if [[ $NEL_EXIT_CODE -eq 0 ]]; then
    touch "$OUTPUT_DIR/.shard_done"
    _DONE=0
    for _s in $(seq 0 $(({total_shards} - 1))); do
        [[ -f "$_PARENT_DIR/shard_$_s/.shard_done" ]] && _DONE=$((_DONE + 1))
    done
    echo "Shard {shard_idx} complete ($_DONE/{total_shards} shards done)."
    if [[ $_DONE -eq {total_shards} ]]; then
        # Remove stale lock from a previously failed merge attempt
        if [[ -d "$_PARENT_DIR/.merge_lock" ]] && [[ ! -f "$_PARENT_DIR/.merge_lock/.done" ]]; then
            rm -rf "$_PARENT_DIR/.merge_lock"
        fi
        if mkdir "$_PARENT_DIR/.merge_lock" 2>/dev/null; then
            echo ""
            echo "=== All {total_shards} shards complete — running merge ==="
            {merge_prefix}nel eval merge "$_PARENT_DIR"
            _MERGE_RC=$?
            if [[ $_MERGE_RC -ne 0 ]]; then
                echo "ERROR: Merge failed (exit $_MERGE_RC). To retry:"
                echo "  rmdir '{parent_output_dir}/.merge_lock' 2>/dev/null; nel eval merge '{parent_output_dir}'"
                rm -rf "$_PARENT_DIR/.merge_lock"
                NEL_EXIT_CODE=1
            else
                touch "$_PARENT_DIR/.merge_lock/.done"
                {report_commands}
            fi
        fi
    fi
else
    echo "Shard {shard_idx} failed (exit $NEL_EXIT_CODE). Skipping completion marker."
    echo "  Manual merge: nel eval merge $_PARENT_DIR"
fi
"""

_HEADER_HETJOB_PREAMBLE = """\
#!/bin/bash
"""

_HEADER_HETJOB_POOL = """\
# ── Het Group {het_idx}: {pool_name} ──
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
{gres_line}
{gpus_per_node_line}
{partition_line}
"""

_HEADER_HETJOB_SEPARATOR = """\
#SBATCH hetjob
"""

_HEADER_HETJOB_FOOTER = """\
#SBATCH --job-name=nel-eval-{job_name}
#SBATCH --output={output_dir}/logs/slurm-%j.log
#SBATCH --error={output_dir}/logs/slurm-%j.log
#SBATCH --time={walltime}
#SBATCH --no-requeue
{account_line}
{sbatch_comment_line}
{sbatch_extra_lines}

set -uo pipefail
export NEL_INNER_EXECUTION=1
export PYTHONUNBUFFERED=1

OUTPUT_DIR="{output_dir}"
NEL_EXIT_CODE=0
JOB_START_EPOCH=$(date +%s)
mkdir -p "$OUTPUT_DIR/logs"

echo "=== NeMo Evaluator (het-job) ==="
echo "Job ID: $SLURM_JOB_ID"
{het_echo_lines}
echo "Start: $(date -Iseconds)"
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
echo "  Logs: $OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log"
{srun_prefix}{cuda_prefix}{service_cmd}>> "$OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log" 2>&1 &
{name_upper}_PID=$!
ln -sf "server-{name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/server-{name}.log"
{name_upper}_URL="http://localhost:{port}/v1"
{name_upper}_MODEL={model}
"""

_GYM_SERVICE = """\
# Service: {name} (gym)
echo "Starting benchmark server: {name}..."
echo "  Logs: $OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log"
nel serve -b {benchmark} --host 0.0.0.0 -p {port} >> "$OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log" 2>&1 &
{name_upper}_PID=$!
ln -sf "server-{name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/server-{name}.log"
"""

_GYM_CMD_SERVICE = """\
# Service: {name} (gym / custom server)
echo "Starting server: {name}..."
echo "  Logs: $OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log"
({server_cmd}) >> "$OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log" 2>&1 &
{name_upper}_PID=$!
ln -sf "server-{name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/server-{name}.log"
"""

_NAT_SERVICE = """\
# Service: {name} (nat agent)
echo "Starting NAT agent server: {name}..."
echo "  Logs: $OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log"
{srun_prefix}nat serve --config_file {config_file} --port {port} --host 0.0.0.0 >> "$OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log" 2>&1 &
{name_upper}_PID=$!
ln -sf "server-{name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/server-{name}.log"
{name_upper}_URL="http://localhost:{port}"
{name_upper}_MODEL="nat-agent"
"""

_MULTINODE_IP_DISCOVERY = """\
# Multi-node IP discovery
MASTER_IP=$(scontrol show hostname "${nodelist_var}" | head -n 1)
echo "Master IP: $MASTER_IP"
export MASTER_IP
"""

_RAY_MULTINODE_SERVICE = """\
# Service: {name} ({svc_type}, multi-node Ray: {num_nodes} nodes)
echo "Starting multi-node Ray {svc_type} server: {name}..."
echo "  Logs: $OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log"

RAY_PORT={ray_port}
_GPUS_PER_NODE=${{SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}
if [ "$_GPUS_PER_NODE" -eq 0 ] 2>/dev/null || [ -z "$_GPUS_PER_NODE" ]; then
    echo "FATAL: Cannot determine GPU count for Ray. Set gpus_per_node in cluster config."
    exit 1
fi
export RAY_ADDRESS="localhost:$RAY_PORT"

# Start Ray head on first node, workers on remaining nodes
{srun_prefix}bash -c '
set -uo pipefail
trap "kill 0 2>/dev/null" EXIT
_GPUS='"$_GPUS_PER_NODE"'
_MASTER='"$MASTER_IP"'
_RAY_PORT='"$RAY_PORT"'
PROC_ID=$SLURM_PROCID
if [ "$PROC_ID" -eq 0 ]; then
    echo "[Ray head] Starting on $(hostname) with $_GPUS GPUs"
    ray start --head --port=$_RAY_PORT --num-gpus=$_GPUS --block &
    RAY_HEAD_PID=$!
    for _i in $(seq 1 120); do
        ray status 2>/dev/null && break
        sleep 2
    done
    echo "[Ray head] Launching {svc_type}..."
    export RAY_ADDRESS="localhost:$_RAY_PORT"
    {service_cmd}
    _SVC_RC=$?
    if [ $_SVC_RC -ne 0 ]; then
        echo "[Ray head] {svc_type} exited with code $_SVC_RC"
        ray stop 2>/dev/null; kill $RAY_HEAD_PID 2>/dev/null
        exit $_SVC_RC
    fi
    wait $RAY_HEAD_PID
else
    echo "[Ray worker $PROC_ID] Joining $_MASTER:$_RAY_PORT with $_GPUS GPUs"
    _joined=0
    for _i in $(seq 1 60); do
        ray start --address="$_MASTER:$_RAY_PORT" --num-gpus=$_GPUS --block && {{ _joined=1; break; }}
        echo "[Ray worker $PROC_ID] Retry $_i/60..."
        sleep 5
    done
    if [ "$_joined" -eq 0 ]; then
        echo "[Ray worker $PROC_ID] FATAL: Failed to join Ray cluster after 60 attempts"
        exit 1
    fi
fi
' >> "$OUTPUT_DIR/logs/server-{name}-$SLURM_JOB_ID.log" 2>&1 &
{name_upper}_PID=$!
ln -sf "server-{name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/server-{name}.log"
{name_upper}_URL="http://localhost:{port}/v1"
{name_upper}_MODEL={model}
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
echo "  Logs: $OUTPUT_DIR/logs/eval-{safe_name}-$SLURM_JOB_ID.log"
{run_prefix}nel eval run \\
    --bench "{bench_name}" \\
    --model-url "${model_url_var}" \\
    --model-id "${model_id_var}" \\
    --repeats {repeats} \\
    {extra_flags}\\
    -o "$OUTPUT_DIR/{safe_name}" 2>&1 | stdbuf -oL tee -a "$OUTPUT_DIR/logs/eval-{safe_name}-$SLURM_JOB_ID.log"
_EVAL_RC=${{PIPESTATUS[0]}}
ln -sf "eval-{safe_name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/eval-{safe_name}.log"
if [ $_EVAL_RC -ne 0 ]; then echo "  FAILED: {bench_name}"; NEL_EXIT_CODE=1; fi
"""

_TASK_CONFIG = """\
# Benchmark {idx}/{total}: {bench_name} (full config)
echo ""
echo "============================================================"
echo "  Benchmark {idx}/{total}: {bench_name} (repeats={repeats})"
echo "============================================================"
echo "  Logs: $OUTPUT_DIR/logs/eval-{safe_name}-$SLURM_JOB_ID.log"
export {svc_url_var}="${{{model_url_bash}}}"
export {svc_model_var}="${{{model_id_bash}}}"
export NEL_OUTPUT_DIR="$OUTPUT_DIR/{safe_name}"
mkdir -p "$NEL_OUTPUT_DIR"
{run_prefix}nel eval run "$OUTPUT_DIR/config_{safe_name}.yaml" {extra_flags}2>&1 | stdbuf -oL tee -a "$OUTPUT_DIR/logs/eval-{safe_name}-$SLURM_JOB_ID.log"
_EVAL_RC=${{PIPESTATUS[0]}}
ln -sf "eval-{safe_name}-$SLURM_JOB_ID.log" "$OUTPUT_DIR/logs/eval-{safe_name}.log"
if [ $_EVAL_RC -ne 0 ]; then echo "  FAILED: {bench_name}"; NEL_EXIT_CODE=1; fi
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

    checks = [_IMAGE_CHECK_LINE.format(label=label, image=img) for label, img in images.items()]
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

_AUTORESUME_PROLOGUE = """\
# --- Auto-resume chain ---
_this_script="$OUTPUT_DIR/nel_eval.sbatch"
_prev_slurm_job_id="${{1:-}}"
_walltime_file="$OUTPUT_DIR/.nel_accumulated_walltime"
_retry_file="$OUTPUT_DIR/.nel_infra_retries"

if [[ "$_prev_slurm_job_id" != "" ]]; then
    for _sacct_try in 1 2 3 4 5; do
        _prev_state=$(sacct -j $_prev_slurm_job_id -P -n -o State | head -n 1)
        [[ -n "$_prev_state" ]] && break
        sleep 2
    done
    _prev_elapsed=$(sacct -j $_prev_slurm_job_id -P -n -o ElapsedRaw | head -n 1)
    _prev_elapsed=${{_prev_elapsed:-0}}
    _accumulated=$(cat "$_walltime_file" 2>/dev/null || echo 0)
    _accumulated=$((_accumulated + _prev_elapsed))
    echo $_accumulated > "$_walltime_file"

    if [[ $_prev_state == 'COMPLETED' ]]; then
        echo "Previous job $_prev_slurm_job_id completed successfully. Exiting."
        exit 0
    elif [[ $_prev_state == CANCELLED* ]]; then
        echo "Previous job $_prev_slurm_job_id was cancelled. Stopping chain."
        exit 0
    elif [[ $_prev_state == 'TIMEOUT' || $_prev_state == 'PREEMPTED' || $_prev_state == 'NODE_FAIL' ]]; then
        echo "Previous job $_prev_slurm_job_id: $_prev_state. Resuming..."
{max_walltime_check}
    else
        _retries=$(cat "$_retry_file" 2>/dev/null || echo 0)
        _retries=$((_retries + 1))
        echo $_retries > "$_retry_file"
        if [[ $_retries -ge {max_retries} ]]; then
            echo "Infra retry limit ({max_retries}) reached after $_prev_state. Stopping."
            exit 1
        fi
        echo "Previous job $_prev_slurm_job_id: $_prev_state. Infra retry $_retries/{max_retries}..."
    fi
fi

echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/.nel_job_chain"
_next_output=$(sbatch --dependency=afternotok:$SLURM_JOB_ID "$_this_script" $SLURM_JOB_ID 2>&1) && {{
    _next_id=$(echo "$_next_output" | grep -oE '[0-9]+')
    if [[ -n "$_next_id" ]]; then
        echo "Auto-resume follow-up queued: $_next_id (afternotok:$SLURM_JOB_ID)"
    fi
}} || echo "WARNING: Failed to submit auto-resume follow-up. Chain will NOT continue on failure."
"""

_MAX_WALLTIME_CHECK = """\
        if [[ $_accumulated -ge {max_walltime_seconds} ]]; then
            echo "Max walltime ({max_walltime}) exceeded ($_accumulated s). Stopping chain."
            exit 1
        fi"""


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


_MODEL_CMD = {
    "vllm": ("vllm serve", "", "--tensor-parallel-size", "--data-parallel-size"),
    "sglang": ("sglang serve", "", "--tp-size", "--dp-size"),
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
    svc,
    deploy_image: str,
    *,
    het_flag: str,
    cluster_mounts: list[str],
    env_keys: list[str],
    mount_home: bool,
    extra_mounts: list[str] | None = None,
) -> str:
    """Build a per-service srun command with merged mounts and env.

    ``env_keys`` is the already-resolved list of original env var names
    that this container needs (from the group's remappings + implicit vars).
    The caller is responsible for emitting re-export commands beforehand.
    """
    if not deploy_image:
        return ""

    svc_mounts = getattr(svc, "container_mounts", [])
    all_mounts = list(dict.fromkeys(cluster_mounts + svc_mounts + (extra_mounts or [])))
    if mount_home:
        all_mounts.append("$HOME:$HOME")
    mount_flag = f"--container-mounts={','.join(all_mounts)} " if all_mounts else ""
    home_flag = "" if mount_home else "--no-container-mount-home "

    env_parts = [f"--container-env={k}" for k in env_keys]

    num_nodes = getattr(svc, "num_nodes", 1)
    ntasks = num_nodes if num_nodes > 1 else 1
    label_flag = " --label" if num_nodes > 1 else ""
    if num_nodes > 1:
        for extra_key in ("MASTER_IP", "RAY_ADDRESS"):
            if f"--container-env={extra_key}" not in env_parts:
                env_parts.append(f"--container-env={extra_key}")
    return (
        f"srun --mpi=pmix --overlap --nodes {num_nodes} --ntasks {ntasks}{label_flag}{het_flag} "
        f"--container-image {deploy_image} "
        f"{mount_flag}{home_flag}{' '.join(env_parts)} "
    )


def _service_block(
    name: str,
    svc,
    *,
    use_containers: bool = False,
    cluster_mounts: list[str] | None = None,
    env_keys: list[str] | None = None,
    reexport_cmds: str = "",
    mount_home: bool = True,
    pool_to_het: dict[str, int] | None = None,
) -> str:
    upper = _safe(name).upper()
    pool = getattr(svc, "node_pool", None)
    het_flag = ""
    if pool and pool_to_het and pool in pool_to_het:
        het_flag = f" --het-group={pool_to_het[pool]}"

    if svc.type in _MODEL_CMD:
        cmd, model_flag, tp_flag_name, dp_flag_name = _MODEL_CMD[svc.type]
        deploy_image = _resolve_service_image(svc)

        # Auto-mount local model paths into the container at /model (read-only),
        # mirroring the checkpoint_path:/checkpoint pattern from nemo-evaluator-launcher.
        model_for_cmd = svc.model
        model_mount: list[str] = []
        if svc.model and svc.model.startswith("/") and use_containers:
            model_for_cmd = "/model"
            model_mount = [f"{svc.model}:/model:ro"]

        model_api_name = getattr(svc, "served_model_name", None) or name

        srun_prefix = ""
        if use_containers:
            srun_prefix = _build_srun_prefix(
                svc,
                deploy_image,
                het_flag=het_flag,
                cluster_mounts=cluster_mounts or [],
                env_keys=env_keys or [],
                mount_home=mount_home,
                extra_mounts=model_mount + ["$OUTPUT_DIR:$OUTPUT_DIR"],
            )
        tp_flag = f"{tp_flag_name} {svc.tensor_parallel_size} " if svc.tensor_parallel_size else ""
        dp_flag = f"{dp_flag_name} {svc.data_parallel_size} " if svc.data_parallel_size else ""
        extra = " ".join(shlex.quote(a) for a in svc.extra_args)
        cuda = ""
        if svc.gpus and isinstance(svc.gpus, list):
            cuda = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in svc.gpus)} "

        model_flag_part = (
            f" {model_flag} {model_for_cmd}" if model_flag else f" {model_for_cmd}" if model_for_cmd else ""
        )
        served_name_flag = f"--served-model-name {shlex.quote(model_api_name)} "
        main_cmd = f"{cmd}{model_flag_part} --port {svc.port} {tp_flag}{dp_flag}{served_name_flag}{extra}"

        setup_list = getattr(svc, "setup_commands", []) or []
        num_nodes = getattr(svc, "num_nodes", 1)
        safe_name = _safe(name)

        if num_nodes > 1:
            full_cmd = f"{main_cmd} --distributed-executor-backend ray"
        else:
            full_cmd = main_cmd

        script_lines = ["set -euo pipefail"] + list(setup_list) + [full_cmd]
        script_file = f"_svc_{safe_name}_cmd_$SLURM_JOB_ID.sh"
        script_heredoc = (
            f"cat > \"$OUTPUT_DIR/logs/{script_file}\" <<'_NEMO_SVC_CMD_EOF_'\n"
            + "\n".join(script_lines)
            + "\n_NEMO_SVC_CMD_EOF_\n"
        )

        reexport_block = f"{reexport_cmds}\n" if reexport_cmds else ""

        if num_nodes > 1:
            # Inside the template's bash -c '...' block, break out of single
            # quotes to expand $OUTPUT_DIR, then re-enter single quotes.
            ray_service_cmd = f"""bash '"$OUTPUT_DIR"'/logs/{script_file}"""
            return (
                reexport_block
                + script_heredoc
                + _RAY_MULTINODE_SERVICE.format(
                    name=name,
                    name_upper=upper,
                    svc_type=svc.type,
                    service_cmd=ray_service_cmd,
                    model=shlex.quote(model_api_name or ""),
                    port=svc.port,
                    num_nodes=num_nodes,
                    ray_port=6379,
                    srun_prefix=srun_prefix,
                )
            )

        service_cmd = f'bash "$OUTPUT_DIR/logs/{script_file}" '
        return (
            reexport_block
            + script_heredoc
            + _MODEL_SERVICE.format(
                name=name,
                name_upper=upper,
                svc_type=svc.type,
                service_cmd=service_cmd,
                model=shlex.quote(model_api_name or ""),
                port=svc.port,
                cuda_prefix=cuda,
                srun_prefix=srun_prefix,
            )
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
                env_keys=env_keys or [],
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
            f"{upper}_MODEL={shlex.quote(svc.model or '')}\n"
            f'{upper}_PID=""\n'
        )

    fallback_model = getattr(svc, "served_model_name", None) or name
    return (
        f"# Service: {name} ({svc.type})\n"
        f'{upper}_URL="http://localhost:{getattr(svc, "port", 8000)}/v1"\n'
        f"{upper}_MODEL={shlex.quote(fallback_model)}\n"
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
            url=svc.base_url,
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
        proxy = getattr(svc, "proxy", None)
        if proxy is not None:
            svc_dict["proxy"] = proxy.model_dump(exclude_none=True, exclude_defaults=True)
        if hasattr(svc, "generation"):
            gen = svc.generation.model_dump(exclude_none=True)
            if gen:
                svc_dict["generation"] = gen
        services[svc_name] = svc_dict

    # Include non-model services (gym) so the inner eval runner can resolve them.
    for attr in ("resource_service", "gym_service"):
        extra_svc_name = getattr(bench.solver, attr, None)
        if extra_svc_name and extra_svc_name not in services:
            extra_svc = config.services.get(extra_svc_name)
            if extra_svc and not extra_svc.is_model_server:
                services[extra_svc_name] = {
                    "type": extra_svc.type,
                    "url": extra_svc.base_url,
                }

    bench_dict = bench.model_dump(exclude_none=True)
    bench_dict.pop("scoring", None)

    return {
        "services": services,
        "benchmarks": [bench_dict],
        "output": {"dir": "${NEL_OUTPUT_DIR}"},
    }


def _format_sbatch_extra_flags(flags: dict) -> str:
    """Format sbatch_extra_flags dict into #SBATCH lines."""
    lines = []
    for key, val in flags.items():
        if val is True:
            lines.append(f"#SBATCH --{key}")
        elif val is not False and val is not None:
            lines.append(f"#SBATCH --{key}={val}")
    return "\n".join(lines)


def _any_multinode_service(config: EvalConfig) -> bool:
    """Return True if any service requests multiple nodes."""
    return any(getattr(svc, "num_nodes", 1) > 1 for svc in config.services.values())


def _compute_pool_nodes(config: EvalConfig, pool_name: str) -> int:
    """Compute total nodes needed for a pool, accounting for multi-node services.

    Returns the max of the pool's declared nodes and any service's num_nodes
    requirement, so that the SBATCH header always requests enough.
    """
    cluster = config.cluster
    if not isinstance(cluster, SlurmCluster):
        return 1
    base = cluster.node_pools[pool_name].nodes
    svc_max = 0
    for svc in config.services.values():
        svc_pool = getattr(svc, "node_pool", None)
        if svc_pool == pool_name:
            svc_max = max(svc_max, getattr(svc, "num_nodes", 1))
    return max(base, svc_max)


def generate_sbatch(
    config: EvalConfig,
    *,
    shard_idx: int | None = None,
    total_shards: int | None = None,
) -> tuple[str, dict[str, dict], SecretsEnvResult]:
    """Generate sbatch script content, sidecar configs, and secrets env.

    When *shard_idx* and *total_shards* are given the generated script is a
    standalone per-shard job (no ``--array``).  Each shard gets its own
    output sub-directory (``shard_N/``) and auto-resume chain.

    Returns:
        (script_text, sidecar_configs, secrets_result)
    """
    cluster = config.cluster
    if not isinstance(cluster, SlurmCluster):
        raise ValueError(f"generate_sbatch requires a SlurmCluster, got {type(cluster).__name__}")

    parent_output_dir = config.output.dir
    output_dir = f"{parent_output_dir}/shard_{shard_idx}" if shard_idx is not None else parent_output_dir
    job_name = _safe(config.benchmarks[0].name) if len(config.benchmarks) == 1 else "multi"
    account_line = f"#SBATCH --account={cluster.account}" if cluster.account else ""
    sbatch_comment_line = f"#SBATCH --comment={shlex.quote(cluster.sbatch_comment)}" if cluster.sbatch_comment else ""
    sbatch_extra_lines = _format_sbatch_extra_flags(cluster.sbatch_extra_flags)

    used_pools = _collect_used_pools(config)
    use_het = len(used_pools) > 1
    pool_to_het = {name: i for i, name in enumerate(used_pools)} if use_het else {}

    # --- Build env groups for disambiguated .secrets.env ---
    env_groups: dict[str, dict[str, str]] = {}

    for name, svc in config.services.items():
        svc_env = {**cluster.container_env, **getattr(svc, "extra_env", {})}
        if svc_env:
            env_groups[f"svc_{name}"] = svc_env

    for bench in config.benchmarks:
        eval_env = dict(cluster.container_env)
        if eval_env:
            env_groups[f"eval_{_safe(bench.name)}"] = eval_env

    secrets_result = generate_secrets_env(env_groups)

    _IMPLICIT_KEYS = ["PYTHONUNBUFFERED", "NEL_INNER_EXECUTION", "NEL_SHARD_IDX", "NEL_TOTAL_SHARDS"]

    parts: list[str] = []

    if use_het:
        parts.append(_HEADER_HETJOB_PREAMBLE)
        for i, pool_name in enumerate(used_pools):
            pool = cluster.node_pools[pool_name]
            effective_nodes = _compute_pool_nodes(config, pool_name)
            gres_line = f"#SBATCH --gres={pool.gres}" if pool.gres else ""
            gpus_per_node_line = f"#SBATCH --gpus-per-node={pool.gpus_per_node}" if pool.gpus_per_node else ""
            partition_line = f"#SBATCH --partition={pool.partition}" if pool.partition else ""
            parts.append(
                _HEADER_HETJOB_POOL.format(
                    het_idx=i,
                    pool_name=pool_name,
                    nodes=effective_nodes,
                    ntasks_per_node=pool.ntasks_per_node,
                    gres_line=gres_line,
                    gpus_per_node_line=gpus_per_node_line,
                    partition_line=partition_line,
                )
            )
            if i < len(used_pools) - 1:
                parts.append(_HEADER_HETJOB_SEPARATOR)

        het_echo_lines = "\n".join(
            f'echo "Het-group {i} ({name}): $SLURM_JOB_NODELIST_HET_GROUP_{i}"' for i, name in enumerate(used_pools)
        )
        parts.append(
            _HEADER_HETJOB_FOOTER.format(
                job_name=job_name,
                output_dir=output_dir,
                walltime=cluster.walltime,
                account_line=account_line,
                sbatch_comment_line=sbatch_comment_line,
                sbatch_extra_lines=sbatch_extra_lines,
                het_echo_lines=het_echo_lines,
            )
        )
    else:
        pool = cluster.node_pools[used_pools[0]]
        effective_nodes = _compute_pool_nodes(config, used_pools[0])
        gres_line = f"#SBATCH --gres={pool.gres}" if pool.gres else ""
        gpus_per_node_line = f"#SBATCH --gpus-per-node={pool.gpus_per_node}" if pool.gpus_per_node else ""
        partition_line = f"#SBATCH --partition={pool.partition}" if pool.partition else ""
        parts.append(
            _HEADER.format(
                job_name=job_name,
                output_dir=output_dir,
                walltime=cluster.walltime,
                nodes=effective_nodes,
                ntasks_per_node=pool.ntasks_per_node,
                gres_line=gres_line,
                gpus_per_node_line=gpus_per_node_line,
                partition_line=partition_line,
                account_line=account_line,
                sbatch_comment_line=sbatch_comment_line,
                sbatch_extra_lines=sbatch_extra_lines,
            )
        )

    is_shard_script = shard_idx is not None

    if cluster.shards is not None and not is_shard_script:
        raise ValueError(
            "generate_sbatch must not be called directly with cluster.shards set. "
            "Use write_sbatch which generates per-shard scripts."
        )

    has_secrets = bool(secrets_result.secrets_content.strip())
    if has_secrets:
        parts.append(_SECRETS_SOURCE)

    if _any_multinode_service(config):
        nodelist_var = "SLURM_JOB_NODELIST"
        if use_het:
            for svc in config.services.values():
                if getattr(svc, "num_nodes", 1) > 1:
                    svc_pool = getattr(svc, "node_pool", None)
                    if svc_pool and svc_pool in pool_to_het:
                        nodelist_var = f"SLURM_JOB_NODELIST_HET_GROUP_{pool_to_het[svc_pool]}"
                    break
        parts.append(_MULTINODE_IP_DISCOVERY.format(nodelist_var=nodelist_var))

    if is_shard_script:
        parts.append(_SHARD_ENV.format(shard_idx=shard_idx, total_shards=total_shards))

    _any_service_has_image = any(getattr(s, "image", None) for s in config.services.values())
    use_containers = (
        cluster.container_mounts
        or cluster.container_env
        or _any_service_has_image
        or any(getattr(s, "container_mounts", []) for s in config.services.values())
        or getattr(cluster, "eval_image", None)
    )

    if use_containers:
        parts.append(_preflight_image_checks(config, cluster))

    if cluster.conda_env:
        parts.append(_CONDA_ACTIVATE.format(conda_env=cluster.conda_env))

    for name, svc in config.services.items():
        group_name = f"svc_{name}"
        svc_env_keys = reexport_keys(group_name, secrets_result) + _IMPLICIT_KEYS
        svc_env_keys = list(dict.fromkeys(svc_env_keys))
        svc_reexport = build_reexport_commands(group_name, secrets_result)
        parts.append(
            _service_block(
                name,
                svc,
                use_containers=use_containers,
                cluster_mounts=cluster.container_mounts,
                env_keys=svc_env_keys,
                reexport_cmds=svc_reexport,
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

    def _eval_srun_prefix(bench_name: str, image: str) -> str:
        """Build srun prefix for eval containers with per-benchmark env keys."""
        group_name = f"eval_{_safe(bench_name)}"
        eval_env_keys = reexport_keys(group_name, secrets_result) + _IMPLICIT_KEYS
        eval_env_keys = list(dict.fromkeys(eval_env_keys))
        _all_mounts = list(cluster.container_mounts) + [f"{output_dir}:{output_dir}"]
        if cluster.mount_home:
            _all_mounts.append("$HOME:$HOME")
        _mount_flag = f"--container-mounts={','.join(_all_mounts)} " if _all_mounts else ""
        _home_flag = "" if cluster.mount_home else "--no-container-mount-home "
        _envs = [f"--container-env={k}" for k in eval_env_keys]
        reexport = build_reexport_commands(group_name, secrets_result)
        prefix = (
            f"srun --mpi=pmix --overlap --unbuffered --nodes 1 --ntasks 1 "
            f"--container-image {image} "
            f"{_mount_flag}{_home_flag}{' '.join(_envs)} \\\n    "
        )
        return f"{reexport}\n{prefix}" if reexport else prefix

    eval_run_prefix = ""
    merge_run_prefix = ""
    if use_containers and base_override:
        eval_run_prefix = _eval_srun_prefix(
            config.benchmarks[0].name if config.benchmarks else "_report", base_override
        )
        # Merge/report need access to the PARENT output dir (all shards),
        # not just a single shard directory.
        _merge_mounts = list(cluster.container_mounts) + [f"{parent_output_dir}:{parent_output_dir}"]
        if cluster.mount_home:
            _merge_mounts.append("$HOME:$HOME")
        _merge_mount_flag = f"--container-mounts={','.join(_merge_mounts)} " if _merge_mounts else ""
        _merge_home_flag = "" if cluster.mount_home else "--no-container-mount-home "
        merge_run_prefix = (
            f"srun --mpi=pmix --overlap --unbuffered --nodes 1 --ntasks 1 "
            f"--container-image {base_override} "
            f"{_merge_mount_flag}{_merge_home_flag}\\\n    "
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
            run_prefix = _eval_srun_prefix(bench.name, eval_image)

        if _needs_full_config(bench):
            sidecar = _extract_bench_config(
                config,
                i - 1,
                svc_url_var=model_url_var,
                svc_model_var=model_id_var,
            )
            sidecar_configs[safe_name] = sidecar
            config_extra_flags = "--resume " if (cluster.auto_resume or is_shard_script) else ""
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
            if cluster.auto_resume or is_shard_script:
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

    if is_shard_script:
        report_lines = []
        ext_map = {
            "markdown": "md",
            "html": "html",
            "csv": "csv",
            "json": "json",
            "latex": "tex",
        }
        for fmt in config.output.report:
            ext = ext_map.get(fmt, fmt)
            report_lines.append(
                f'{merge_run_prefix}nel eval report "$_PARENT_DIR" -f {fmt} -o "$_PARENT_DIR/report.{ext}"'
            )
        parts.append(
            _SHARD_FINISH.format(
                shard_idx=shard_idx,
                total_shards=total_shards,
                parent_output_dir=parent_output_dir,
                merge_prefix=merge_run_prefix,
                report_commands="\n            ".join(report_lines) if report_lines else "",
            )
        )
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
            kill_cmds.append(f'    [ -n "${{{upper}_PID:-}}" ] && kill ${upper}_PID 2>/dev/null || true')
    for name, svc in config.services.items():
        if not isinstance(svc, ExternalApiService):
            upper = _safe(name).upper()
            kill_cmds.append(f'    [ -n "${{{upper}_PID:-}}" ] && wait ${upper}_PID 2>/dev/null || true')

    if cluster.auto_resume:
        max_walltime_check = ""
        if cluster.max_walltime is not None:
            max_walltime_seconds = _parse_walltime(cluster.max_walltime)
            max_walltime_check = _MAX_WALLTIME_CHECK.format(
                max_walltime_seconds=max_walltime_seconds,
                max_walltime=cluster.max_walltime,
            )
        prologue = _AUTORESUME_PROLOGUE.format(
            max_retries=cluster.max_retries,
            max_walltime_check=max_walltime_check,
        )
        parts.insert(1, prologue)

    cleanup_body = "\n".join(kill_cmds) if kill_cmds else "    echo 'No managed services.'"
    cleanup_idx = 2 if cluster.auto_resume else 1
    parts.insert(cleanup_idx, _CLEANUP_FUNC.format(kill_commands=cleanup_body))

    parts.append(_FOOTER)

    script = "\n".join(parts)
    script = re.sub(r"\n{3,}", "\n\n", script)
    return script, sidecar_configs, secrets_result


def stamp_output_dir(config: EvalConfig) -> str | None:
    """Append a timestamped run-ID subdirectory to config.output.dir.

    Called once before write_sbatch / generate_sbatch so the timestamped
    path is baked into the sbatch script and all metadata.  Skipped when
    running inside a SLURM job (NEL_INNER_EXECUTION=1) or when the user
    has opted out (output.timestamped=false).

    Returns the generated run_id so callers can reuse it (avoids a second
    call to generate_run_id producing a different timestamp).
    """
    if not config.output.timestamped or os.environ.get("NEL_INNER_EXECUTION") == "1":
        return None
    from nemo_evaluator.run_store import generate_run_id

    rid = generate_run_id(config)
    config.output.dir = str(Path(config.output.dir) / rid)
    return rid


def _write_single_script(
    out: Path,
    script: str,
    sidecar_configs: dict[str, dict],
    secrets_result: SecretsEnvResult,
    *,
    dry_run: bool = False,
) -> tuple[Path, list[Path]]:
    """Write one sbatch script + its sidecar/secrets into *out*."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    path = out / "nel_eval.sbatch"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)

    extra_paths: list[Path] = []

    if secrets_result.secrets_content.strip():
        secrets_path = out / ".secrets.env"
        secrets_path.write_text(secrets_result.secrets_content, encoding="utf-8")
        secrets_path.chmod(0o600)
        extra_paths.append(secrets_path)

        if dry_run:
            import click

            click.echo("\n.secrets.env (redacted):")
            click.echo(redact_secrets_env_content(secrets_result.secrets_content))

    for safe_name, cfg_dict in sidecar_configs.items():
        cfg_path = out / f"config_{safe_name}.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False), encoding="utf-8")
        extra_paths.append(cfg_path)

    return path, extra_paths


def write_sbatch(
    config: EvalConfig,
    output_dir: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Write sbatch script(s) + sidecar config YAMLs + .secrets.env.

    When ``cluster.shards`` is set, generates N independent per-shard
    scripts under ``shard_0/``, ``shard_1/``, etc.  Each shard is a
    standalone SLURM job with its own auto-resume chain.

    Returns ``(script_paths, extra_paths)`` where *script_paths* has one
    entry per shard (or a single entry when not sharded).
    """
    out = Path(output_dir or config.output.dir)
    cluster = config.cluster
    n_shards = getattr(cluster, "shards", None) if isinstance(cluster, SlurmCluster) else None

    if n_shards:
        script_paths: list[Path] = []
        all_extras: list[Path] = []
        for i in range(n_shards):
            script, sidecars, secrets = generate_sbatch(
                config,
                shard_idx=i,
                total_shards=n_shards,
            )
            shard_dir = out / f"shard_{i}"
            path, extras = _write_single_script(
                shard_dir,
                script,
                sidecars,
                secrets,
                dry_run=dry_run,
            )
            script_paths.append(path)
            all_extras.extend(extras)
        return script_paths, all_extras

    script, sidecar_configs, secrets_result = generate_sbatch(config)
    path, extras = _write_single_script(
        out,
        script,
        sidecar_configs,
        secrets_result,
        dry_run=dry_run,
    )
    return [path], extras
