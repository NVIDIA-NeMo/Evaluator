# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: This script is only an example of using NeMo with NeMo-Run's APIs and is subject to change without notice.
# This script is used for evaluation on local and slurm executors using NeMo-Run.
# It uses deploy method from src/nemo_eval/api.py to deploy nemo2.0 ckpt on PyTriton or Ray server and uses evaluate
# method from src/nemo_eval/api.py to run evaluation on it.
# (https://github.com/NVIDIA/NeMo-Run) to configure and execute the runs.

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import nemo_run as run
from helpers import wait_and_evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.slurm import SlurmJobDetails

logger = logging.getLogger(__name__)


def to_dict(arg: str) -> dict[str, str]:
    """Split a comma-separated KEY=VALUE string into a dictionary."""
    return dict(item.split("=", 1) for item in arg.split(",") if item)


@dataclass(kw_only=True)
class CustomJobDetailsRay(SlurmJobDetails):
    """Custom job details for Ray jobs.

    Overrides ls_term so TunnelLogIterator can discover srun log files by job ID.
    Must be set on the *main* SlurmExecutor so that _save_job_dir records it;
    setting it only on executor_eval has no effect on log tailing.
    """

    @property
    def ls_term(self) -> str:
        """This term will be used to fetch the logs.

        The command used to list the files is ls -1 {ls_term} 2> /dev/null
        %j is replaced with the Slurm job ID by TunnelLogIterator._check_finished.
        """
        assert self.folder
        # Matches srun output files: log-{job_name}_{job_id}_0.out
        return os.path.join(self.folder, "ray-job.log")


ENDPOINT_TYPES = {"chat": "chat/completions/", "completions": "completions/"}
# [snippet-deploy-start]
TRITON_DEPLOY_SCRIPT = """
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py \
  --megatron_checkpoint {megatron_checkpoint} \
  --triton_model_name megatron_model \
  --server_address {server_address} \
  --server_port {server_port} \
  --num_gpus {devices} \
  --num_nodes {nodes} \
  --tensor_model_parallel_size {tensor_model_parallel_size} \
  --pipeline_model_parallel_size {pipeline_model_parallel_size} \
  --expert_model_parallel_size {expert_model_parallel_size} \
  --max_batch_size {max_batch_size} \
  --inference_max_seq_length {inference_max_seq_length} \
  {additional_args}
"""

RAY_DEPLOY_SCRIPT = """
# Run deploy in background so the wrapper can monitor for eval completion
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint {megatron_checkpoint} \
  --model_id megatron_model \
  --port {server_port} \
  --host {server_address} \
  --num_gpus {devices} \
  --tensor_model_parallel_size {tensor_model_parallel_size} \
  --pipeline_model_parallel_size {pipeline_model_parallel_size} \
  --expert_model_parallel_size {expert_model_parallel_size} \
  --max_batch_size {max_batch_size} \
  --num_replicas {num_replicas} \
  --inference_max_seq_length {inference_max_seq_length} \
  {additional_args} &
# [snippet-deploy-end]
DEPLOY_PID=$!

# Poll for eval completion (success or failure) while deploy is running. This is required since deployment is still alive
# even after evaluation is finished or evaluation fails.
while kill -0 $DEPLOY_PID 2>/dev/null; do
    if [ -f "${{LOG_DIR:-/tmp}}/EVAL_DONE" ]; then
        kill $DEPLOY_PID 2>/dev/null || true
        wait $DEPLOY_PID 2>/dev/null || true
        if [ -f "${{LOG_DIR:-/tmp}}/EVAL_SUCCESS" ]; then
            echo "[INFO] Evaluation succeeded. Deploy stopped. Exiting cleanly."
            exit 0
        else
            echo "[ERROR] Evaluation failed. Deploy stopped."
            exit 1
        fi
    fi
    sleep 5
done

# Deploy exited on its own (before eval finished) -- propagate its exit code
wait $DEPLOY_PID 2>/dev/null
exit $?
"""


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Evaluation")
    parser.add_argument(
        "--megatron_checkpoint",
        type=str,
        required=True,
        help="Megatron-Bridge checkpoint to be evaluated",
    )
    parser.add_argument(
        "--serving_backend",
        type=str,
        default="pytriton",
        help="Serving backend to be used",
        choices=["pytriton", "ray"],
    )
    parser.add_argument(
        "--server_port", type=int, default=8080, help="Port for FastAPI or Ray server"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0",
        help="IP address for FastAPI or Ray server",
    )
    parser.add_argument(
        "--triton_address",
        type=str,
        default="0.0.0.0",
        help="IP address for Triton server",
    )
    parser.add_argument(
        "--triton_port", type=int, default=8000, help="Port for Triton server"
    )
    parser.add_argument(
        "--num_replicas", type=int, default=1, help="Num of replicas for Ray server"
    )
    parser.add_argument(
        "--num_cpus_per_replica",
        type=int,
        default=None,
        help="Num of CPUs per replica for Ray server",
    )
    parser.add_argument(
        "--endpoint_type",
        type=str,
        default="completions",
        help="Whether to use completions or chat endpoint. Refer to the docs for details on tasks that are completions"
        "v/s chat.",
        choices=list(ENDPOINT_TYPES),
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=4096,
        help="Max input length of the model",
    )
    parser.add_argument(
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="Tensor parallelism size to deploy the model",
    )
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Pipeline parallelism size to deploy the model",
    )
    parser.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=1,
        help="Expert parallelism size to deploy the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for deployment and evaluation",
    )
    parser.add_argument(
        "--additional_args",
        type=str,
        default="",
        help="Additional arguments to pass to the deployment script. Refer to the deploy script for more details.",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="mmlu",
        help="Evaluation benchmark to run. Refer to the docs for more details on the tasks/benchmarks.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit evaluation to `limit` samples. Default: use all samples.",
    )
    parser.add_argument(
        "--parallel_requests",
        type=int,
        default=8,
        help="Number of parallel requests to send to server. Default: use default for the task.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer. Default: None",
    )
    parser.add_argument(
        "--tokenizer_backend",
        type=str,
        default="huggingface",
        help="Backend to use for the tokenizer. Default: huggingface",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=1000,
        help="Time in seconds for the eval client. Default: 1000s",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for your experiment title which will be appended after the model/exp name.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--evaluation_result_dir",
        type=str,
        default="/results/",
        help="Directory to store evaluation results.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dryrun and exit",
        default=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
        default=False,
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Num nodes for the executor"
    )
    parser.add_argument(
        "--devices", type=int, default=8, help="Num devices per node for the executor"
    )
    parser.add_argument(
        "--container_image",
        type=str,
        default="nvcr.io/nvidia/nemo:25.07",
        help="Container image for the run, only used in case of slurm runs."
        "Can be a path as well in case of .sqsh file.",
    )
    slurm_args = parser.add_argument_group("Slurm arguments")
    slurm_args.add_argument(
        "--account",
        type=str,
        default=os.environ.get("ACCOUNT", ""),
        help="Slurm account to use for experiment. Defaults to $ACCOUNT env var.",
    )
    slurm_args.add_argument(
        "--partition",
        type=str,
        default=os.environ.get("PARTITION", ""),
        help="Slurm partition to use for experiment. Defaults to $PARTITION env var.",
    )
    slurm_args.add_argument(
        "--time_limit",
        type=str,
        default="04:00:00",
        help="Maximum time limit for the job (format: HH:MM:SS). Default: 04:00:00",
    )
    slurm_args.add_argument(
        "--job_dir",
        type=str,
        default=os.environ.get("NEMORUN_HOME", ""),
        help="Directory for job logs and artifacts on the cluster. Defaults to $NEMORUN_HOME env var.",
    )
    slurm_args.add_argument(
        "--ssh_host",
        type=str,
        default=os.environ.get("SSH_HOST", ""),
        help="SSH host for SSHTunnel (default executor). Defaults to $SSH_HOST env var.",
    )
    slurm_args.add_argument(
        "--ssh_user",
        type=str,
        default=os.environ.get("SSH_USER", ""),
        help="SSH user for SSHTunnel (default executor). Defaults to $SSH_USER env var.",
    )
    slurm_args.add_argument(
        "--local_tunnel",
        action="store_true",
        default=False,
        help="Use LocalTunnel instead of SSHTunnel. Useful when submitting from the cluster head node.",
    )
    slurm_args.add_argument(
        "--custom_mounts",
        type=str,
        default=os.environ.get("CUSTOM_MOUNTS", ""),
        help="Comma-separated list of mounts (src:dst). Defaults to $CUSTOM_MOUNTS env var.",
    )
    slurm_args.add_argument(
        "--custom_env_vars",
        type=to_dict,
        default=to_dict(os.environ.get("CUSTOM_ENV_VARS", "")),
        help="Comma-separated KEY=VALUE pairs of extra environment variables (e.g. 'FOO=bar,BAZ=1'). "
        "Defaults to $CUSTOM_ENV_VARS env var.",
    )
    return parser


def slurm_executor(
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    container_image: str,
    job_dir: str,
    ssh_host: str = "",
    ssh_user: str = "",
    local_tunnel: bool = False,
    time: str = "04:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (account and partition and nodes and devices):
        raise RuntimeError(
            "Please set account, partition, nodes and devices args for using this function."
        )
    if not local_tunnel and not (ssh_host and ssh_user):
        raise RuntimeError(
            "Please set --ssh_host and --ssh_user (or $SSH_HOST / $SSH_USER) for SSHTunnel, "
            "or pass --local_tunnel to submit from the cluster head node."
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    # [snippet-slurm-executor-start]
    env_vars = {
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # Recommended to use this over run.Packager() as it can lead to fiddle serialization errors importing
    # 'wait_and_evaluate' method from 'helpers'
    packager = run.Config(run.GitArchivePackager, subpath="scripts")

    if local_tunnel:
        # Do not pass job_dir to LocalTunnel: when job_dir is given, nemo_run sets
        # tunnel.job_dir = job_dir/title/exp_id but writes the sbatch script to
        # NEMORUN_HOME/experiments/title/exp_id, causing a path mismatch that
        # prevents sbatch from finding the script.  With no job_dir, both paths
        # resolve through NEMORUN_HOME/experiments/... and stay consistent.
        tunnel = run.LocalTunnel(
            job_dir=os.path.join(get_nemorun_home(), "experiments")
        )
    else:
        tunnel = run.SSHTunnel(
            user=ssh_user,
            host=ssh_host,
            job_dir=job_dir,
        )

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=tunnel,
        nodes=nodes,
        ntasks_per_node=devices,
        exclusive=True,
        # archives and uses the local code. Use packager=run.Packager() to use the code code mounted on clusters
        packager=packager,
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time
    # [snippet-slurm-executor-end]

    return executor


def local_executor_torchrun() -> run.LocalExecutor:
    # [snippet-local-executor-start]
    env_vars = {
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "HF_TOKEN": "xxxxxx",  # [hf-token-local]
    }

    executor = run.LocalExecutor(env_vars=env_vars)
    # [snippet-local-executor-end]
    return executor


def main():
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    additional_args = args.additional_args
    commons_args = {
        "megatron_checkpoint": args.megatron_checkpoint,
        "server_port": args.server_port,
        "server_address": args.server_address,
        "max_input_len": args.max_input_len,
        "tensor_model_parallel_size": args.tensor_parallelism_size,
        "pipeline_model_parallel_size": args.pipeline_parallelism_size,
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "max_batch_size": args.batch_size,
        "inference_max_seq_length": args.max_input_len,
        "devices": args.devices
        if args.serving_backend == "pytriton"
        else args.devices * args.nodes,
        "nodes": args.nodes,
        "num_replicas": args.num_replicas,
    }

    exp_name = "NeMoEvaluation"
    if args.serving_backend == "pytriton":
        additional_args += (
            f" --triton_port {args.triton_port}"
            f" --triton_http_address {args.triton_address}"
        )
        deploy_script = TRITON_DEPLOY_SCRIPT.format(
            **commons_args, additional_args=additional_args
        )
        deploy_run_script = run.Script(inline=deploy_script)

    elif args.serving_backend == "ray":
        # Ray deployment with nemo-run requires python executable path to be set, else it cannot find the libraries
        # like mcore, mbridge, etc.
        additional_args += (
            ' --runtime_env \'{"py_executable": "/opt/venv/bin/python"}\''
        )
        if args.num_cpus_per_replica:
            additional_args += f" --num_cpus_per_replica {args.num_cpus_per_replica}"
        deploy_script = RAY_DEPLOY_SCRIPT.format(
            **commons_args, additional_args=additional_args
        )
        deploy_run_script = run.Script(
            inline=deploy_script, metadata={"use_with_ray_cluster": True}
        )
    else:
        raise ValueError(f"Invalid serving backend: {args.serving_backend}")

    print(deploy_script)
    # [snippet-config-start]
    api_endpoint = run.Config(
        ApiEndpoint,
        url=f"http://{args.server_address}:{args.server_port}/v1/{ENDPOINT_TYPES[args.endpoint_type]}",
        type=args.endpoint_type,
        model_id="megatron_model",
    )
    eval_target = run.Config(EvaluationTarget, api_endpoint=api_endpoint)
    extra = {}
    if args.tokenizer_path:
        extra["tokenizer"] = args.tokenizer_path
        extra["tokenizer_backend"] = args.tokenizer_backend
    eval_params = run.Config(
        ConfigParams,
        limit_samples=args.limit,
        parallelism=args.parallel_requests,
        request_timeout=args.request_timeout,
        extra=extra,
    )
    eval_config = run.Config(
        EvaluationConfig,
        type=args.eval_task,
        params=eval_params,
        output_dir=args.evaluation_result_dir,
    )

    eval_fn = run.Partial(
        wait_and_evaluate,
        target_cfg=eval_target,
        eval_cfg=eval_config,
        serving_backend=args.serving_backend,
    )
    # [snippet-config-end]

    executor: run.Executor
    executor_eval: run.Executor
    if args.slurm:
        custom_mounts = (
            [m for m in args.custom_mounts.split(",") if m]
            if args.custom_mounts
            else []
        )
        executor = slurm_executor(
            account=args.account,
            partition=args.partition,
            nodes=args.nodes,
            devices=args.devices,
            container_image=args.container_image,
            job_dir=args.job_dir,
            ssh_host=args.ssh_host,
            ssh_user=args.ssh_user,
            local_tunnel=args.local_tunnel,
            time=args.time_limit,
            custom_mounts=custom_mounts,
            custom_env_vars=args.custom_env_vars or None,
        )
        executor.srun_args = ["--mpi=pmix", "--overlap"]
        executor_eval = executor.clone()
        executor_eval.srun_args = [
            "--ntasks-per-node=1",
            "--nodes=1",
        ]  ## so that eval is laucnhed only on main node
        # or node with index 0
        # Set on main executor so _save_job_dir records the correct ls_term for log tailing.
        # executor_eval is a ResourceRequest after merge and its ls_term is not used.
        executor.job_details = CustomJobDetailsRay()
    else:
        executor = local_executor_torchrun()
        executor_eval = None
    # [snippet-experiment-start]
    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        if args.slurm:
            exp.add(
                [deploy_run_script, eval_fn],
                executor=[executor, executor_eval],
                name=exp_name,
                tail_logs=True,
            )
        else:
            exp.add(
                deploy_run_script,
                executor=executor,
                name=f"{exp_name}_deploy",
                tail_logs=True,
            )
            exp.add(
                eval_fn, executor=executor, name=f"{exp_name}_evaluate", tail_logs=True
            )

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run()
    # [snippet-experiment-end]

    if not args.dryrun:
        # Status check must happen after __exit__ which waits for Slurm jobs to finish
        status_dict = exp.status(return_dict=True) or {}
        failed = [
            f"{name}: {info.get('status', 'UNKNOWN')}"
            for name, info in status_dict.items()
            if str(info.get("status")) != "SUCCEEDED" and "deploy" not in name.lower()
        ]
        if failed:
            logger.error(f"Experiment finished with failures: {failed}")
            sys.exit(1)


if __name__ == "__main__":
    main()
