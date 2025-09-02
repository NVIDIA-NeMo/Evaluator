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
from typing import Optional

import nemo_run as run
from nemo_eval.api import deploy
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.core.evaluate import evaluate

ENDPOINT_TYPES = {"chat": "chat/completions/", "completions": "completions/"}


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Evaluation")
    parser.add_argument(
        "--nemo_checkpoint",
        type=str,
        required=True,
        help="NeMo 2.0 checkpoint to be evaluated",
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
        default=1,
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
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for deployment and evaluation",
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
    return parser


def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    container_image: str,
    time: str = "04:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (
        user and host and remote_job_dir and account and partition and nodes and devices
    ):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this ",
            "function.",
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "HF_TOKEN": "xxxxxx",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        exclusive=True,
        # archives and uses the local code. Use packager=run.Packager() to use the code code mounted on clusters
        packager=run.GitArchivePackager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def local_executor_torchrun() -> run.LocalExecutor:
    env_vars = {
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "HF_TOKEN": "xxxxxx",
    }

    executor = run.LocalExecutor(env_vars=env_vars)

    return executor


def main():
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    exp_name = "NeMoEvaluation"
    deploy_fn = run.Partial(
        deploy,
        nemo_checkpoint=args.nemo_checkpoint,
        serving_backend=args.serving_backend,
        server_port=args.server_port,
        server_address=args.server_address,
        triton_address=args.triton_address,
        triton_port=args.triton_port,
        num_replicas=args.num_replicas,
        num_cpus_per_replica=args.num_cpus_per_replica,
        max_input_len=args.max_input_len,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        max_batch_size=args.batch_size,
        num_gpus=args.devices,
        num_nodes=args.nodes,
    )

    api_endpoint = run.Config(
        ApiEndpoint,
        url=f"http://{args.server_address}:{args.server_port}/v1/{ENDPOINT_TYPES[args.endpoint_type]}",
        type=args.endpoint_type,
    )
    eval_target = run.Config(EvaluationTarget, api_endpoint=api_endpoint)
    eval_params = run.Config(
        ConfigParams,
        limit_samples=args.limit,
        parallelism=args.parallel_requests,
        request_timeout=args.request_timeout,
    )
    eval_config = run.Config(EvaluationConfig, type=args.eval_task, params=eval_params)

    eval_fn = run.Partial(evaluate, target_cfg=eval_target, eval_cfg=eval_config)

    executor: run.Executor
    executor_eval: run.Executor
    if args.slurm:
        # TODO: Set your custom parameters for the Slurm Executor.
        executor = slurm_executor(
            user="",
            host="",
            remote_job_dir="",
            account="",
            partition="",
            nodes=args.nodes,
            devices=args.devices,
            container_image=args.container_image,
            custom_mounts=[],
        )
        executor.srun_args = ["--mpi=pmix", "--overlap"]
        executor_eval = executor.clone()
        executor_eval.srun_args = [
            "--ntasks-per-node=1",
            "--nodes=1",
        ]  ## so that eval is laucnhed only on main node
        # or node with index 0
    else:
        executor = local_executor_torchrun()
        executor_eval = None

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        if args.slurm:
            exp.add(
                [deploy_fn, eval_fn],
                executor=[executor, executor_eval],
                name=exp_name,
                tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
            )
        else:
            exp.add(deploy_fn, executor=executor, name=f"{exp_name}_deploy")
            exp.add(eval_fn, executor=executor, name=f"{exp_name}_evaluate")

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run()


if __name__ == "__main__":
    main()
