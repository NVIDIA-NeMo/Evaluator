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

import logging
from pathlib import Path
from typing import Optional, Union

AnyPath = Union[Path, str]

logger = logging.getLogger(__name__)


def deploy(
    nemo_checkpoint: Optional[AnyPath] = None,
    hf_model_id_path: Optional[AnyPath] = None,
    serving_backend: str = "pytriton",
    model_name: str = "megatron_model",
    server_port: int = 8080,
    server_address: str = "0.0.0.0",
    triton_address: str = "0.0.0.0",
    triton_port: int = 8000,
    num_gpus: int = 1,
    num_nodes: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    max_input_len: int = 4096,
    max_batch_size: int = 8,
    # Specific to nemo checkpoint
    enable_flash_decode: bool = True,
    enable_cuda_graphs: bool = True,
    legacy_ckpt: bool = False,
    # Specific to huggingface checkpoint
    use_vllm_backend: bool = True,
    # Ray deployment specific args
    num_replicas: int = 1,
    num_cpus: Optional[int] = None,
    include_dashboard: bool = True,
):
    """
    Deploys nemo model on either PyTriton server or Ray Serve.

    Args:
        nemo_checkpoint (Path): Path for nemo checkpoint.
        hf_model_id_path (Path): Huggingface model id or local path to the model. Supported only for Ray backend.
        serving_backend (str): Backend to use for serving ("pytriton" or "ray"). Default: "pytriton".
        model_name (str): Name for the model that gets deployed on PyTriton or Ray.
        server_port (int): HTTP port for the FastAPI or Ray server. Default: 8080.
        server_address (str): HTTP address for the FastAPI or Ray server. Default: "0.0.0.0".
        triton_address (str): HTTP address for Triton server. Default: "0.0.0.0".
        triton_port (int): Port for Triton server. Default: 8000.
        num_gpus (int): Number of GPUs per node. Default: 1.
        num_nodes (int): Number of nodes. Default: 1.
        tensor_parallelism_size (int): Tensor parallelism size. Default: 1.
        pipeline_parallelism_size (int): Pipeline parallelism size. Default: 1.
        context_parallel_size (int): Context parallelism size. Default: 1.
        expert_model_parallel_size (int): Expert parallelism size. Default: 1.
        max_input_len (int): Max input length of the model. Default: 4096.
        max_batch_size (int): Max batch size of the model. Default: 8.
        ##### Specific to nemo checkpoint #####
        enable_flash_decode (bool): If True runs inferencewith flash decode enabled. Default: True. Applicable only for
        nemo checkpoint.
        enable_cuda_graphs (bool): Whether to enable CUDA graphs for inference. Default: True. Applicable only for
        nemo checkpoint.
        legacy_ckpt (bool): Indicates whether the checkpoint is in the legacy format. Default: False. Applicable only
        for nemo checkpoint.
        ##### Specific to huggingface checkpoint #####
        use_vllm_backend (bool): Whether to use VLLM backend. Default: True. Applicable only for huggingface
        checkpoint.
        ##### Ray deployment specific args #####
        num_replicas (int): Number of model replicas for Ray deployment. Default: 1. Only applicable for Ray backend.
        num_cpus (int): Number of CPUs to allocate for the Ray cluster. If None, will use all available CPUs.
        Default: None.
        include_dashboard (bool): Whether to include Ray dashboard. Default: True.
    """
    import torch

    if serving_backend == "ray":  # pragma: no cover
        from nemo_deploy.deploy_ray import DeployRay

        # Initialize Ray deployment
        ray_deployer = DeployRay(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            include_dashboard=include_dashboard,
            host=server_address,
            port=server_port,
        )
        if nemo_checkpoint is not None:
            # Deploy nemo checkpoint in-framework(via mcore inference engine) with Ray backend
            ray_deployer.deploy_inframework_model(
                nemo_checkpoint=nemo_checkpoint,
                num_gpus=num_gpus,
                tensor_model_parallel_size=tensor_parallelism_size,
                pipeline_model_parallel_size=pipeline_parallelism_size,
                expert_model_parallel_size=expert_model_parallel_size,
                context_parallel_size=context_parallel_size,
                model_id=model_name,
                num_cpus_per_replica=num_cpus,
                num_replicas=num_replicas,
                enable_cuda_graphs=enable_cuda_graphs,
                enable_flash_decode=enable_flash_decode,
                legacy_ckpt=legacy_ckpt,
                max_batch_size=max_batch_size,
            )
        elif hf_model_id_path is not None:
            # Deploy huggingface checkpoint directly or via vllm backend on Ray
            ray_deployer.deploy_huggingface_model(
                hf_model_id_path=hf_model_id_path,
                device_map="cuda",
                model_id=model_name,
                num_replicas=num_replicas,
                num_cpus_per_replica=num_cpus,
                num_gpus_per_replica=num_gpus,
                max_ongoing_requests=max_batch_size,
                use_vllm_backend=use_vllm_backend,
            )

    else:  # pytriton backend
        import os

        import uvicorn
        from nemo_deploy import DeployPyTriton

        if triton_port == server_port:
            raise ValueError(
                "FastAPI port and Triton server port cannot use the same port,"
                " but were both set to {triton_port}. Please change them"
            )

        # Store triton ip, port relevant for FastAPI as env vars
        os.environ["TRITON_HTTP_ADDRESS"] = triton_address
        os.environ["TRITON_PORT"] = str(triton_port)

        try:
            from nemo_deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2
        except Exception as e:
            raise ValueError(
                f"Unable to import MegatronLLMDeployable, due to: {type(e).__name__}: {e} cannot run "
                f"evaluation with in-framework deployment"
            )

        triton_deployable = MegatronLLMDeployableNemo2(
            nemo_checkpoint_filepath=nemo_checkpoint,
            num_devices=num_gpus,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_parallelism_size,
            pipeline_model_parallel_size=pipeline_parallelism_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            inference_max_seq_length=max_input_len,
            enable_flash_decode=enable_flash_decode,
            enable_cuda_graphs=enable_cuda_graphs,
            max_batch_size=max_batch_size,
            legacy_ckpt=legacy_ckpt,
        )

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                try:
                    nm = DeployPyTriton(
                        model=triton_deployable,
                        triton_model_name=model_name,
                        max_batch_size=max_batch_size,
                        http_port=triton_port,
                        address=triton_address,
                    )

                    logger.info("Triton deploy function will be called.")
                    nm.deploy()
                    nm.run()
                except Exception as error:
                    logger.error("Error message has occurred during deploy function. Error message: " + str(error))
                    return

                try:
                    # start fastapi server which acts as a proxy to Pytriton server. Applies to PyTriton backend only.
                    try:
                        logger.info("REST service will be started.")
                        uvicorn.run(
                            "nemo_deploy.service.fastapi_interface_to_pytriton:app",
                            host=server_address,
                            port=server_port,
                            reload=True,
                        )
                    except Exception as error:
                        logger.error(
                            "Error message has occurred during REST service start. Error message: " + str(error)
                        )
                    logger.info("Model serving on Triton will be started.")
                    nm.serve()
                except Exception as error:
                    logger.error("Error message has occurred during deploy function. Error message: " + str(error))
                    return

                logger.info("Model serving will be stopped.")
                nm.stop()
            elif torch.distributed.get_rank() > 0:
                triton_deployable.generate_other_ranks()
