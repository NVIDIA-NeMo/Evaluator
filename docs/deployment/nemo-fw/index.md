(deployment-nemo-fw)=
# Deploy and Evaluate Checkpoints Trained by NeMo Framework

The NeMo Framework is NVIDIA’s GPU-accelerated, end-to-end training platform for large language models (LLMs), multimodal models, and speech models. It enables seamless scaling of both pretraining and post-training workloads, from a single GPU to clusters with thousands of nodes, supporting Hugging Face/PyTorch and Megatron models. NeMo includes a suite of libraries and curated training recipes to help users build models from start to finish.

The NeMo Evaluator is integrated within NeMo Framework, offering streamlined deployment and advanced evaluation capabilities for models trained using NeMo, leveraging state-of-the-art evaluation harnesses.

## Features

- **Multi-Backend Deployment**: Supports PyTriton and multi-instance evaluations using the Ray Serve deployment backend
- **Production-Ready**: Supports high-performance inference with CUDA graphs and flash decoding
- **Multi-GPU and Multi-Node Support**: Enables distributed inference across multiple GPUs and compute nodes
- **OpenAI-Compatible API**: Provides RESTful endpoints aligned with OpenAI API specifications

## Architecture

### 1. Deployment Layer

- **PyTriton Backend**: Provides high-performance inference through the NVIDIA Triton Inference Server, with OpenAI API compatibility via a FastAPI interface. Supports model parallelism across single-node and multi-node configurations. Note: Multi-instance evaluation is not supported.
- **Ray Backend**: Enables multi-instance evaluation with model parallelism on a single node using Ray Serve, while maintaining OpenAI API compatibility. Multi-node support is coming soon.

For more information on the deployment, please see [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy).

### 2. Evaluation Layer

- **NeMo Evaluator**: Provides standardized benchmark evaluations using packages from NVIDIA Eval Factory, bundled in the NeMo Framework container. The `lm-evaluation-harness` is pre-installed by default, and additional evaluation packages can be added as needed. For more information, see {ref}`core-wheels` and {ref}`lib-core`.



```{toctree}
:maxdepth: 1
:hidden:

Introduction <self>
PyTriton Serving Backend <pytriton>
Ray Serving Backend <ray>
Evaluate Megatron Bridge Checkpoints <mbridge>
Evaluate Automodel Checkpoints <hf>
```