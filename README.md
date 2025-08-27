<div align="center">

# NeMo Eval

[![codecov](https://codecov.io/github/NVIDIA-NeMo/Eval/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Eval)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Eval/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Eval/actions/workflows/cicd-main.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/NVIDIA-NeMo/Eval/blob/main/pyproject.toml)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-NeMo-red.svg)](https://github.com/NVIDIA-NeMo/)

[Documentation](https://docs.nvidia.com/nemo/eval/latest/index.html) | [Examples](https://github.com/NVIDIA-NeMo/Eval?tab=readme-ov-file#-usage-examples) | [Contributing](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md)
</div>

## Overview

The NeMo Framework is NVIDIA’s GPU-accelerated, end-to-end training platform for large language models (LLMs), multimodal models, and speech models. It enables seamless scaling of both pretraining and post-training workloads, from a single GPU to clusters with thousands of nodes, supporting Hugging Face/PyTorch and Megatron models. NeMo includes a suite of libraries and curated training recipes to help users build models from start to finish.

The Eval library ("NeMo Eval") is a comprehensive evaluation module within the NeMo Framework for LLMs. It offers streamlined deployment and advanced evaluation capabilities for models trained using NeMo, leveraging state-of-the-art evaluation harnesses.

![image](./NeMo_Repo_Overview_Eval.png)

## 🚀 Features

- **Multi-Backend Deployment**: Supports PyTriton and multi-instance evaluations using the Ray Serve deployment backend
- **Comprehensive Evaluation**: Includes state-of-the-art evaluation harnesses for academic benchmarks, reasoning benchmarks, code generation, and safety testing
- **Adapter System**: Features a flexible architecture with chained interceptors for customizable request and response processing
- **Production-Ready**: Supports high-performance inference with CUDA graphs and flash decoding
- **Multi-GPU and Multi-Node Support**: Enables distributed inference across multiple GPUs and compute nodes
- **OpenAI-Compatible API**: Provides RESTful endpoints aligned with OpenAI API specifications

## 🔧 Install NeMo Eval

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU(s) (tested on RTX A6000, A100, H100)
- NeMo Framework container (recommended)

#### Recommended Requirements

- Python 3.12
- PyTorch 2.7
- CUDA 12.9
- Ubuntu 24.04

### Use pip

For quick exploration of NeMo Eval, we recommend installing our pip package:

```bash
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
pip install --no-build-isolation nemo-eval
```

### Use Docker

For optimal performance and user experience, use the latest version of the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Please fetch the most recent $TAG and run the following command to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

### Use uv

To install NeMo Eval with uv, please refer to our [Contribution guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md).

## 🚀 Quick Start

### 1. Deploy a Model

```python
from nemo_eval.api import deploy

# Deploy a NeMo checkpoint
deploy(
    nemo_checkpoint="/path/to/your/checkpoint",
    serving_backend="pytriton",  # or "ray"
    server_port=8080,
    num_gpus=1,
    max_input_len=4096,
    max_batch_size=8
)
```

### 2. Evaluate the Model

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
print(results)
```

## 📊 Support Matrix

| Checkpoint Type | Inference Backend | Deployment Server | Evaluation Harnesses Supported |
|----------------|-------------------|-------------|--------------------------|
|         NeMo FW checkpoint via Megatron Core backend         |    [Megatron Core in-framework inference engine](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/inference)               |     PyTriton (single and multi node model parallelism), Ray (single node model parallelism with multi instance evals)        |          lm-evaluation-harness, simple-evals, BigCode, BFCL, safety-harness, garak                |

## 🏗️ Architecture

### Core Components

#### 1. Deployment Layer

- **PyTriton Backend**: Provides high-performance inference through the NVIDIA Triton Inference Server, with OpenAI API compatibility via a FastAPI interface. Supports model parallelism across single-node and multi-node configurations. Note: Multi-instance evaluation is not supported.
- **Ray Backend**: Enables multi-instance evaluation with model parallelism on a single node using Ray Serve, while maintaining OpenAI API compatibility. Multi-node support is coming soon.

#### 2. Evaluation Layer

- **NVIDIA Eval Factory**: Provides standardized benchmark evaluations using packages from NVIDIA Eval Factory, bundled in the NeMo Framework container. The `lm-evaluation-harness` is pre-installed by default, and additional tools listed in the [support matrix](#-support-matrix) can be added as needed. For more information, see the [documentation](https://github.com/NVIDIA-NeMo/Eval/tree/main/docs).

- **Adapter System**: Flexible request/response processing pipeline with **Interceptors** that provide modular processing:
  - **Available Interceptors**: Modular components for request/response processing
    - **SystemMessageInterceptor**: Customize system prompts
    - **RequestLoggingInterceptor**: Log incoming requests
    - **ResponseLoggingInterceptor**: Log outgoing responses
    - **ResponseReasoningInterceptor**: Process reasoning outputs
    - **EndpointInterceptor**: Route requests to the actual model

## 📖 Usage Examples

### Basic Deployment with PyTriton as the Serving Backend

```python
from nemo_eval.api import deploy

# Deploy model
deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="pytriton",
    server_port=8080,
    num_gpus=1,
    max_input_len=8192,
    max_batch_size=4
)
```

### Basic Evaluation

```Python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
# Configure Endpoint
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    model_id="megatron_model"
)
# Evaluation target configuration
target = EvaluationTarget(api_endpoint=api_endpoint)
# Configure EvaluationConfig with type, number of samples to evaluate on, etc.
config = EvaluationConfig(type="gsm8k",
            output_dir="results",
            params=ConfigParams(
                    limit_samples=10
                ))

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
```

### Use Adapters

The example below demonstrates how to configure an Adapter to provide a custom system prompt. Requests and responses are processed through interceptors, which are automatically selected based on the parameters defined in `AdapterConfig`.

```python
from nemo_eval.utils.api import AdapterConfig

# Configure adapter for reasoning
adapter_config = AdapterConfig(
    api_url="http://0.0.0.0:8080/v1/completions/",
    use_reasoning=True,
    end_reasoning_token="</think>",
    custom_system_prompt="You are a helpful assistant that thinks step by step.",
    max_logged_requests=5,
    max_logged_responses=5
)

# Run evaluation with adapter
results = evaluate(
    target_cfg=target,
    eval_cfg=config,
    adapter_cfg=adapter_config
)
```

### Deploy with Multiple GPUs

```python
# Deploy with tensor parallelism or pipeline parallelism
deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="pytriton",
    num_gpus=4,
    tensor_parallelism_size=4,
    pipeline_parallelism_size=1,
    max_input_len=8192,
    max_batch_size=8
)
```

### Deploy with Ray

```python
# Deploy using Ray Serve
deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="ray",
    num_gpus=2,
    num_replicas=2,
    num_cpus_per_replica=8,
    server_port=8080,
    include_dashboard=True,
    cuda_visible_devices="0,1"
)
```

## 📁 Project Structure

```
Eval/
├── src/nemo_eval/           # Main package
│   ├── api.py               # Main API functions
│   ├── package_info.py      # Package metadata
│   ├── adapters/            # Adapter system
│   │   ├── server.py        # Adapter server
│   │   ├── utils.py         # Adapter utilities
│   │   └── interceptors/    # Request/response interceptors
│   └── utils/               # Utility modules
│       ├── api.py           # API configuration classes
│       ├── base.py          # Base utilities
│       └── ray_deploy.py    # Ray deployment utilities
├── tests/                   # Test suite
│   ├── unit_tests/          # Unit tests
│   └── functional_tests/    # Functional tests
├── tutorials/               # Tutorial notebooks
├── scripts/                 # Reference nemo-run scripts
├── docs/                    # Documentation
├── docker/                  # Docker configuration
└── external/                # External dependencies
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md) for details on development setup, testing, and code style guidelines

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/NVIDIA-NeMo/Eval/blob/main/LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)
- **Documentation**: [NeMo Documentation](https://nemo-framework-documentation.gitlab-master-pages.nvidia.com/eval-build/)

## 🔗 Related Projects

- [NeMo Export Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) - Model export and deployment

---

**Note**: This project is actively maintained by NVIDIA. For the latest updates and features, please check our [releases page](https://github.com/NVIDIA-NeMo/Eval/releases).
