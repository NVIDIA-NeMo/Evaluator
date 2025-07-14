# NeMo Eval

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NVIDIA-NeMo/Eval/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/NVIDIA-NeMo/Eval/blob/main/pyproject.toml)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-NeMo-red.svg)](https://github.com/NVIDIA-NeMo/)

**NeMo Eval** is a comprehensive evaluation module under Nemo Framework for Large Language Models (LLMs). It provides seamless deployment and evaluation capabilities for models trained using Nemo Framework via state-of-the-art evaluation harnesses.

## 🚀 Features

- **Multi-Backend Deployment**: Support for both PyTriton and Ray Serve deployment backends
- **Comprehensive evaluation**: State-of-the-art evaluation harnesses including reasoning benchmarks, code generation, safety testing
- **Adapter System**: Flexible adapter architecture using a chain of interceptors for customizing request/response processing
- **Production Ready**: Optimized for high-performance inference with CUDA graphs and flash decoding
- **Multi-GPU & Multi-Node Support**: Distributed inference across multiple devices and nodes
- **OpenAI-Compatible API**: RESTful endpoints compatible with OpenAI API standards

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Support Matrix](#-support-matrix)
- [Architecture](#-architecture)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Support](#-support)
- [Related Projects](#-related-projects)

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU(s)  (tested on RTX A6000, A100, H100)
- NeMo Framework container (recommended)

### Using pip

```bash
pip install nemo-eval
```

### Using uv

```bash
# Install uv if you haven't already
pip install uv

# Install nemo-eval
uv pip install nemo-eval
```

### From Source

```bash
git clone https://github.com/NVIDIA-NeMo/Eval.git
cd Eval
pip install -e .
```

### From Source with uv

```bash
git clone https://github.com/NVIDIA-NeMo/Eval.git
cd Eval
uv sync
```

### Using Docker

The recommended approach is to use the NeMo Framework container:

```bash
docker pull nvcr.io/nvidia/nemo:25.07
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/nemo:25.07
```

## 🚀 Quick Start

### 1. Deploy a Model

```python
from nemo_eval.api import deploy

# Deploy a NeMo checkpoint
deploy(
    nemo_checkpoint="/path/to/your/checkpoint",
    serving_backend="pytriton",  # or "ray"
    server_port=8000,
    fastapi_port=8080,
    num_gpus=1,
    max_input_len=4096,
    max_batch_size=8
)
```

### 2. Evaluate the Model

```python
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationTarget, EvaluationConfig, ApiEndpoint

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k")

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
print(results)
```

## 📊 Support Matrix

| Checkpoint Type | Inference Backend | Deployment Server | Evaluation Harnesses Supported |
|----------------|-------------------|-------------|--------------------------|
|         NeMo FW checkpoint via megatron-core backend         |    Megatron Core in-framework inference engine               |     PyTriton (single and multi node model parallelism), Ray (single node model parallelism with multi instance evals)        |          lm-evaluation-harness, simple-evals, BigCode, BFCL, safety-harness, garak                |

## 🏗️ Architecture

### Core Components

#### 1. Deployment Layer

- **PyTriton Backend**: High-performance inference using NVIDIA Triton Inference Server and OpenAI API compatibility via FastAPI Interface with model parallelism across single and multi node. Does not support multi instance evaluation.
- **Ray Backend**: Single node model parallel multi instance evaluation using Ray Serve with OpenAI API compatibility. Multi node support coming soon.

#### 2. Evaluation Layer

- **NVIDIA Eval Factory**: Standardized benchmark evaluation with eval packages from NVIDIA Eval Factory that are installed in the NeMo Framework container. lm-evaluation-harness is installed inside the NeMo Framework container by default while the rest from the [support matrix](#-support-matrix) can be installed on-demand. More details in the [docs](https://github.com/NVIDIA-NeMo/Eval/tree/main/docs).

- **Adapter System**: Flexible request/response processing pipeline with **Interceptors** that provide modular processing

```
┌───────────────────────┐
│  NVIDIA Eval Factory  │
└───▲──────┬────────────┘
    │      │
    │      │
┌───┼──────┼──────────────────────────────────────────────────┐
│   │      ▼                                                  │
│ AdapterServer (@ localhost:<free port>)                     │
│                                                             │
│   ▲      │       chain of RequestInterceptors:              │
│   │      │       flask.Request                              │
│   │      │       is passed on the way up                    │
│   │      │                                                  │
│   │ ┌────▼───────────────────────────────────────────────┐  │
│   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►  │
│   │ └─────────────────────┼──────────────────────────────┘  │   │
│   │                       │                                 │   │
│   │                       └─────────────┐                   │   │
│   │                                     │                   │   │
│ ┌─┼─────────────────────────────────────▼────┐              │   │
│ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤  │
│ └────────────────────────────────────────────┘              │   │
│                                                             │   │
│              Chain of ResponseInterceptors:                 │   │
│              requests.Response is passed on the way down    │   │
│                                                             │   │
└─────────────────────────────────────────────────────────────┘
```

- **Available Interceptors**: Modular components for request/response processing
  - **SystemMessageInterceptor**: Customize system prompts
  - **RequestLoggingInterceptor**: Log incoming requests
  - **ResponseLoggingInterceptor**: Log outgoing responses
  - **ResponseReasoningInterceptor**: Process reasoning outputs
  - **EndpointInterceptor**: Route requests to the actual model

## 📖 Usage Examples

### Basic Deployment with PyTriton as the serving backend

```python
import logging
from nemo_eval.api import deploy

# Set up logging
logging.getLogger().setLevel(logging.INFO)

# Deploy model
deploy(
    nemo_checkpoint="/path/to/checkpoint",
    serving_backend="pytriton",
    server_port=8000,
    fastapi_port=8080,
    num_gpus=1,
    max_input_len=8192,
    max_batch_size=4
)
```

### Basic Evaluation

```Python
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationTarget, EvaluationConfig, ApiEndpoint, ConfigParams  
# Configure Endpoint
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
)
# Evaluation target configuration
target = EvaluationTarget(api_endpoint=api_endpoint)
# Configure EvaluationConfig with type, num of samples to evaluate on etc.,
config = EvaluationConfig(type="gsm8k",
            params=ConfigParams(
                    limit_samples=10
                ))

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
```

### Using Adapters

The example below shows how to configure an Adapter that allows to provide a custom system prompt. Requests/responses are processed through interceptors. Interceptors are automatically selected based on the `AdapterConfig` parameters you provide.

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

### Multi-GPU Deployment

```python
# Deploy with tensor parallelism or pipleline parallelism
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
    server_port=8000,
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
- **Documentation**: [NeMo Documentation](https://github.com/NVIDIA-NeMo/Eval/tree/main/docs)

## 🔗 Related Projects

- [NeMo Export Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) - Model export and deployment

---

**Note**: This project is actively maintained by NVIDIA. For the latest updates and features, please check our [releases page](https://github.com/NVIDIA-NeMo/Eval/releases).
