---
orphan: true
---
# Evaluate checkpoints trained by NeMo Framework

The NeMo Framework is NVIDIA’s GPU-accelerated, end-to-end training platform for large language models (LLMs), multimodal models, and speech models. It enables seamless scaling of both pretraining and post-training workloads, from a single GPU to clusters with thousands of nodes, supporting Hugging Face/PyTorch and Megatron models. NeMo includes a suite of libraries and curated training recipes to help users build models from start to finish.

The NeMo Evaluator is integrated within NeMo Framework, offering streamlined deployment and advanced evaluation capabilities for models trained using NeMo, leveraging state-of-the-art evaluation harnesses.

![image](../../NeMo_Repo_Overview_Eval.png)

## ✨ Features

- **Multi-Backend Deployment**: Supports PyTriton and multi-instance evaluations using the Ray Serve deployment backend
- **Production-Ready**: Supports high-performance inference with CUDA graphs and flash decoding
- **Multi-GPU and Multi-Node Support**: Enables distributed inference across multiple GPUs and compute nodes
- **OpenAI-Compatible API**: Provides RESTful endpoints aligned with OpenAI API specifications

## 🚀 Quick Start

### 1. Start NeMo Framework Container

For optimal performance and user experience, use the latest version of the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Please fetch the most recent `$TAG` and run the following command to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

### 2. Deploy a Model

```bash
# Deploy a NeMo checkpoint
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "/path/to/your/checkpoint" \
  --model_id megatron_model \
  --port 8080 \
  --host 0.0.0.0
```

### 3. Evaluate the Model

```python
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
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

For more information on the deployment, please see [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy).

#### 2. Evaluation Layer

- **NeMo Evaluator**: Provides standardized benchmark evaluations using packages from NVIDIA Eval Factory, bundled in the NeMo Framework container. The `lm-evaluation-harness` is pre-installed by default, and additional tools listed in the [support matrix](#-support-matrix) can be added as needed. For more information, see the [documentation](evaluation-doc).


## 📖 Usage Examples

### Basic Deployment with PyTriton as the Serving Backend

```bash
# Deploy model
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py \
  --nemo_checkpoint "/path/to/checkpoint" \
  --triton_model_name "megatron_model" \
  --server_port 8080 \
  --num_gpus 1 \
  --max_batch_size 4 \
  --inference_max_seq_length 8192
```

### Basic Evaluation

```Python
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
# Configure Endpoint
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model"
)
# Evaluation target configuration
target = EvaluationTarget(api_endpoint=api_endpoint)
# Configure EvaluationConfig with type, number of samples to evaluate on, etc.
config = EvaluationConfig(
    type="gsm8k",
    output_dir="results",
    params=ConfigParams(
        parallelism=4,      # number of concurrent requests to the server
        temperature=0,      # sampling temperature
        top_p=0,            # sampling top_p
        limit_samples=10,   # number of samples to evaluate on
    )
)

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
```

### Deploy with Multiple GPUs

```bash
# Deploy with tensor parallelism or pipeline parallelism
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py \
  --nemo_checkpoint "/path/to/checkpoint" \
  --triton_model_name "megatron_model" \
  --server_port 8080 \
  --num_gpus 4 \
  --tensor_parallelism_size 4 \
  --pipeline_parallelism_size 1 \
  --max_batch_size 8 \
  --inference_max_seq_length 8192
```

### Deploy with Ray

```bash
# Deploy 2 model replicas on 2 GPUs using Ray Serve
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "/path/to/checkpoint" \
  --model_id megatron_model \
  --num_gpus 2 \
  --num_replicas 2 \
  --num_cpus_per_replica 8 \
  --port 8080 \
  --include_dashboard \
  --cuda_visible_devices "0,1"
```

## 🔗 Related Projects

- [NeMo Export Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) - Model export and deployment


```{toctree}
:caption: Basic Usage
:hidden:

evaluation-doc.md
evaluation-hf.md
evaluation-mbridge.md
```

```{toctree}
:caption: Advanced
:hidden:

evaluation-with-ray.md
logprobs.md
custom-task.md
optional-eval-package.md
```
