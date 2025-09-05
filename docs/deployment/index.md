(deployment-overview)=

# Model Serving & Deployment

Deploy and serve models for evaluation using NeMo Evaluator's flexible deployment options. Choose from launcher-managed deployment, self-managed serving, or hosted endpoints based on your needs.

## Overview

NeMo Evaluator provides multiple approaches to model deployment and serving, from fully managed orchestration to self-hosted endpoints. The platform separates model serving from evaluation execution, enabling flexible architectures and scalable workflows.

### Key Concepts
- **Model-Evaluation Separation**: Models serve via OpenAI-compatible APIs, evaluations run in containers
- **Multi-Backend Support**: Deploy locally, on HPC clusters, or in the cloud  
- **Lifecycle Management**: Automatic deployment, scaling, and cleanup
- **Resource Optimization**: Efficient GPU utilization and request batching

## Deployment Approaches

### 🚀 **Launcher-Managed Deployment** (Recommended)
Let the launcher handle model deployment and evaluation orchestration:

```bash
# Launcher deploys model AND runs evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name slurm_vllm_llama_3_1_8b \
    -o deployment.model_path=/shared/models/llama-3.1-8b \
    -o evaluation.tasks='["mmlu_pro", "gsm8k"]'
```

**Benefits:**
- ✅ Automatic deployment lifecycle management  
- ✅ Multi-backend support (local, Slurm, Lepton)
- ✅ Built-in cleanup and resource management
- ✅ Integrated monitoring and logging

### ⚙️ **Self-Managed Deployment**
Deploy your own model endpoint for use with evaluations:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` PyTriton Backend
:link: pytriton-deployment
:link-type: ref
High-performance inference through NVIDIA Triton Inference Server with multi-node model parallelism support.
:::

:::{grid-item-card} {octicon}`organization;1.5em;sd-mr-1` Ray Serve
:link: ray-serve  
:link-type: ref
Multi-instance evaluation with single-node model parallelism and horizontal scaling.
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` vLLM Serving
:link: #vllm-serving
:link-type: ref
Fast inference serving with optimized attention mechanisms and continuous batching.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Hosted Endpoints
:link: #hosted-endpoints  
:link-type: ref
Use existing hosted models from NVIDIA Build, OpenAI, or other providers.
:::

::::

### 🐳 **Container-Integrated Deployment**
Deploy models within evaluation containers for isolated workflows:

```bash
# Pull container with integrated model serving
docker run --gpus all -v /models:/models \
    nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 \
    eval-factory run_eval \
        --eval_type mmlu_pro \
        --model_path /models/llama-3.1-8b \
        --output_dir /results
```

## Integration with Launcher Backends

### Local Backend Integration
```yaml
# examples/local_with_deployment.yaml
deployment:
  type: vllm
  model_path: /path/to/model
  port: 8080
  gpu_memory_utilization: 0.9
  
execution:
  backend: local
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: deployed-model
```

### Slurm Backend Integration  
```yaml
# examples/slurm_with_deployment.yaml
deployment:
  type: nemo_framework
  model_path: /shared/models/llama-3.1-8b.nemo
  tensor_parallelism: 4
  pipeline_parallelism: 1
  
execution:
  backend: slurm
  partition: gpu
  nodes: 1
  gpus_per_node: 4
  time_limit: "02:00:00"
  
evaluation:
  tasks:
    - name: mmlu_pro
      params:
        limit_samples: 1000
```

### Lepton Backend Integration
```yaml
# examples/lepton_with_deployment.yaml
deployment:
  type: vllm
  model_path: meta-llama/Llama-3.1-8B-Instruct
  resource_shape: gpu.a100.1x
  
execution:
  backend: lepton
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
    - name: humaneval
```

## vLLM Serving

Fast inference serving with optimized attention and continuous batching:

```bash
# Install vLLM
pip install vllm

# Deploy model with vLLM
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8080 \
    --served-model-name llama-3.1-8b \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

**Integration with NeMo Evaluator:**
```bash
# Run evaluation against vLLM endpoint
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_external_endpoint \
    -o target.api_endpoint.url=http://localhost:8080/v1/chat/completions \
    -o target.api_endpoint.model_id=llama-3.1-8b
```

## Hosted Endpoints

Use existing hosted models without deployment:

### NVIDIA Build
```bash
# Run evaluation against NVIDIA Build endpoint
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name hosted_nvidia_build \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o target.api_endpoint.api_key=${NGC_API_KEY}
```

### OpenAI Compatible
```bash
# Run evaluation against OpenAI API
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name hosted_openai \
    -o target.api_endpoint.url=https://api.openai.com/v1/chat/completions \
    -o target.api_endpoint.model_id=gpt-4 \
    -o target.api_endpoint.api_key=${OPENAI_API_KEY}
```

## Evaluation Adapters

Advanced request/response processing for all deployment types:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Usage & Configuration
:link: adapters-usage
:link-type: ref
Learn how to enable adapters and configure interceptor chains for any deployment.
:::

:::{grid-item-card} Reasoning Cleanup
:link: adapters-recipe-reasoning
:link-type: ref
Strip intermediate reasoning tokens before scoring across all model types.
:::

:::{grid-item-card} Custom System Prompt (Chat)
:link: adapters-recipe-system-prompt
:link-type: ref
Enforce standard system prompts for consistent evaluation across endpoints.
:::

:::{grid-item-card} Advanced Interceptors
:link: ../nemo-evaluator/reference/configuring_interceptors
:link-type: doc
Configure logging, caching, reasoning, and custom request processing.
:::

::::
