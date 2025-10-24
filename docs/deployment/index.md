---
orphan: true
---

(deployment-overview)=

# Serve and Deploy Models

Deploy and serve models with NeMo Evaluator's flexible deployment options. Select a deployment strategy that matches your workflow, infrastructure, and requirements.

## Overview

NeMo Evaluator keeps model serving separate from evaluation execution, giving you flexible architectures and scalable workflows. Choose who manages deployment based on your needs.

### Key Concepts

- **Model-Evaluation Separation**: Models serve via OpenAI-compatible APIs, evaluations run in containers
- **Deployment Responsibility**: Choose who manages the model serving infrastructure
- **Multi-Backend Support**: Deploy locally, on HPC clusters, or in the cloud  
- **Universal Interceptors**: Request/response processing works across all deployment types

## Deployment Strategy Guide

### **Launcher-Orchestrated Deployment** (Recommended)
Let NeMo Evaluator Launcher handle both model deployment and evaluation orchestration:

```bash
# Launcher deploys model AND runs evaluation
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name slurm_llama_3_1_8b_instruct \
    -o deployment.checkpoint_path=/shared/models/llama-3.1-8b
```

**When to use:**

- You want automated deployment lifecycle management
- You need multi-backend execution (local, Slurm, Lepton)
- You prefer integrated monitoring and cleanup
- You want the simplest path from model to results

**Supported deployment types:** vLLM, NIM, SGLang, or no deployment (existing endpoints)

:::{seealso}
For detailed YAML configuration reference for each deployment type, see the {ref}`configuration-overview` in the NeMo Evaluator Launcher library.
:::

### **Bring-Your-Own-Endpoint**
You handle model deployment, NeMo Evaluator handles evaluation:

**Launcher users with existing endpoints:**
```bash
# Point launcher to your deployed model
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://localhost:8080/v1/completions
```

**Core library users:**
```python
from nemo_evaluator import evaluate, ApiEndpoint, EvaluationTarget, EvaluationConfig, ConfigParams

api_endpoint = ApiEndpoint(url="http://localhost:8080/v1/completions")
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="mmlu_pro", output_dir="./results")
evaluate(target_cfg=target, eval_cfg=config)
```

**When to use:**

- You have existing model serving infrastructure
- You need custom deployment configurations
- You want to deploy once and run many evaluations
- You have specific security or compliance requirements

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Manual Deployment
:link: bring-your-own-endpoint/manual-deployment
:link-type: doc
Deploy using vLLM, Ray Serve, or other serving frameworks.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Hosted Services
:link: bring-your-own-endpoint/hosted-services
:link-type: doc
Use NVIDIA Build, OpenAI, or other hosted model APIs.
:::

::::

### Available Deployment Types

The launcher supports multiple deployment types through Hydra configuration:

**vLLM Deployment**
```yaml
deployment:
  type: vllm
  checkpoint_path: /path/to/model  # Or HuggingFace model ID
  served_model_name: my-model
  tensor_parallel_size: 8
  data_parallel_size: 1
```

**NIM Deployment**  
```yaml
deployment:
  type: nim
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
```

**SGLang Deployment**
```yaml
deployment:
  type: sglang
  checkpoint_path: /path/to/model  # Or HuggingFace model ID
  served_model_name: my-model
  tensor_parallel_size: 8
  data_parallel_size: 1
```

**No Deployment**
```yaml
deployment:
  type: none  # Use existing endpoint
```

### Execution Backend Integration

**Local Backend**
```yaml
# Evaluates against existing endpoints only (no deployment)
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results
  
target:
  api_endpoint:
    url: http://localhost:8080/v1/completions
    model_id: my-model
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
```

**Slurm Backend**  
```yaml
# Deploys model on Slurm and runs evaluation
defaults:
  - execution: slurm/default
  - deployment: vllm
  - _self_

deployment:
  checkpoint_path: /shared/models/llama-3.1-8b
  served_model_name: meta-llama/Llama-3.1-8B-Instruct
  
execution:
  account: my-account
  output_dir: /shared/results
  partition: gpu
  walltime: "02:00:00"
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gpqa_diamond
```

**Lepton Backend**
```yaml
# Deploys model on Lepton and runs evaluation
defaults:
  - execution: lepton/default
  - deployment: vllm
  - _self_

deployment:
  checkpoint_path: meta-llama/Llama-3.1-8B-Instruct
  served_model_name: llama-3.1-8b-instruct
  lepton_config:
    resource_shape: gpu.1xh200
  
execution:
  output_dir: ./results
  
evaluation:
  tasks:
    - name: mmlu_pro
    - name: ifeval
```

## Bring-Your-Own-Endpoint Options

Choose from these approaches when managing your own deployment:

### Manual Deployment
- **vLLM**: High-performance serving with PagedAttention optimization
- **Custom serving**: Any OpenAI-compatible endpoint (verify compatibility with our [Testing Endpoint Compatibility](bring-your-own-endpoint/testing-endpoint-oai-compatibility.md) guide)

### Hosted Services  
- **NVIDIA Build**: Ready-to-use hosted models with OpenAI-compatible APIs
- **OpenAI API**: Direct integration with OpenAI's models
- **Other providers**: Any service providing OpenAI-compatible endpoints

### Enterprise Integration
- **Kubernetes deployments**: Container orchestration in production environments
- **Existing MLOps pipelines**: Integration with current model serving infrastructure
- **Custom infrastructure**: Specialized deployment requirements

## Usage Examples

### With Launcher
```bash
# Point to any existing endpoint
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://your-endpoint:8080/v1/completions \
    -o target.api_endpoint.model_id=your-model-name
```

### With Core Library
```python
from nemo_evaluator import (
    evaluate,
    ApiEndpoint,
    EvaluationConfig,
    EvaluationTarget,
    ConfigParams
)

# Configure any endpoint
api_endpoint = ApiEndpoint(
    url="http://your-endpoint:8080/v1/completions",
    model_id="your-model-name"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="results",
    params=ConfigParams(limit_samples=100)
)

evaluate(target_cfg=target, eval_cfg=config)
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
:link: ../libraries/nemo-evaluator/interceptors/index
:link-type: doc
Configure logging, caching, reasoning, and custom request processing.
:::

::::
