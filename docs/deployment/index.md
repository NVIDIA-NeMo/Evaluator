(deployment-overview)=

# Serve and Deploy Models

Deploy and serve models with NeMo Evaluator's flexible deployment options. Select a deployment strategy that matches your workflow, infrastructure, and requirements.

## Overview

NeMo Evaluator keeps model serving separate from evaluation execution, giving you flexible architectures and scalable workflows. Choose who manages deployment based on your needs.

### Key Concepts

- **Model-Evaluation Separation**: Models serve via OpenAI-compatible APIs, evaluations run in containers
- **Deployment Responsibility**: Choose who manages the model serving infrastructure
- **Multi-Backend Support**: Deploy locally, on HPC clusters, or in the cloud  

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
- You prefer integrated monitoring and cleanup
- You want the simplest path from model to results

**Supported deployment types:** vLLM, NIM, SGLang, TRT-LLM, or no deployment (existing endpoints)

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
    -o target.api_endpoint.url=http://localhost:8080/v1/chat/completions
```

**Core library users:**
```python
from nemo_evaluator import evaluate, ApiEndpoint, EvaluationTarget, EvaluationConfig

api_endpoint = ApiEndpoint(url="http://localhost:8080/v1/completions")
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="./results")
evaluate(target_cfg=target, eval_cfg=config)
```

**When to use:**

- You have existing model serving infrastructure
- You need custom deployment configurations
- You want to deploy once and run many evaluations
- You have specific security or compliance requirements

<!-- ::::{grid} 1 2 2 2
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

:::: -->

## Available Deployment Types

The launcher supports multiple deployment types through Hydra configuration:

**vLLM Deployment**
```yaml
deployment:
  type: vllm
  image: vllm/vllm-openai:latest
  hf_model_handle: hf-model/handle  # HuggingFace ID
  checkpoint_path: null             # or provide a path to the stored checkpoint
  served_model_name: your-model-name
  port: 8000
```

**NIM Deployment**  
```yaml
deployment:
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.6
  served_model_name: meta/llama-3.1-8b-instruct
  port: 8000
```

**SGLang Deployment**
```yaml
deployment:
  type: sglang
  image: lmsysorg/sglang:latest
  hf_model_handle: hf-model/handle  # HuggingFace ID
  checkpoint_path: null             # or provide a path to the stored checkpoint
  served_model_name: your-model-name
  port: 8000
```

**No Deployment**
```yaml
deployment:
  type: none  # Use existing endpoint
```

## Bring-Your-Own-Endpoint Options

Choose from these approaches when managing your own deployment:

<!-- TODO: uncomment once manual deployment guide is ready -->
<!-- 
### Manual Deployment
- **vLLM**: High-performance serving with PagedAttention optimization
- **Custom serving**: Any OpenAI-compatible endpoint (verify compatibility with our {ref}`deployment-testing-compatibility` guide) -->

### Hosted Services  
- **NVIDIA Build**: Ready-to-use hosted models with OpenAI-compatible APIs
- **OpenAI API**: Direct integration with OpenAI's models
- **Other providers**: Any service providing OpenAI-compatible endpoints

### Enterprise Integration
- **Kubernetes deployments**: Container orchestration in production environments
- **Existing MLOps pipelines**: Integration with current model serving infrastructure
- **Custom infrastructure**: Specialized deployment requirements
