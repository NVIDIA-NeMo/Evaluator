(launcher-orchestrated-deployment)=

# Launcher-Orchestrated Deployment

Let NeMo Evaluator Launcher handle both model deployment and evaluation orchestration automatically. This is the recommended approach for most users, providing automated lifecycle management, multi-backend support, and integrated monitoring.

## Overview

Launcher-orchestrated deployment means the launcher:
- Deploys your model using the specified deployment type
- Manages the model serving lifecycle
- Runs evaluations against the deployed model
- Handles cleanup and resource management

Based on PR #108, the launcher supports multiple deployment backends and execution environments.

## Quick Start

```bash
# Deploy model and run evaluation in one command
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o deployment.model_path=/path/to/your/model \
    -o deployment.type=vllm
```

## Execution Backends

Choose the execution backend that matches your infrastructure:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Local Deployment
:link: local
:link-type: doc
Deploy and evaluate on your local machine. Perfect for development, testing, and small-scale evaluations.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Slurm Deployment
:link: slurm
:link-type: doc
Deploy on HPC clusters with Slurm workload manager. Ideal for large-scale evaluations with multi-node parallelism.
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Lepton Deployment
:link: lepton
:link-type: doc
Deploy on Lepton AI cloud platform. Best for cloud-native deployments with managed infrastructure and auto-scaling.
:::

::::

## Deployment Types

The launcher supports multiple deployment types from PR #108's implementation:

### vLLM Deployment
- **Fast inference** with optimized attention mechanisms
- **Continuous batching** for high throughput
- **Tensor parallelism** support for large models
- **Memory optimization** with configurable GPU utilization

### NIM Deployment  
- **Production-grade reliability** with enterprise features
- **NVIDIA optimized containers** for maximum performance
- **Built-in monitoring** and logging capabilities
- **Enterprise security** features

### SGLang Deployment
- **Structured generation** support for complex tasks
- **Function calling** capabilities
- **JSON mode** for structured outputs
- **Efficient batching** for high throughput

### No Deployment
- **Use existing endpoints** without launcher deployment
- **Bring-your-own-endpoint** integration
- **Flexible configuration** for any OpenAI-compatible API

## Configuration Overview

Basic configuration structure for launcher-orchestrated deployment:

```yaml
# Deployment configuration
deployment:
  type: vllm  # or nim, sglang, none
  model_path: /path/to/model
  # ... deployment-specific options

# Execution backend
execution:
  backend: local  # or slurm, lepton
  # ... backend-specific options

# Evaluation tasks
evaluation:
  tasks:
    - name: mmlu_pro
    - name: gsm8k
  # ... evaluation options

# Target endpoint (auto-configured)
target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: deployed-model
```

## Key Benefits

### Automated Lifecycle Management
- **Deployment automation**: No manual setup required
- **Resource management**: Automatic allocation and cleanup  
- **Error handling**: Built-in retry and recovery mechanisms
- **Monitoring integration**: Real-time status and logging

### Multi-Backend Support
- **Consistent interface**: Same commands work across all backends
- **Environment flexibility**: Local development to production clusters
- **Resource optimization**: Backend-specific optimizations
- **Scalability**: From single GPU to multi-node deployments

### Integrated Workflows
- **End-to-end automation**: From model to results in one command
- **Configuration management**: Version-controlled, reproducible configs
- **Result integration**: Built-in export and analysis tools
- **Monitoring and debugging**: Comprehensive logging and status tracking

## Getting Started

1. **Choose your backend**: Start with {ref}`launcher-orchestrated-local` for development
2. **Configure your model**: Set deployment type and model path
3. **Run evaluation**: Use the launcher to deploy and evaluate
4. **Monitor progress**: Check status and logs during execution
5. **Analyze results**: Export and analyze evaluation outcomes

## Next Steps

- **Local Development**: Start with {ref}`launcher-orchestrated-local` for testing
- **Scale Up**: Move to {ref}`launcher-orchestrated-slurm` for production workloads  
- **Cloud Native**: Try {ref}`launcher-orchestrated-lepton` for managed infrastructure
- **Configure Adapters**: Set up {ref}`adapters` for custom processing

```{toctree}
:maxdepth: 1
:hidden:

Local Deployment <local>
Slurm Deployment <slurm>
Lepton Deployment <lepton>
```
