---
orphan: true
---

(troubleshooting-index)=

# Troubleshooting

Comprehensive troubleshooting guide for {{ product_name_short }} evaluations, organized by problem type and complexity level.

This section provides systematic approaches to diagnose and resolve evaluation issues. Start with the quick diagnostics below to verify your basic setup, then navigate to the appropriate troubleshooting category based on where your issue occurs in the evaluation workflow.

## Quick Start

Before diving into specific problem areas, run these basic checks to verify your evaluation environment:

::::{tab-set}

:::{tab-item} Launcher Quick Check

```bash
# Verify launcher installation and basic functionality
nv-eval --version

# List available tasks
nv-eval ls tasks

# Validate configuration without running
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run

# Check recent runs
nv-eval ls runs
```

:::

:::{tab-item} Model Endpoint Check

```python
import requests

# Check health endpoint (adjust based on your deployment)
# vLLM/SGLang/NIM: use /health
# NeMo/Triton: use /v1/triton_health
health_response = requests.get("http://0.0.0.0:8080/health", timeout=5)
print(f"Health Status: {health_response.status_code}")

# Test completions endpoint
test_payload = {
    "prompt": "Hello",
    "model": "megatron_model", 
    "max_tokens": 5
}
response = requests.post("http://0.0.0.0:8080/v1/completions/", json=test_payload)
print(f"Completions Status: {response.status_code}")
```

:::

:::{tab-item} Core API Check

```python
from nemo_evaluator import show_available_tasks

try:
    print("Available frameworks and tasks:")
    show_available_tasks()
except ImportError as e:
    print(f"Missing dependency: {e}")
```

:::

::::

## Troubleshooting Categories

Choose the category that best matches your issue for targeted solutions and debugging steps.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Setup & Installation
:link: setup-issues/index
:link-type: doc

Installation problems, authentication setup, and model deployment issues to get {{ product_name_short }} running.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Runtime & Execution
:link: runtime-issues/index
:link-type: doc

Configuration validation and launcher management during evaluation execution.
:::

::::

## Getting Help

### Log Collection

When reporting issues, include:

1. System Information:

   ```bash
   python --version
   pip list | grep nvidia
   nvidia-smi
   ```

2. Configuration Details:

   ```python
   print(f"Task: {eval_cfg.type}")
   print(f"Endpoint: {target_cfg.api_endpoint.url}")
   print(f"Model: {target_cfg.api_endpoint.model_id}")
   ```

3. Error Messages: Full stack traces and error logs

### Community Resources

- **GitHub Issues**: [{{ product_name_short }} Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)
- **Documentation**: {ref}`template-home`

### Professional Support

For enterprise support, contact: [nemo-toolkit@nvidia.com](mailto:nemo-toolkit@nvidia.com)
