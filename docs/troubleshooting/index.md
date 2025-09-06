(troubleshooting-index)=

# Troubleshooting

Common issues, solutions, and debugging techniques for NeMo Eval evaluations.

This section covers the most frequent problems encountered when running evaluations and provides systematic approaches to diagnose and resolve them. Start with the quick diagnostics to verify your basic setup, then navigate to specific issue categories as needed.

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

# Check health endpoint
health_response = requests.get("http://0.0.0.0:8080/v1/triton_health")
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
from nemo_eval.utils.base import list_available_evaluations

try:
    tasks = list_available_evaluations()
    print("Available frameworks:", list(tasks.keys()))
except ImportError as e:
    print(f"Missing dependency: {e}")
```

:::

::::

## Troubleshooting Topics

Choose the category that best matches your issue for targeted solutions and debugging steps.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Launcher Issues
:link: launcher-issues
:link-type: doc
Configuration validation, job management, multi-backend execution, and export problems
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation Issues
:link: installation-issues
:link-type: ref
Module import errors, missing dependencies, and framework installation problems
:::

:::{grid-item-card} {octicon}`key;1.5em;sd-mr-1` Authentication
:link: authentication
:link-type: ref
HuggingFace tokens, dataset access permissions, and gated model issues
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Issues
:link: deployment-issues
:link-type: ref
Model deployment problems, server connectivity, and inference failures
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: configuration-issues
:link-type: ref
Config parameters, tokenizer setup, and endpoint configuration problems
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Performance
:link: performance-issues
:link-type: ref
Memory optimization, scaling issues, and resource management
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Debugging & Best Practices
:link: debugging-guide
:link-type: ref
Debugging techniques, logging, monitoring, and prevention strategies
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Common Issues
:link: common-issues
:link-type: doc
Frequently encountered problems and their solutions based on user patterns
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

- **GitHub Issues**: [NeMo Eval Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)
- **Documentation**: [Complete Documentation](../index.md)

### Professional Support

For enterprise support, contact: [nemo-toolkit@nvidia.com](mailto:nemo-toolkit@nvidia.com)
