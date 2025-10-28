---
orphan: true
---

# Setup and Installation Issues

Solutions for getting {{ product_name_short }} up and running, including installation problems, authentication setup, and model deployment issues.

## Common Setup Problems

Before diving into specific issues, verify your basic setup with these quick checks:

::::{tab-set}

:::{tab-item} Installation Check

```bash
# Verify core packages are installed
pip list | grep nvidia

# Check for missing evaluation frameworks
python -c "from nemo_evaluator import show_available_tasks; show_available_tasks()"
```

:::

:::{tab-item} Authentication Check

```bash
# Verify HuggingFace token
huggingface-cli whoami

# Test token access
python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"
```

:::

:::{tab-item} Deployment Check

```bash
# Check if deployment server is running
# Use /health for vLLM, SGLang, NIM deployments
# Use /v1/triton_health for NeMo/Triton deployments
curl -I http://0.0.0.0:8080/health

# Verify GPU availability
nvidia-smi
```

:::

::::

## Setup Categories

Choose the category that matches your setup issue:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation Issues
:link: installation
:link-type: doc

Module import errors, missing dependencies, and framework installation problems.
:::

:::{grid-item-card} {octicon}`key;1.5em;sd-mr-1` Authentication Setup
:link: authentication
:link-type: doc

HuggingFace tokens, dataset access permissions, and gated model authentication.
:::

::::

:::{toctree}
:caption: Setup Issues
:hidden:

Installation <installation>
Authentication <authentication>
:::
