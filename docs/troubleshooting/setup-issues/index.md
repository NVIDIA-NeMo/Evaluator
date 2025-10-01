# Setup and Installation Issues

Solutions for getting NeMo Eval up and running, including installation problems, authentication setup, and model deployment issues.

## Common Setup Problems

Before diving into specific issues, verify your basic setup with these quick checks:

::::{tab-set}

:::{tab-item} Installation Check

```bash
# Verify core packages are installed
pip list | grep nvidia

# Check for missing evaluation frameworks
python -c "from nemo_eval.utils.base import list_available_evaluations; print(list(list_available_evaluations().keys()))"
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
curl -I http://0.0.0.0:8080/v1/triton_health

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

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Issues
:link: deployment
:link-type: doc

Model deployment problems, server connectivity, and inference setup failures.
:::

::::

## Quick Resolution Steps

1. **Start with Installation**: Ensure all required packages are installed
2. **Configure Authentication**: Set up tokens for gated datasets and models
3. **Test Deployment**: Verify your model endpoint is accessible
4. **Validate Setup**: Run a minimal evaluation to confirm everything works

If setup issues persist, check the {doc}`../advanced/index` section for comprehensive debugging techniques.

:::{toctree}
:caption: Setup Issues
:hidden:

Installation <installation>
Authentication <authentication>
Deployment <deployment>
:::
