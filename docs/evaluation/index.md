(evaluation-overview)=

# Model Evaluation

Comprehensive guide to AI model evaluation using the NeMo Evaluator platform. Choose from multiple approaches to assess model performance across 100+ benchmarks spanning LLMs, VLMs, agentic AI, and retrieval systems.

## Overview

NeMo Evaluator provides three main approaches for running evaluations:

### **NeMo Evaluator Launcher** (Recommended)
Unified CLI and orchestration for most evaluation needs:
- **100+ benchmarks** across 18 evaluation harnesses
- **Multi-backend execution** (local, Slurm, cloud)
- **Built-in result export** to MLflow, W&B, Google Sheets
- **Configuration management** with reproducible runs

### **NeMo Evaluator Core**
Programmatic API for custom evaluation pipelines:
- **Python API** for integration into ML workflows
- **Direct container access** for specialized use cases
- **Advanced adapter configuration** for request/response processing
- **Custom framework support** via Framework Definition Files

### **Container Direct**
Direct access to NGC evaluation containers:
- **Pre-built containers** for each evaluation framework
- **Guaranteed reproducibility** across environments
- **Isolated evaluation environments** with all dependencies
- **Custom container workflows** for specialized needs

## Choose Your Approach

### For Most Users: Start with the Launcher
```bash
# Install and run your first evaluation
pip install nemo-evaluator-launcher
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
```

### For Developers: Use the Core API
```python
# Programmatic evaluation
from nemo_evaluator.core.evaluate import evaluate
result = evaluate(eval_cfg=config, target_cfg=target)
```

### For Container Workflows: Direct NGC Access
```bash
# Pull and run evaluation container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
```

## Prerequisites

### Required
- **OpenAI-compatible endpoint**: Your model must expose a compatible API
- **API credentials**: Access tokens for your model endpoint
- **Docker** (for container-based workflows)

### Optional
- **GPU access**: For local model deployment
- **HPC cluster access**: For Slurm-based execution
- **Cloud credentials**: For hosted execution backends

---

## Evaluation Workflows

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run Evaluations
:link: run-evals/index
:link-type: doc
Step-by-step guides for different evaluation scenarios using launcher, core API, and container workflows.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Launcher Workflows
:link: ../libraries/nemo-evaluator-launcher/quickstart
:link-type: doc
Unified CLI for running evaluations across local, Slurm, and cloud backends with built-in result export.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Core API Workflows
:link: ../libraries/nemo-evaluator/workflows/python-api
:link-type: doc
Programmatic evaluation using Python API for integration into ML pipelines and custom workflows.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Container Workflows
:link: ../libraries/nemo-evaluator/workflows/using_containers
:link-type: doc
Direct container access for specialized use cases and custom evaluation environments.
:::

::::

## Configuration and Customization

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Parameters
:link: eval-parameters
:link-type: ref
Comprehensive reference for evaluation configuration parameters, optimization patterns, and framework-specific settings.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom Task Configuration
:link: eval-custom-tasks
:link-type: ref
Learn how to configure evaluations for tasks without pre-defined configurations using custom benchmark definitions.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Benchmark Catalog
:link: eval-benchmarks
:link-type: ref
Explore 100+ available benchmarks across 18 evaluation harnesses and their specific use cases.
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Extend Framework
:link: ../libraries/nemo-evaluator/extending/framework_definition_file
:link-type: doc
Add custom evaluation frameworks using Framework Definition Files for specialized benchmarks.
:::

::::

## Advanced Features

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Multi-Backend Execution
:link: ../libraries/nemo-evaluator-launcher/executors/overview
:link-type: doc
Run evaluations on local machines, HPC clusters, or cloud platforms with unified configuration.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Result Export
:link: ../libraries/nemo-evaluator-launcher/exporters/overview
:link-type: doc
Export evaluation results to MLflow, Weights & Biases, Google Sheets, and other platforms.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Adapter System
:link: ../libraries/nemo-evaluator/interceptors/index
:link-type: doc
Configure request/response processing, logging, caching, and custom interceptors.
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Troubleshooting
:link: ../troubleshooting/index
:link-type: doc
Resolve common evaluation issues, debug configuration problems, and optimize evaluation performance.
:::

::::

## Core Evaluation Concepts

For architectural details and core concepts, see {ref}`evaluation-model`. For container specifications, see the [Container Reference](../libraries/nemo-evaluator/containers/index.md).
