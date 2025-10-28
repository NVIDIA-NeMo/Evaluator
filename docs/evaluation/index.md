(evaluation-overview)=

# About Evaluation

Evaluate LLMs, VLMs, agentic systems, and retrieval models across 100+ benchmarks using unified workflows.

## Before You Start

Before you run evaluations, ensure you have:

1. **Chosen your approach**: See {ref}`get-started-overview` for installation and setup guidance
2. **Deployed your model**: See {ref}`deployment-overview` for deployment options
3. **OpenAI-compatible endpoint**: Your model must expose a compatible API (see {ref}`deployment-testing-compatibility`).
4. **API credentials**: Access tokens for your model endpoint and Hugging Face Hub.

---

## Quick Start: Academic Benchmarks

:::{admonition} Fastest path to evaluate academic benchmarks
:class: tip

**For researchers and data scientists**: Evaluate your model on standard academic benchmarks in 3 steps.

**Step 1: Choose Your Approach**
- **Launcher CLI** (Recommended): `nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct`
- **Python API**: Direct programmatic control with `evaluate()` function

**Step 2: Select Benchmarks**

Common academic suites:
- **General Knowledge**: `mmlu_pro`, `gpqa_diamond`
- **Mathematical Reasoning**: `AIME_2025`, `mgsm`
- **Instruction Following**: `ifbench`, `mtbench`



Discover all available tasks:
```bash
nemo-evaluator-launcher ls tasks
```

**Step 3: Run Evaluation**

Create `config.yml`:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

evaluation:
  tasks:
    - name: mmlu_pro
    - name: ifbench
```

Launch the job:

```bash
export NGC_API_KEY=nvapi-...

nemo-evaluator-launcher run \
    --config-dir . \
    --config-name config.yml \
    -o execution.output_dir=results \
    -o +target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o +target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o +target.api_endpoint.api_key_name=NGC_API_KEY
```

<!-- **Next Steps**:
- {ref}`text-gen` - Complete text generation guide
- {ref}`eval-parameters` - Optimize configuration parameters
- {ref}`eval-benchmarks` - Explore all available benchmarks -->
:::

---

## Evaluation Workflows

Select a workflow based on your environment and desired level of control.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Launcher Workflows
:link: ../get-started/quickstart/launcher
:link-type: doc
Unified CLI for running evaluations across local, Slurm, and cloud backends with built-in result export.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Core API Workflows
:link: ../libraries/nemo-evaluator/workflows/python-api
:link-type: doc
Programmatic evaluation using Python API for integration into ML pipelines and custom workflows.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Container Workflows
:link: ../libraries/nemo-evaluator/containers/index
:link-type: doc
Direct container access for specialized use cases and custom evaluation environments.
:::

::::

## Configuration and Customization

Configure your evaluations, create custom tasks, explore benchmarks, and extend the framework with these guides.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Benchmark Catalog
:link: eval-benchmarks
:link-type: ref
Explore 100+ available benchmarks across 18 evaluation harnesses and their specific use cases.
:::

:::{grid-item-card} {octicon}`plus;1.5em;sd-mr-1` Extend Framework
:link: ../libraries/nemo-evaluator/extending/framework-definition-file/index
:link-type: doc
Add custom evaluation frameworks using Framework Definition Files for specialized benchmarks.
:::

::::

## Advanced Features

Scale your evaluations, export results, customize adapters, and resolve issues with these advanced features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Multi-Backend Execution
:link: ../libraries/nemo-evaluator-launcher/configuration/executors/index
:link-type: doc
Run evaluations on local machines, HPC clusters, or cloud platforms with unified configuration.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Result Export
:link: ../libraries/nemo-evaluator-launcher/exporters/index
:link-type: doc
Export evaluation results to MLflow, Weights & Biases, Google Sheets, and other platforms.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Adapter System
:link: ../libraries/nemo-evaluator/interceptors/index
:link-type: doc
Configure request/response processing, logging, caching, and custom interceptors.
:::

::::

## Core Evaluation Concepts

- For architectural details and core concepts, refer to {ref}`evaluation-model`.
- For container specifications, refer to {ref}`nemo-evaluator-containers`.
