---
orphan: true
---

(evaluation-overview)=

# About Evaluation

Evaluate LLMs, VLMs, agentic systems, and retrieval models across 100+ benchmarks using unified workflows.

## Before You Start

Before you run evaluations, ensure you have:

1. **Chosen your approach**: See {ref}`get-started-overview` for installation and setup guidance
2. **Deployed your model**: See {ref}`deployment-overview` for deployment options
3. **OpenAI-compatible endpoint**: Your model must expose a compatible API
4. **API credentials**: Access tokens for your model endpoint

---

## Quick Start: Academic Benchmarks

:::{admonition} Fastest path to evaluate academic benchmarks
:class: tip

**For researchers and data scientists**: Evaluate your model on standard academic benchmarks in 3 steps.

**Step 1: Choose Your Approach**
- **Launcher CLI** (Recommended): `nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct`
- **Python API**: Direct programmatic control with `evaluate()` function

**Step 2: Select Benchmarks**

Common academic suites:
- **Language Understanding**: `mmlu_pro`, `arc_challenge`, `hellaswag`, `truthfulqa`
- **Mathematical Reasoning**: `gsm8k`, `math`
- **Instruction Following**: `ifeval`, `gpqa_diamond`

Discover all available tasks:
```bash
nemo-evaluator-launcher ls tasks
```

**Step 3: Run Evaluation**

Using Launcher CLI:
```bash
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["mmlu_pro", "gsm8k", "arc_challenge"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}
```

Using Python API:
```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams, EndpointType
)

# Configure and run
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=100,      # Start with subset
        temperature=0.01,       # Near-deterministic
        max_new_tokens=512,
        parallelism=4
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        type=EndpointType.CHAT,
        api_key="YOUR_API_KEY"
    )
)

result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
```

**Next Steps**:
- {ref}`text-gen` - Complete text generation guide
- {ref}`eval-parameters` - Optimize configuration parameters
- {ref}`eval-benchmarks` - Explore all available benchmarks
:::

---

## Evaluation Workflows

Select a workflow based on your environment and desired level of control.

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
:link: ../libraries/nemo-evaluator/containers/index
:link-type: doc
Direct container access for specialized use cases and custom evaluation environments.
:::

::::

## Configuration and Customization

Configure your evaluations, create custom tasks, explore benchmarks, and extend the framework with these guides.

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

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Troubleshooting
:link: ../troubleshooting/index
:link-type: doc
Resolve common evaluation issues, debug configuration problems, and optimize evaluation performance.
:::

::::

## Core Evaluation Concepts

- For architectural details and core concepts, refer to {ref}`evaluation-model`.
- For container specifications, refer to {ref}`nemo-evaluator-containers`.
