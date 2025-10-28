(template-home)=

# NeMo Evaluator SDK Documentation

Welcome to the NeMo Evaluator SDK Documentation.

````{div} sd-d-flex-row
```{button-ref} get-started/install
:ref-type: doc
:color: primary
:class: sd-rounded-pill sd-mr-3

Install
```

```{button-ref} get-started/quickstart/launcher
:ref-type: doc
:color: secondary
:class: sd-rounded-pill

Quickstart Evaluations
```
````

---

## Introduction to NeMo Evaluator SDK

Discover how NeMo Evaluator SDK works and explore its key features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` About NeMo Evaluator SDK
:link: about/index
:link-type: doc
Explore the NeMo Evaluator Core and Launcher architecture
:::

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Key Features
:link: about/key-features
:link-type: doc
Discover NeMo Evaluator SDK's powerful capabilities.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Concepts
:link: about/concepts/index
:link-type: doc
Master core concepts powering NeMo Evaluator SDK.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Release Notes
:link: about/release-notes/index
:link-type: doc
Release notes for the NeMo Evaluator SDK.
:::
::::

## Choose a Quickstart

Select the evaluation approach that best fits your workflow and technical requirements.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Launcher
:link: gs-quickstart-launcher
:link-type: ref

Use the CLI to orchestrate evaluations with automated container management.
+++
{bdg-secondary}`cli`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Core
:link: gs-quickstart-core
:link-type: ref

Get direct Python API access with full adapter features, custom configurations, and workflow integration capabilities.

+++
{bdg-secondary}`api`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Container
:link: gs-quickstart-container
:link-type: ref

Gain full control over the container environment with volume mounting, environment variable management, and integration into Docker-based CI/CD pipelines.

+++
{bdg-secondary}`Docker`
:::

::::

<!-- ## Evaluation Workflows

Explore different evaluation methodologies tailored to specific model capabilities and use cases.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Text Generation
:link: text-gen
:link-type: ref
Evaluate models through natural language generation for academic benchmarks, reasoning tasks, and general knowledge assessment.

:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Log-Probability
:link: log-probability
:link-type: ref
Assess model confidence and uncertainty using log-probabilities for multiple-choice scenarios without text generation.

:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: code-generation
:link-type: ref
Evaluate programming capabilities through code generation, completion, and Algorithmic Problem Solving.

:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Safety & Security
:link: safety-security
:link-type: ref
Test AI safety, alignment, and security vulnerabilities using specialized safety harnesses and probing techniques.

:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Function Calling
:link: function-calling
:link-type: ref
Assess tool use capabilities, API calling accuracy, and structured output generation for agent-like behaviors.

:::

:::: -->

<!-- ## Model Deployment

Choose your deployment strategy based on your infrastructure needs and operational preferences.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Launcher-Orchestrated
:link: launcher-orchestrated-deployment
:link-type: ref
Let the launcher handle model deployment and evaluation orchestration automatically. Recommended for most users.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Bring-Your-Own-Endpoint
:link: bring-your-own-endpoint
:link-type: ref
Deploy and manage model serving yourself, then point NeMo Evaluator to your endpoint for full infrastructure control.
:::

:::: -->

<!-- ### Evaluation Adapters

Customize model behavior during evaluation with interceptors for preprocessing, post-processing, and response modification.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Usage
:link: adapters-usage
:link-type: ref
Learn how to enable adapters and pass `AdapterConfig` to `evaluate`.
:::

:::{grid-item-card} Reasoning Cleanup
:link: adapters-recipe-reasoning
:link-type: ref
Strip intermediate reasoning tokens before scoring.
:::

:::{grid-item-card} Custom System Prompt (Chat)
:link: adapters-recipe-system-prompt
:link-type: ref
Enforce a standard system prompt for chat endpoints.
:::

:::{grid-item-card} Request Parameter Modification
:link: adapters-recipe-response-shaping
:link-type: ref
Standardize request parameters across endpoint providers.
:::

:::{grid-item-card} Logging Caps
:link: adapters-recipe-logging
:link-type: ref
Control logging volume for requests and responses.
:::

:::{grid-item-card} Configuration
:link: adapters-configuration
:link-type: ref
View available `AdapterConfig` options and defaults.
:::

:::: -->

## Libraries

### Launcher

Orchestrate evaluations across different execution backends with unified CLI and programmatic interfaces.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: libraries/nemo-evaluator-launcher/configuration/index
:link-type: doc

Complete configuration schema, examples, and advanced patterns for all use cases.
+++
{bdg-secondary}`Setup`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Executors
:link: libraries/nemo-evaluator-launcher/configuration/executors/index
:link-type: doc

Run evaluations on local machines, HPC clusters (Slurm), or cloud platforms (Lepton AI).
+++
{bdg-secondary}`Execution`
:::

:::{grid-item-card} {octicon}`upload;1.5em;sd-mr-1` Exporters
:link: libraries/nemo-evaluator-launcher/exporters/index
:link-type: doc

Export results to MLflow, Weights & Biases, Google Sheets, or local files with one command.
+++
{bdg-secondary}`Export`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Python API
:link: libraries/nemo-evaluator-launcher/api
:link-type: doc

Programmatic access for notebooks, automation, and custom evaluation workflows.
+++
{bdg-secondary}`API`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: libraries/nemo-evaluator-launcher/cli
:link-type: doc

Complete command-line interface documentation with examples and usage patterns.
+++
{bdg-secondary}`CLI`
:::

::::

### Core

Access the core evaluation engine directly with containerized benchmarks and flexible adapter architecture.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Workflows
:link: libraries/nemo-evaluator/workflows/index
:link-type: doc

Use the evaluation engine through Python API, containers, or programmatic workflows.
+++
{bdg-secondary}`Integration`
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containers
:link: libraries/nemo-evaluator/containers/index
:link-type: doc

Ready-to-use evaluation containers with curated benchmarks and frameworks.
+++
{bdg-secondary}`Containers`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Interceptors
:link: libraries/nemo-evaluator/interceptors/index
:link-type: doc

Configure request/response interceptors for logging, caching, and custom processing.
+++
{bdg-secondary}`Customization`
:::

:::{grid-item-card} {octicon}`log;1.5em;sd-mr-1` Logging
:link: libraries/nemo-evaluator/logging
:link-type: doc

Comprehensive logging setup for evaluation runs, debugging, and audit trails.
+++
{bdg-secondary}`Monitoring`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Extending
:link: libraries/nemo-evaluator/extending/index
:link-type: doc

Add custom benchmarks and frameworks by defining configuration and interfaces.
+++
{bdg-secondary}`Extension`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: libraries/nemo-evaluator/api
:link-type: doc

Python API documentation for programmatic evaluation control and integration.
+++
{bdg-secondary}`API`
:::

::::

:::{toctree}
:hidden:
Home <self>
:::

:::{toctree}
:caption: About
:hidden:

Overview <about/index>
Key Features <about/key-features>
Concepts <about/concepts/index>
Release Notes <about/release-notes/index>
:::

:::{toctree}
:caption: Get Started
:hidden:

Getting Started <get-started/index>
Install SDK <get-started/install>
Quickstart <get-started/quickstart/index>
:::

:::{toctree}
:caption: Tutorials
:hidden:

About Tutorials <tutorials/index>
Tutorials for NeMo Framework <tutorials/nemo-fw/index>
Evaluate an Existing Endpoint <tutorials/local-evaluation-of-existing-endpoint>
:::


<!--
TODO: Add below once ready
Evaluation Configuration Parameters <evaluation/parameters>
Custom Task Configuration <evaluation/custom-tasks> 
-->
:::{toctree}
:caption: Evaluation
:hidden:

About Model Evaluation <evaluation/index>
Benchmark Catalog <evaluation/benchmarks>
:::

<!-- :::{toctree}
:caption: NeMo Framework
:hidden:

About NeMo Framework <nemo-fw/index>
::: -->

<!-- TODO: Add these pages once ready:
Evaluation Adapters <deployment/adapters/index>
Launcher-Orchestrated <deployment/launcher-orchestrated/index> -->

:::{toctree}
:caption: Model Deployment
:hidden:

About Model Deployment <deployment/index>
Bring-Your-Own-Endpoint <deployment/bring-your-own-endpoint/index>
:::

:::{toctree}
:caption: Libraries
:hidden:

About NeMo Evaluator Libraries <libraries/index>
Launcher <libraries/nemo-evaluator-launcher/index>
Core <libraries/nemo-evaluator/index>
:::

<!-- :::{toctree}
:caption: Troubleshooting
:hidden:

About Troubleshooting <troubleshooting/index>
Setup & Installation <troubleshooting/setup-issues/index>
Runtime & Execution <troubleshooting/runtime-issues/index>
::: -->

:::{toctree}
:caption: References
:hidden:

About References <references/index>
FAQ <references/faq>
:::