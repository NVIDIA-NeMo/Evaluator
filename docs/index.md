(template-home)=

# NeMo Evaluator Documentation

Welcome to NeMo Evaluator - NVIDIA's comprehensive platform for AI model evaluation and benchmarking.

## Overview

NeMo Evaluator is NVIDIA's open-source evaluation stack that provides a unified platform for AI model evaluation and benchmarking. It consists of two core libraries: **nemo-evaluator** (the core evaluation engine) and **nemo-evaluator-launcher** (the user interface and orchestration layer). Together, these components enable consistent, scalable, and reproducible evaluation of GenAI models spanning LLMs, VLMs, agentic AI, and retrieval systems.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeMo Evaluator Ecosystem                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐     │
│  │   nemo-         │                    │   nemo-         │     │
│  │  evaluator      │                    │  evaluator-     │     │
│  │                 │                    │  launcher       │     │
│  │  Core evaluation│◄──────────────────►│  User interface │     │
│  │  engine &       │                    │  & orchestration│     │
│  │  adapters       │                    │                 │     │
│  └─────────────────┘                    └─────────────────┘     │
│           │                              │                      │
│           ▼                              ▼                      │
│  ┌─────────────────┐                    ┌─────────────────┐     │
│  │   Evaluation    │                    │   CLI & API     │     │
│  │   Frameworks    │                    │   Interface     │     │
│  │   & Containers  │                    │                 │     │
│  └─────────────────┘                    └─────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Choose Your Entry Point

NeMo Evaluator provides two primary paths for different user needs. Choose based on your workflow requirements:

```{mermaid}
flowchart LR
    A[New to NeMo Evaluator?] --> B{What's your use case?}
    
    B -->|Quick evaluations<br/>CLI interface<br/>Multi-backend support| C[🚀 Launcher<br/>Recommended]
    B -->|Custom workflows<br/>Programmatic control<br/>Integrations| D[⚙️ Core<br/>Advanced]
    
    C --> E[Start with Launcher Quickstart]
    D --> F[Start with Core API Guide]
    
    style C fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style D fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
```

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` NeMo Evaluator Launcher
:link: nemo-evaluator-launcher-overview
:link-type: ref
**Recommended for most users**: Unified CLI and orchestration for running evaluations across local, Slurm, and cloud backends with 100+ benchmarks.

**Best for**: Researchers, ML engineers, teams wanting turnkey evaluation capabilities
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NeMo Evaluator Core
:link: nemo-evaluator-overview
:link-type: ref
**For developers and integrations**: Core evaluation engine, adapter system, and containerized frameworks for programmatic access.

**Best for**: Custom pipelines, system integrations, framework extensions
:::

:::{grid-item-card} {octicon}`info;1.5em;sd-mr-1` About & Concepts
:link: about-concepts
:link-type: ref
Learn the core concepts: evaluation model, adapters, deployment patterns, and system architecture.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: get-started-overview
:link-type: ref
Get up and running with your first evaluation in just a few minutes using either the launcher or core library.
:::

::::

## Evaluation Workflows

```{note}
You need access to a model endpoint to run evaluations. If you already have an OpenAI-compatible endpoint, continue with the guides below. Otherwise, first deploy an endpoint in {ref}`deployment-overview`.
```

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
Evaluate programming capabilities through code generation, completion, and algorithmic problem solving.

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

::::

## Model Deployment

### Backend Options

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` PyTriton Backend
:link: pytriton-deployment
:link-type: ref
High-performance inference through NVIDIA Triton Inference Server with multi-node model parallelism support for production deployments.
:::

:::{grid-item-card} {octicon}`organization;1.5em;sd-mr-1` Ray Serve
:link: ray-serve
:link-type: ref
Multi-instance evaluation capabilities with single-node model parallelism and horizontal scaling for accelerated evaluations.
:::

::::

### Evaluation Adapters

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

:::{grid-item-card} Response Shaping
:link: adapters-recipe-response-shaping
:link-type: ref
Normalize outputs for evaluators and downstream tools.
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

About Getting Started <get-started/index>
Install Eval <get-started/install>
Quickstart <get-started/quickstart>
:::

:::{toctree}
:caption: NeMo Evaluator Launcher
:hidden:

About Launcher <nemo-evaluator-launcher/index>
Quickstart <nemo-evaluator-launcher/quickstart>
Executors <nemo-evaluator-launcher/executors/overview>
Exporters <nemo-evaluator-launcher/exporters/overview>
:::

:::{toctree}
:caption: NeMo Evaluator Core
:hidden:

About Core <nemo-evaluator/index>
Python API <nemo-evaluator/workflows/python-api>
Container Workflows <nemo-evaluator/workflows/using_containers>
API Reference <nemo-evaluator/reference/api>
CLI Reference <nemo-evaluator/reference/cli>
Container Reference <nemo-evaluator/reference/containers>
Extending <nemo-evaluator/extending/framework_definition_file>
:::

:::{toctree}
:caption: Tutorials
:hidden:

About Tutorials <tutorials/index>
:::

:::{toctree}
:caption: Evaluation
:hidden:

About Model Evaluation <evaluation/index>
Run Evals <evaluation/run-evals/index>
Custom Task Configuration <evaluation/custom-tasks>
Benchmark Catalog <evaluation/benchmarks>
Troubleshooting <troubleshooting/index>
:::

:::{toctree}
:caption: Model Deployment
:hidden:

About Model Deployment <deployment/index>
PyTriton Backend <deployment/pytriton>
Ray Serve Deployment <deployment/ray-serve>
Evaluation Adapters <deployment/adapters/index>
:::

:::{toctree}
:caption: Troubleshooting
:hidden:

About Troubleshooting <troubleshooting/index>
Installation Issues <troubleshooting/installation-issues>
Authentication <troubleshooting/authentication>
Deployment Issues <troubleshooting/deployment-issues>
Configuration <troubleshooting/configuration-issues>
Performance <troubleshooting/performance-issues>
Debugging & Best Practices <troubleshooting/debugging-guide>
:::

:::{toctree}
:caption: References
:hidden:

About References <references/index>
Eval Parameters <evaluation/parameters>
Eval Utils <references/evaluation-utils>
API Documentation <apidocs/index.rst>
:::
