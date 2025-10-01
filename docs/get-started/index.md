(get-started-overview)=

# Get Started

## Before You Start

Before you begin, make sure you have:

- **Python Environment**: Python 3.10 or higher (up to 3.13)
- **OpenAI-Compatible Endpoint**: Hosted or self-deployed model API
- **Docker**: For container-based evaluation workflows (optional)
- **NVIDIA GPU**: For local model deployment (optional)

---

## Quick Start Path

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: gs-install
:link-type: ref
Install NeMo Eval and set up your evaluation environment with all necessary dependencies.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: gs-quickstart
:link-type: ref
Deploy your first model and run a simple evaluation in just a few minutes.
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Integration Patterns
:link: integration-patterns
:link-type: doc
Learn advanced integration patterns for the three-tier architecture.
:::

::::

## Entry Point Decision Guide

NeMo Evaluator provides three primary entry points, each designed for different user needs and workflows. Use this guide to choose the right approach for your use case.

```{mermaid}
flowchart TD
    A[I need to evaluate AI models] --> B{What's your primary goal?}
    
    B -->|Quick evaluations with minimal setup| C[NeMo Evaluator Launcher]
    B -->|Custom integrations and workflows| D[NeMo Evaluator Core]
    B -->|Direct container control| E[Direct Container Usage]
    
    C --> C1[ Unified CLI interface<br/> Multi-backend execution<br/> Built-in result export<br/> 100+ benchmarks ready]
    
    D --> D1[ Programmatic API control<br/> Custom evaluation workflows<br/> Adapter/interceptor system<br/> Framework extensions]
    
    E --> E1[ Maximum flexibility<br/> Custom container workflows<br/> Direct framework access<br/> Advanced users only]
    
    C1 --> F[Start with Launcher Quickstart]
    D1 --> G[Start with Core API Guide]
    E1 --> H[Start with Container Reference]
    
    style C fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#fff3e0
```

## What You'll Learn

By the end of this section, you'll be able to:

1. **Install and configure** NeMo Evaluator components for your needs
2. **Choose the right approach** from the three-tier architecture
3. **Run your first evaluation** using hosted or self-deployed endpoints
4. **Configure advanced features** like adapters and interceptors
5. **Integrate evaluations** into your ML workflows

## Typical Workflows

### **Launcher Workflow** (Most Users)
1. **Install** NeMo Evaluator Launcher
2. **Configure** endpoint and benchmarks in YAML
3. **Run** evaluations with single CLI command
4. **Export** results to MLflow, W&B, or local files

### **Core API Workflow** (Developers)
1. **Install** NeMo Evaluator Core library
2. **Configure** adapters and interceptors programmatically
3. **Integrate** into existing ML pipelines
4. **Customize** evaluation logic and processing

### **Container Workflow** (Container Users)
1. **Pull** pre-built evaluation containers
2. **Run** evaluations directly in isolated environments
3. **Mount** data and results for persistence
4. **Combine** with existing container orchestration
