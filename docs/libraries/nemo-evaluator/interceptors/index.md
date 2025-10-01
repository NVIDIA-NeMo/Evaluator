(nemo-evaluator-interceptors)=

# Interceptors

The interceptor system is the core innovation of NeMo Evaluator, providing sophisticated **adapter and interceptor architecture** for fine-grained control over request and response processing during model evaluation.

## Overview

The adapter system transforms simple model API calls into sophisticated evaluation workflows through a configurable pipeline of **interceptors**. Each interceptor can inspect, modify, or augment requests and responses as they flow through the evaluation process.

```{mermaid}
graph LR
    A[Evaluation Request] --> B[Adapter System]
    B --> C[Interceptor Pipeline]
    C --> D[Model API]
    D --> E[Response Pipeline]
    E --> F[Processed Response]
    
    subgraph "Request Processing"
        C --> G[System Message]
        G --> H[Payload Modifier]
        H --> I[Request Logging]
        I --> J[Caching Check]
        J --> K[Endpoint Call]
    end
    
    subgraph "Response Processing"
        E --> L[Response Logging]
        L --> M[Reasoning Extraction]
        M --> N[Progress Tracking]
        N --> O[Cache Storage]
    end
    
    style B fill:#f3e5f5
    style C fill:#e1f5fe
    style E fill:#e8f5e8
```

Configure request/response interceptors to add logging, caching, and custom processing to your evaluations.

## Core Interceptors

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`log;1.5em;sd-mr-1` Request & Response Logging
:link: logging
:link-type: doc

Log incoming requests and outgoing responses for debugging and analysis.
:::

:::{grid-item-card} {octicon}`cache;1.5em;sd-mr-1` Caching
:link: caching
:link-type: doc

Cache requests and responses to improve performance and reduce API calls.
:::

::::

## Specialized Interceptors

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`comment;1.5em;sd-mr-1` System Messages
:link: system-messages
:link-type: doc

Modify system messages and prompts in requests.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Payload Modification
:link: payload-modification
:link-type: doc

Add, remove, or modify request parameters.
:::

:::{grid-item-card} {octicon}`brain;1.5em;sd-mr-1` Reasoning
:link: reasoning
:link-type: doc

Handle reasoning tokens and track reasoning metrics.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Progress Tracking
:link: progress-tracking
:link-type: doc

Track evaluation progress and status updates.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Endpoint
:link: endpoint
:link-type: doc

Route requests to model APIs with retry logic and error handling.
:::

::::

## Process Post-Evaluation Results

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`report;1.5em;sd-mr-1` Post-Evaluation Hooks
:link: post-evaluation-hooks
:link-type: doc

Run additional processing, reporting, or cleanup after evaluations complete.
:::

::::

:::{toctree}
:caption: Interceptors
:hidden:

Request & Response Logging <logging>
Caching <caching>
System Messages <system-messages>
Payload Modification <payload-modification>
Reasoning <reasoning>
Progress Tracking <progress-tracking>
Endpoint <endpoint>
Post-Evaluation Hooks <post-evaluation-hooks>
:::
