(nemo-evaluator-interceptors)=

# Interceptors

Interceptors provide fine-grained control over request and response processing during model evaluation through a configurable pipeline architecture.

## Overview

The adapter system processes model API calls through a configurable pipeline of interceptors. Each interceptor can inspect, modify, or augment requests and responses as they flow through the evaluation process.

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

## Request Interceptors

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`comment;1.5em;sd-mr-1` System Messages
:link: system-messages
:link-type: doc

Modify system messages in requests.
:::

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Payload Modification
:link: payload-modification
:link-type: doc

Add, remove, or modify request parameters.
:::

:::{grid-item-card} {octicon}`sign-in;1.5em;sd-mr-1` Request Logging
:link: request-logging
:link-type: doc

Logs requests for debugging, analysis, and audit purposes.
:::

::::

## Request-Response Interceptors

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cache;1.5em;sd-mr-1` Caching
:link: caching
:link-type: doc

Cache requests and responses to improve performance and reduce API calls.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` Endpoint
:link: endpoint
:link-type: doc

Communicates with the model endpoint.
:::

::::

## Response 

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`sign-out;1.5em;sd-mr-1` Response Logging
:link: response-logging
:link-type: doc

Logs responses for debugging, analysis, and audit purposes.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Progress Tracking
:link: progress-tracking
:link-type: doc

Track evaluation progress and status updates.
:::

:::{grid-item-card} {octicon}`alert;1.5em;sd-mr-1` Raising on Client Errors
:link: raise-client-error
:link-type: doc

Allows to fail fast on non-retryable client errors
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Reasoning
:link: reasoning
:link-type: doc

Handle reasoning tokens and track reasoning metrics.
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Response Statistics
:link: response-stats
:link-type: doc

Collects statistics from API responses for metrics collection and analysis.
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

System Messages <system-messages>
Payload Modification <payload-modification>
Request Logging <request-logging>
Caching <caching>
Endpoint <endpoint>
Response Logging <response-logging>
Progress Tracking <progress-tracking>
Raising on Client Errors <raise-client-error>
Reasoning <reasoning>
Response Statistics <response-stats>
Post-Evaluation Hooks <post-evaluation-hooks>
:::
