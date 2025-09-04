(deployment-overview)=

# Model Deployment

Deploy NeMo Framework models for evaluation using different serving backends optimized for various use cases.

## Overview

NeMo Eval supports multiple deployment backends to accommodate different evaluation scenarios, from single-GPU deployments to multi-instance distributed setups. Choose the deployment method that best fits your hardware and evaluation requirements.

## Backend Options

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

### Choosing a Deployment Backend

For backend selection guidance, scaling modes, and adapter placement, refer to {ref}`deployment-concepts`.

## Evaluation Adapters

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
