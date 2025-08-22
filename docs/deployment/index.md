(deployment-overview)=

# Model Deployment

Deploy NeMo Framework models for evaluation using different serving backends optimized for various use cases.

## Overview

NeMo Eval supports multiple deployment backends to accommodate different evaluation scenarios, from single-GPU deployments to multi-instance distributed setups. Choose the deployment method that best fits your hardware and evaluation requirements.

## Deployment Options

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

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Evaluation Adapters
:link: adapters
:link-type: ref
Flexible request/response interceptors for custom processing, logging, and transformation during evaluation.
:::

::::

## Key Features

### Performance Optimizations
- CUDA graphs and flash decoding for optimized inference
- Multi-GPU and multi-node distributed computing
- Automatic load balancing across replicas

### API Compatibility
- OpenAI-compatible REST API endpoints
- Support for both completions (`/v1/completions`) and chat (`/v1/chat/completions`) endpoints
- Flexible adapter system with interceptor pipelines

### Monitoring & Validation
- Real-time health monitoring
- Endpoint validation and status checking
- Comprehensive logging and debugging support

## Choosing a Deployment Backend

| Use Case | Recommended Backend | Key Benefits |
|----------|-------------------|--------------|
| Production deployment | PyTriton | High performance, multi-node support |
| Accelerated evaluation | Ray Serve | Multi-instance, horizontal scaling |
| Custom processing | Adapters | Request/response transformation |


