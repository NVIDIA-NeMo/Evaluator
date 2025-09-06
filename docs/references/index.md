# References

Comprehensive reference documentation for NeMo Evaluator APIs, functions, and configuration options.

## API References

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Launcher API
:link: ../libraries/nemo-evaluator-launcher/api
:link-type: doc
Complete Python API reference for programmatic evaluation workflows and job management.
:::

:::{grid-item-card} {octicon}`command-palette;1.5em;sd-mr-1` CLI Reference
:link: ../libraries/nemo-evaluator-launcher/cli
:link-type: doc
Comprehensive command-line interface reference with all commands, options, and examples.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Schema
:link: ../libraries/nemo-evaluator-launcher/configuration
:link-type: doc
Complete configuration reference with examples for all executors and deployment types.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment API
:link: deployment-api
:link-type: ref
Complete reference for the deploy() function with all parameters, examples, and configuration options.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Evaluation Utilities
:link: evaluation-utils
:link-type: ref
Reference for evaluation discovery, health checking, and utility functions.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Auto-Generated API Docs
:link: ../apidocs/index
:link-type: ref
Sphinx-generated API documentation for all modules and classes.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Evaluation Parameters
:link: ../evaluation/parameters
:link-type: ref
Complete guide to evaluation parameters, optimization settings, and configuration patterns.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Benchmark Catalog
:link: ../evaluation/benchmarks
:link-type: ref
Comprehensive catalog of 100+ benchmarks across 18 evaluation harnesses.
:::

::::

## Contributing to NeMo Evaluator

### Getting Started
1. **Fork the Repository**: Start by forking the [NeMo Eval GitHub repository](https://github.com/NVIDIA-NeMo/Eval)
2. **Set Up Development Environment**: Follow the setup instructions for local development
3. **Review Contributing Guidelines**: Read the [Contributing Guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md)

### Development Areas
- **Core Evaluation Framework**: Enhance the evaluation engine and harness integrations
- **Deployment Backends**: Improve or add new serving backend options
- **Adapter System**: Extend the interceptor pipeline capabilities
- **Documentation**: Help improve guides, examples, and API documentation

## Architecture Overview

NeMo Evaluator is built with a modular architecture:

- **Core API Layer**: Primary interface for deployment and evaluation operations
- **Backend Adapters**: Pluggable deployment backends (PyTriton, Ray Serve)
- **Evaluation Harnesses**: Integration layer for external evaluation frameworks
- **Interceptor System**: Flexible middleware for request/response processing

## Project Governance

**Maintainer**: NVIDIA Corporation  
**License**: Apache License 2.0  
**Contact**: [nemo-toolkit@nvidia.com](mailto:nemo-toolkit@nvidia.com)  
**Repository**: [GitHub - NVIDIA-NeMo/Eval](https://github.com/NVIDIA-NeMo/Eval)

