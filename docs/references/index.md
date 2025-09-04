# References

Comprehensive reference documentation for NeMo Eval APIs, functions, and configuration options.

## API References

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

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

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Reference
:link: ../evaluation/parameters
:link-type: ref
Complete guide to evaluation parameters, optimization settings, and configuration patterns.
:::

::::

## Contributing to NeMo Eval

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

NeMo Eval is built with a modular architecture:

- **Core API Layer**: Primary interface for deployment and evaluation operations
- **Backend Adapters**: Pluggable deployment backends (PyTriton, Ray Serve)
- **Evaluation Harnesses**: Integration layer for external evaluation frameworks
- **Interceptor System**: Flexible middleware for request/response processing

## Project Governance

**Maintainer**: NVIDIA Corporation  
**License**: Apache License 2.0  
**Contact**: [nemo-toolkit@nvidia.com](mailto:nemo-toolkit@nvidia.com)  
**Repository**: [GitHub - NVIDIA-NeMo/Eval](https://github.com/NVIDIA-NeMo/Eval)

