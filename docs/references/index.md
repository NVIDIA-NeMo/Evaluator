(development-overview)=

# Development

Resources and guides for contributing to and extending NeMo Eval.

## Overview

This section provides information for developers who want to contribute to NeMo Eval, extend its functionality, or understand its internal architecture. Whether you're fixing bugs, adding features, or building custom extensions, these resources will help you get started.

## Development Resources

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Documentation Guide
:link: documentation
:link-type: ref
Learn how to contribute to and maintain the NeMo Eval documentation, including style guidelines and build processes.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Documentation
:link: ../apidocs/index
:link-type: ref
Comprehensive API reference documentation for all NeMo Eval modules and functions.
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

