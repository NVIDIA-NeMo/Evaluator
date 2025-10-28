(about-release-notes)=

# Release Notes

## NeMo Evaluator SDK — General Availability (0.1.0)

NVIDIA is excited to announce the general availability of NeMo Evaluator SDK, an open-source platform for robust, reproducible, and scalable evaluation of large language models.

### Overview

NeMo Evaluator SDK provides a comprehensive solution for AI model evaluation and benchmarking, enabling researchers, ML engineers, and organizations to assess model performance across diverse capabilities including reasoning, code generation, function calling, and safety. The platform consists of two core libraries:

- **{ref}`nemo-evaluator <lib-core>`**: The core evaluation engine that manages interactions between evaluation harnesses and models being tested
- **{ref}`nemo-evaluator-launcher <lib-launcher>`**: The orchestration layer providing unified CLI and programmatic interfaces for multi-backend execution

### Key Features

**Reproducibility by Default**: All configurations, random seeds, and software provenance are captured automatically for auditable and repeatable evaluations.

**Scale Anywhere**: Run evaluations from a local machine to a Slurm cluster or cloud-native backends without changing your workflow.

**State-of-the-Art Benchmarking**: Access a comprehensive suite of over 100 benchmarks from 21+ popular open-source evaluation harnesses, including popular frameworks such as lm-evaluation-harness, bigcode-evaluation-harness, simple-evals, and specialized tools for safety, function calling, and agentic AI evaluation.

**Extensible and Customizable**: Integrate new evaluation harnesses, add custom benchmarks with proprietary data, and define custom result exporters for existing MLOps tooling.

**OpenAI-Compatible API Support**: Evaluate any model that exposes an OpenAI-compatible endpoint, including hosted services (build.nvidia.com), self-hosted solutions (NVIDIA NIM, vLLM, TensorRT-LLM), and models trained with NeMo framework.

**Containerized Execution**: All evaluations run in open-source Docker containers for auditable and trustworthy results, with pre-built containers available through the NVIDIA NGC catalog.
