(nemo-evaluator-containers)=

# NeMo Evaluator Containers

NeMo Evaluator provides a collection of specialized containers for different evaluation frameworks and tasks. Each container is optimized and tested to work seamlessly with NVIDIA hardware and software stack, providing consistent, reproducible environments for AI model evaluation.

## Container Categories

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Language Models
:link: language-models
:link-type: doc

Containers for evaluating large language models across academic benchmarks and custom tasks.
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Code Generation
:link: code-generation
:link-type: doc

Specialized containers for evaluating code generation and programming capabilities.
:::

:::{grid-item-card} {octicon}`eye;1.5em;sd-mr-1` Vision-Language
:link: vision-language
:link-type: doc

Multimodal evaluation containers for vision-language understanding and reasoning.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Safety & Security
:link: safety-security
:link-type: doc

Containers focused on safety evaluation, bias detection, and security testing.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Specialized Tools
:link: specialized-tools
:link-type: doc

Containers focused on agentic AI capabilities and advanced reasoning.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Efficiency
:link: efficiency
:link-type: doc

Containers for evaluating speed of input processing and output generation.
:::

::::

---

## Quick Start

### Basic Container Usage

```bash
# Pull a container
docker pull nvcr.io/nvidia/eval-factory/<container-name>:<tag>

# Example: Pull simple-evals container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}

# Run with GPU support
docker run -it nvcr.io/nvidia/eval-factory/<container-name>:<tag>
```

### Prerequisites

- Docker and NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU (for GPU-accelerated evaluation)
- Sufficient disk space for models and datasets

For detailed usage instructions, refer to the {ref}`cli-workflows` guide.

:::{toctree}
:caption: Container Reference
:hidden:

Language Models <language-models>
Code Generation <code-generation>
Vision-Language <vision-language>
Safety & Security <safety-security>
Specialized Tools <specialized-tools>
Efficiency <efficiency>
:::
