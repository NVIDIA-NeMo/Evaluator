# NeMo Evaluator Containers

NeMo Evaluator provides a collection of specialized containers for different evaluation frameworks and tasks. Each container is optimized and tested to work seamlessly with NVIDIA hardware and software stack, providing consistent, reproducible environments for AI model evaluation.

## NGC Container Catalog

| Container | Description | NGC Catalog | Latest Tag |
|-----------|-------------|-------------|------------|
| **agentic_eval** | Agentic AI evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) | `25.07.3` |
| **rag_retriever_eval** | RAG system evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) | `25.07.3` |
| **simple-evals** | Basic evaluation tasks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) | `25.07.3` |
| **lm-evaluation-harness** | Language model benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) | `25.07.3` |
| **bigcode-evaluation-harness** | Code generation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) | `25.07.3` |
| **mtbench** | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) | `25.07.1` |
| **helm** | Holistic evaluation framework | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) | `25.07.2` |
| **tooltalk** | Tool usage evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) | `25.07.1` |
| **bfcl** | Function calling evaluation | [Link](https://catalog.ngc.nvidia.com/teams/eval-factory/containers/bfcl) | `25.07.3` |
| **garak** | Security and robustness testing | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) | `25.07.1` |
| **safety-harness** | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) | `25.07.3` |
| **vlmevalkit** | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) | `25.07.1` |

---

## Container Categories

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Language Models
:link: language-models
:link-type: doc

Containers for evaluating large language models across academic benchmarks and custom tasks.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: code-generation
:link-type: doc

Specialized containers for evaluating code generation and programming capabilities.
:::

:::{grid-item-card} {octicon}`eye;1.5em;sd-mr-1` Vision-Language
:link: vision-language
:link-type: doc

Multimodal evaluation containers for vision-language understanding and reasoning.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Safety & Security
:link: safety-security
:link-type: doc

Containers focused on safety evaluation, bias detection, and security testing.
:::

::::

---

## Quick Start

### Basic Container Usage

```bash
# Pull a container
docker pull nvcr.io/nvidia/eval-factory/<container-name>:<tag>

# Example: Pull simple-evals container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3

# Run with GPU support
docker run --gpus all -it nvcr.io/nvidia/eval-factory/<container-name>:<tag>
```

### Prerequisites

- Docker or NVIDIA Container Toolkit
- NVIDIA GPU (for GPU-accelerated evaluation)
- Sufficient disk space for models and datasets

For detailed usage instructions, see the [Container Workflows](../workflows/using_containers.md) guide.

:::{toctree}
:caption: Container Reference
:hidden:

Language Models <language-models>
Code Generation <code-generation>
Vision-Language <vision-language>
Safety & Security <safety-security>
Specialized Tools <specialized-tools>
:::
