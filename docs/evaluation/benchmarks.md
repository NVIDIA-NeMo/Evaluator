(eval-benchmarks)=

# Benchmark Catalog

Comprehensive catalog of 100+ benchmarks across 18 evaluation harnesses, all available through NGC containers and the NeMo Evaluator platform.


## Overview

NeMo Evaluator provides access to 100+ benchmarks through pre-built NGC containers and the unified launcher CLI. Each container specializes in different evaluation domains while maintaining consistent interfaces and reproducible results.

## Available via Launcher

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter by category (if supported)
nv-eval ls tasks --filter reasoning
nv-eval ls tasks --filter safety
nv-eval ls tasks --filter coding
```

## Benchmark Categories

###  **Academic and Reasoning**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **simple-evals** | MMLU Pro, GSM8K, ARC Challenge | Core academic benchmarks | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) |
| **lm-evaluation-harness** | MMLU, HellaSwag, TruthfulQA, PIQA | Language model evaluation suite | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) |
| **hle** | Humanity's Last Exam | Multi-modal benchmark at the frontier of human knowledge | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle) |
| **ifbench** | Instruction Following Benchmark | Precise instruction following evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench) |
| **mmath** | Multilingual Mathematical Reasoning | Math reasoning across multiple languages | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath) |
| **mtbench** | MT-Bench | Multi-turn conversation evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) |

**Example Usage:**
```bash
# Run academic benchmark suite
nv-eval run \
    --config-dir examples \
    --config-name academic_benchmark_suite \
    -o 'evaluation.tasks=["mmlu_pro", "gsm8k", "arc_challenge"]'
```

###  **Code Generation**  
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **bigcode-evaluation-harness** | HumanEval, MBPP, APPS | Code generation and completion | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) |
| **livecodebench** | Live coding contests from LeetCode, AtCoder, CodeForces | Contamination-free coding evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench) |
| **scicode** | Scientific research code generation | Scientific computing and research | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode) |

**Example Usage:**
```bash
# Run code generation evaluation
nv-eval run \
    --config-dir examples \
    --config-name coding_evaluation \
    -o 'evaluation.tasks=["humaneval", "mbpp"]'
```

###  **Safety and Security**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **safety-harness** | Toxicity, bias, alignment tests | Safety and bias evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) |
| **garak** | Prompt injection, jailbreaking | Security vulnerability scanning | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) |

**Example Usage:**
```bash
# Run comprehensive safety evaluation
nv-eval run \
    --config-dir examples \
    --config-name comprehensive_safety \
    -o 'evaluation.tasks=["toxicity", "bias_detection", "jailbreak_resistance"]'
```

###  **Function Calling and Agentic AI**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **bfcl** | Berkeley Function Calling Leaderboard | Function calling evaluation | [Link](https://catalog.ngc.nvidia.com/teams/eval-factory/containers/bfcl) |
| **agentic_eval** | Tool usage, planning tasks | Agentic AI evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) |
| **tooltalk** | Tool interaction evaluation | Tool usage assessment | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) |

###  **Vision-Language Models**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **vlmevalkit** | VQA, image captioning, visual reasoning | Vision-language model evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) |

###  **Retrieval and RAG**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **rag_retriever_eval** | Document retrieval, context relevance | RAG system evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) |

###  **Domain-Specific**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **helm** | Medical AI evaluation (MedHELM) | Healthcare-specific benchmarking | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) |

## Container Details

For detailed specifications of each container, see {ref}`nemo-evaluator-containers`.

### Quick Container Access

Pull and run any evaluation container directly:

```bash
# Academic benchmarks
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}

# Code generation
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}

# Safety evaluation
docker pull nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }}
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }}
```

### Available Tasks by Container

For a complete list of available tasks in each container:

```bash
# List tasks in any container
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} eval-factory ls

# Or use the launcher for unified access
nv-eval ls tasks
```

## Migration from Legacy Framework

If you're migrating from the legacy framework-based approach:

### **Old Approach** (Deprecated)
```python
from nemo_eval.utils.base import list_available_evaluations
available_tasks = list_available_evaluations()
```

### **New Approach** (Recommended)
```bash
# Use launcher for unified access
nv-eval ls tasks
nv-eval run --config-dir examples --config-name local_mmlu_evaluation
```

For detailed migration guidance, see {ref}`integration-patterns`.

## Next Steps

- **Start Evaluating**: Use {ref}`launcher-quickstart` for immediate access to all benchmarks
- **Container Details**: Browse the complete {ref}`nemo-evaluator-containers` for specifications
- **Custom Benchmarks**: Learn to {ref}`framework-definition-file`
- **Advanced Usage**: Explore [Multi-Backend Execution](../libraries/nemo-evaluator-launcher/executors/index) for scale
