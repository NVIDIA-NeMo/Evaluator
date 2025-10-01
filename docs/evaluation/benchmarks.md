(eval-benchmarks)=

# Benchmark Catalog

Comprehensive catalog of 100+ benchmarks across 18 evaluation harnesses, all available through NGC containers and the NeMo Evaluator platform.


## Overview

NeMo Evaluator provides access to 100+ benchmarks through pre-built NGC containers and the unified launcher CLI. Each container specializes in different evaluation domains while maintaining consistent interfaces and reproducible results.

## Available via Launcher

```bash
# List all available benchmarks
nv-eval ls tasks

# Output as JSON for programmatic filtering
nv-eval ls tasks --json
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
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=[{name: mmlu_pro}, {name: gsm8k}, {name: arc_challenge}]'
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
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=[{name: humaneval}, {name: mbpp}]'
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
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=[{name: aegis_v2}, {name: garak}]'
```

###  **Function Calling and Agentic AI**
| Container | Benchmarks | Description | NGC Catalog |
|-----------|------------|-------------|-------------|
| **bfcl** | Berkeley Function Calling Leaderboard | Function calling evaluation | [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) |
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

## Integration Patterns

NeMo Evaluator provides multiple integration options to fit your workflow:

```bash
# Launcher CLI (recommended for most users)
nv-eval ls tasks
nv-eval run --config-dir examples --config-name local_mmlu_evaluation

# Container direct execution
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} eval-factory ls

# Python API (for programmatic control)
# See the Python API documentation for details
```

## Next Steps

- **Start Evaluating**: Use {ref}`launcher-quickstart` for immediate access to all benchmarks
- **Container Details**: Browse the complete {ref}`nemo-evaluator-containers` for specifications
- **Custom Benchmarks**: Learn to create custom evaluations with {ref}`framework-definition-file`
- **Advanced Usage**: Explore {ref}`executors <lib-launcher>` for multi-backend execution at scale
