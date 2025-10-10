(eval-benchmarks)=

# Benchmark Catalog

Comprehensive catalog of 100+ benchmarks across 18 evaluation harnesses, all available through NGC containers and the NeMo Evaluator platform.


## Overview

NeMo Evaluator provides access to benchmarks across multiple domains through pre-built NGC containers and the unified launcher CLI. Each container specializes in different evaluation domains while maintaining consistent interfaces and reproducible results.

## Available via Launcher

```{literalinclude} _snippets/commands/list_tasks.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

## Choosing Benchmarks for Academic Research

:::{admonition} Benchmark Selection Guide
:class: tip

**For Language Understanding & General Knowledge**:
Recommended suite for comprehensive model evaluation:
- `mmlu_pro` - Expert-level knowledge across 14 domains
- `arc_challenge` - Complex reasoning and science questions
- `hellaswag` - Commonsense reasoning about situations
- `truthfulqa` - Factual accuracy vs. plausibility

```bash
nv-eval run \
    --config-dir examples \
    --config-name local_academic_suite \
    -o 'evaluation.tasks=["mmlu_pro", "arc_challenge", "hellaswag", "truthfulqa"]'
```

**For Mathematical & Quantitative Reasoning**:
- `gsm8k` - Grade school math word problems
- `math` - Competition-level mathematics
- `mgsm` - Multilingual math reasoning

**For Instruction Following & Alignment**:
- `ifeval` - Precise instruction following
- `gpqa_diamond` - Graduate-level science questions
- `mtbench` - Multi-turn conversation quality

**See benchmark details below** for complete task descriptions and requirements.
:::

## Benchmark Categories

###  **Academic and Reasoning**

```{list-table}
:header-rows: 1
:widths: 20 30 30 20

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **simple-evals**
  - MMLU Pro, GSM8K, ARC Challenge
  - Core academic benchmarks
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)
* - **lm-evaluation-harness**
  - MMLU, HellaSwag, TruthfulQA, PIQA
  - Language model evaluation suite
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)
* - **hle**
  - Humanity's Last Exam
  - Multi-modal benchmark at the frontier of human knowledge
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle)
* - **ifbench**
  - Instruction Following Benchmark
  - Precise instruction following evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench)
* - **mmath**
  - Multilingual Mathematical Reasoning
  - Math reasoning across multiple languages
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath)
* - **mtbench**
  - MT-Bench
  - Multi-turn conversation evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)
```

**Example Usage:**
```bash
# Run academic benchmark suite
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["mmlu_pro", "gsm8k", "arc_challenge"]'
```

**Python API Example:**
```python
# Evaluate multiple academic benchmarks
academic_tasks = ["mmlu_pro", "gsm8k", "arc_challenge"]
for task in academic_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"./results/{task}/",
        params=ConfigParams(temperature=0.01, parallelism=4)
    )
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
```

###  **Code Generation**

```{list-table}
:header-rows: 1
:widths: 25 30 30 15

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **bigcode-evaluation-harness**
  - HumanEval, MBPP, APPS
  - Code generation and completion
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness)
* - **livecodebench**
  - Live coding contests from LeetCode, AtCoder, CodeForces
  - Contamination-free coding evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench)
* - **scicode**
  - Scientific research code generation
  - Scientific computing and research
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode)
```

**Example Usage:**
```bash
# Run code generation evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["humaneval", "mbpp"]'
```

###  **Safety and Security**

```{list-table}
:header-rows: 1
:widths: 25 35 25 15

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **safety-harness**
  - Toxicity, bias, alignment tests
  - Safety and bias evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)
* - **garak**
  - Prompt injection, jailbreaking
  - Security vulnerability scanning
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)
```

**Example Usage:**
```bash
# Run comprehensive safety evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["aegis_v2", "garak"]'
```

###  **Function Calling and Agentic AI**

```{list-table}
:header-rows: 1
:widths: 25 30 30 15

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **bfcl**
  - Berkeley Function Calling Leaderboard
  - Function calling evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl)
* - **agentic_eval**
  - Tool usage, planning tasks
  - Agentic AI evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)
* - **tooltalk**
  - Tool interaction evaluation
  - Tool usage assessment
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)
```

###  **Vision-Language Models**

```{list-table}
:header-rows: 1
:widths: 25 35 25 15

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **vlmevalkit**
  - VQA, image captioning, visual reasoning
  - Vision-language model evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)
```

###  **Retrieval and RAG**

```{list-table}
:header-rows: 1
:widths: 25 35 25 15

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **rag_retriever_eval**
  - Document retrieval, context relevance
  - RAG system evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval)
```

###  **Domain-Specific**

```{list-table}
:header-rows: 1
:widths: 25 35 25 15

* - Container
  - Benchmarks
  - Description
  - NGC Catalog
* - **helm**
  - Medical AI evaluation (MedHELM)
  - Healthcare-specific benchmarking
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)
```

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
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} nemo-evaluator ls

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
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} nemo-evaluator ls

# Python API (for programmatic control)
# See the Python API documentation for details
```

## Benchmark Selection Best Practices

### For Academic Publications

**Recommended Core Suite**:
1. **MMLU Pro** or **MMLU** - Broad knowledge assessment
2. **GSM8K** - Mathematical reasoning
3. **ARC Challenge** - Scientific reasoning
4. **HellaSwag** - Commonsense reasoning
5. **TruthfulQA** - Factual accuracy

This suite provides comprehensive coverage across major evaluation dimensions.

### For Model Development

**Iterative Testing**:
- Start with `limit_samples=100` for quick feedback during development
- Run full evaluations before major releases
- Track metrics over time to measure improvement

**Configuration**:
```python
# Development testing
params = ConfigParams(
    limit_samples=100,      # Quick iteration
    temperature=0.01,       # Deterministic
    parallelism=4
)

# Production evaluation
params = ConfigParams(
    limit_samples=None,     # Full dataset
    temperature=0.01,       # Deterministic
    parallelism=8          # Higher throughput
)
```

### For Specialized Domains

- **Code Models**: Focus on `humaneval`, `mbpp`, `livecodebench`
- **Instruction Models**: Emphasize `ifeval`, `mtbench`, `gpqa_diamond`
- **Multilingual Models**: Include `arc_multilingual`, `hellaswag_multilingual`, `mgsm`
- **Safety-Critical**: Prioritize `safety-harness` and `garak` evaluations

## Next Steps

- **Quick Start**: See {ref}`evaluation-overview` for the fastest path to your first evaluation
- **Task-Specific Guides**: Explore {ref}`eval-run` for detailed evaluation workflows
- **Configuration**: Review {ref}`eval-parameters` for optimizing evaluation settings
- **Container Details**: Browse {ref}`nemo-evaluator-containers` for complete specifications
- **Custom Benchmarks**: Learn {ref}`framework-definition-file` for custom evaluations
