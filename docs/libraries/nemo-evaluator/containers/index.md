(nemo-evaluator-containers)=

# NeMo Evaluator Containers

NeMo Evaluator provides a collection of specialized containers for different evaluation frameworks and tasks. Each container is optimized and tested to work seamlessly with NVIDIA hardware and software stack, providing consistent, reproducible environments for AI model evaluation.

## NGC Container Catalog

```{list-table}
:header-rows: 1
:widths: 20 25 15 15 25

* - Container
  - Description
  - NGC Catalog
  - Latest Tag
  - Key Benchmarks
* - **agentic_eval**
  - Agentic AI evaluation framework
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)
  - `{{ docker_compose_latest }}`
  - agentic_eval_answer_accuracy, agentic_eval_goal_accuracy_with_reference, agentic_eval_goal_accuracy_without_reference, agentic_eval_topic_adherence, agentic_eval_tool_call_accuracy
* - **bfcl**
  - Function calling evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl)
  - `{{ docker_compose_latest }}`
  - bfclv2, bfclv2_ast, bfclv2_ast_prompting, bfclv3, bfclv3_ast, bfclv3_ast_prompting
* - **bigcode-evaluation-harness**
  - Code generation evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness)
  - `{{ docker_compose_latest }}`
  - humaneval, humanevalplus, mbpp, mbppplus
* -  **compute-eval**
  - CUDA code evaluation
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/compute-eval)
  - `{{ docker_compose_latest }}`
  - cccl_problems, combined_problems, cuda_problems 
* - **garak**
  - Security and robustness testing
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)
  - `{{ docker_compose_latest }}`
  - garak
* - **genai-perf** 
  - GenAI performance benchmarking 
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf)
  - `{{ docker_compose_latest }}`
  - genai_perf_generation, genai_perf_summarization
* - **helm**
  - Holistic evaluation framework
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)
  - `{{ docker_compose_latest }}`
  - aci_bench, ehr_sql, head_qa, med_dialog_healthcaremagic
* - **hle**
  - Academic knowledge and problem solving
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle)
  - `{{ docker_compose_latest }}`
  - hle
* - **ifbench**
  - Instruction following evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench)
  - `{{ docker_compose_latest }}`
  - ifbench
* - **livecodebench**
  - Live coding evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench)
  - `{{ docker_compose_latest }}`
  - livecodebench_0724_0125, livecodebench_0824_0225
* - **lm-evaluation-harness**
  - Language model benchmarks
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)
  - `{{ docker_compose_latest }}`
  - mmlu, gsm8k, hellaswag, arc_challenge, truthfulqa
* - **mmath**
  - Multilingual math reasoning
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath)
  - `{{ docker_compose_latest }}`
  - mmath_ar, mmath_en, mmath_es, mmath_fr, mmath_zh
* - **mtbench**
  - Multi-turn conversation evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)
  - `{{ docker_compose_latest }}`
  - mtbench, mtbench-cor1
* - **nemo-skills** 
  - Language model benchmarks (science, math, agentic)
  - [Link](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills)
  - `{{ docker_compose_latest }}`
  - ns_aime2024, ns_aime2025, ns_aime2025_ef, ns_bfcl_v3, ns_gpqa, ns_gpqa_ef, ns_hle, ns_livecodebench, ns_mmlu, ns_mmlu_pro 
* - **rag_retriever_eval**
  - RAG system evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval)
  - `{{ docker_compose_latest }}`
  - RAG, Retriever
* - **safety-harness**
  - Safety and bias evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)
  - `{{ docker_compose_latest }}`
  - aegis_v2
* - **scicode**
  - Coding for scientific research
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode)
  - `{{ docker_compose_latest }}`
  - scicode, scicode_background
* - **simple-evals**
  - Basic evaluation tasks
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)
  - `{{ docker_compose_latest }}`
  - mmlu, mmlu_pro, gpqa_diamond, humaneval, math_test_500
* - **tooltalk**
  - Tool usage evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)
  - `{{ docker_compose_latest }}`
  - tooltalk
* - **vlmevalkit**
  - Vision-language model evaluation
  - [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)
  - `{{ docker_compose_latest }}`
  - ai2d_judge, chartqa, ocrbench, slidevqa
```

---

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
:::
