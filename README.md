<div align="center">

  <a href="https://github.com/NVIDIA-NeMo/Evaluator/blob/main/LICENSE">![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)</a>
  <a href="https://codecov.io/github/NVIDIA-NeMo/Evaluator">![codecov](https://codecov.io/github/NVIDIA-NeMo/Evaluator/graph/badge.svg)</a>
  <a href="https://pypi.org/project/nemo-evaluator/">![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green)</a>
  <a href="https://github.com/NVIDIA-NeMo/Evaluator/graphs/contributors">![Contributors](https://img.shields.io/github/contributors/NVIDIA-NeMo/Evaluator)</a>
  <a href="https://github.com/NVIDIA-NeMo/Evaluator/releases">![Release](https://img.shields.io/github/release/NVIDIA-NeMo/Evaluator)</a>
  <a href="https://pypi.org/project/nemo-evaluator/">![Open Source](https://badgen.net/badge/open%20source/❤/blue?icon=github)</a>

</div>

# NVIDIA NeMo Evaluator SDK

**Reproducible, scalable evaluation for large language models.** Run 100+ benchmarks across 18 evaluation harnesses with container-first architecture. Scale from laptop to multi-node clusters with unified configuration.

> *Part of the [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) software suite for managing the AI agent lifecycle.*

## What You Can Do

| Capability | Key Features | Details |
|------------|-------------|---------|
| **100+ Benchmarks** | Academic • Code Generation • Safety • Function Calling • Vision-Language • RAG | See sections below |
| **Multi-Backend Execution** | Local Docker • Slurm HPC • Lepton AI • Custom Cloud | [Launcher Docs](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/index.html) |
| **Container-First** | 18 NGC containers with locked dependencies and reproducible environments | [NGC Catalog](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=label%3A%22NSPECT-JL1B-TVGU%22) |
| **Extensible** | Custom benchmarks • Private datasets • Custom exporters | [Extension Guide](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/extending/index.html) |

## Quick Start

```bash
# Install the launcher
pip install nemo-evaluator-launcher

# Get an API key from build.nvidia.com
export NGC_API_KEY=<YOUR_API_KEY>

# Run your first evaluation
nemo-evaluator-launcher run \
  --config-dir packages/nemo-evaluator-launcher/examples \
  --config-name local_nvidia_nemotron_nano_9b_v2

# Check results
nemo-evaluator-launcher status <job_id>
```

**Full setup:** [Installation Guide](https://docs.nvidia.com/nemo/evaluator/latest/get-started/install.html) • [Quickstart](https://docs.nvidia.com/nemo/evaluator/latest/get-started/quickstart/index.html) • [Examples](packages/nemo-evaluator-launcher/examples/)

---

## Evaluation Capabilities

### Academic Benchmarks

Comprehensive suite for evaluating reasoning, knowledge, and language understanding across multiple domains.

| Harness | Benchmarks | Documentation |
|---------|-----------|---------------|
| **simple-evals** | MMLU Pro • GSM8K • MATH-500 • AIME 24/25 • GPQA-D • MGSM • SimpleQA | [Simple Evals](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals) |
| **lm-evaluation-harness** | MMLU • HellaSwag • TruthfulQA • ARC Challenge • PIQA • WinoGrande • IFEval | [LM Eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness) |
| **hle** | Humanity's Last Exam (multi-modal frontier benchmark) | [HLE](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle) |
| **ifbench** | Instruction Following Benchmark | [IFBench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench) |
| **mmath** | Multilingual math reasoning (10 languages: EN, ZH, AR, ES, FR, JA, KO, PT, TH, VI) | [MMath](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath) |
| **mtbench** | Multi-turn conversation evaluation | [MT-Bench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench) |

---

### Code Generation

Evaluate programming capabilities with industry-standard benchmarks and contamination-free datasets.

| Harness | Benchmarks | Documentation |
|---------|-----------|---------------|
| **bigcode-evaluation-harness** | HumanEval • HumanEval+ • MBPP • MBPP+ • Multiple (20 languages) | [BigCode](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness) |
| **livecodebench** | Live coding contests (LeetCode, AtCoder, CodeForces) • Versions v1-v6 | [LiveCodeBench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench) |
| **compute-eval** | CUDA code evaluation (CCCL, CUDA problems) | [Compute Eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/compute-eval) |
| **scicode** | Scientific research code generation | [SciCode](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode) |

---

### Safety and Security

Comprehensive safety assessment with specialized containers for content safety, security vulnerabilities, and bias evaluation.

| Harness | Capabilities | Documentation |
|---------|-------------|---------------|
| **safety-harness** | Content safety (Aegis v2, multilingual) • Bias evaluation (BBQ) • WildGuard | [Safety Harness](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness) |
| **garak** | Prompt injection • Jailbreaking • Security vulnerability scanning | [Garak](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak) |

---

### Function Calling and Agentic AI

Test tool usage, function calling accuracy, and agentic behaviors with specialized evaluation frameworks.

| Harness | Benchmarks | Documentation |
|---------|-----------|---------------|
| **bfcl** | Berkeley Function Calling Leaderboard v2 and v3 | [BFCL](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl) |
| **agentic_eval** | Tool call accuracy • Goal accuracy • Topic adherence • Answer accuracy | [Agentic Eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval) |
| **tooltalk** | Tool interaction evaluation | [ToolTalk](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk) |

---

### Vision-Language Models

Evaluate vision-language capabilities with VQA, chart understanding, OCR, and document analysis.

| Harness | Benchmarks | Documentation |
|---------|-----------|---------------|
| **vlmevalkit** | AI2D • ChartQA • OCRBench • SlideVQA | [VLMEvalKit](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit) |

---

### Specialized Domains

Domain-specific evaluation for RAG systems, medical AI, and performance benchmarking.

| Harness | Focus Area | Documentation |
|---------|-----------|---------------|
| **rag_retriever_eval** | Document retrieval • Context relevance • RAG system evaluation | [RAG Eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval) |
| **helm** | Medical AI evaluation (MedHELM suite) | [HELM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm) |
| **genai-perf** | Performance benchmarking (generation, summarization) | [GenAI Perf](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf) |
| **nemo-skills** | Science, math, and agentic benchmarks (AIME, BFCL, GPQA, HLE, LiveCodeBench) | [NeMo Skills](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo_skills) |

---

## Key Features

### Reproducibility by Design

Every evaluation is reproducible with automatic capture of configurations, random seeds, and software provenance.

**Container-First Architecture:**
- Pre-built NGC containers with locked dependencies
- Isolated execution environments
- Version-tagged releases for exact reproducibility
- Auditable and trustworthy results

```bash
# Run the same evaluation on different backends with identical results
nemo-evaluator-launcher run --config-name local_evaluation
nemo-evaluator-launcher run --config-name slurm_evaluation
nemo-evaluator-launcher run --config-name lepton_evaluation
```

---

### Scale Anywhere

Run evaluations from local development to production HPC clusters with unified configuration.

| Backend | Use Case | Setup |
|---------|----------|-------|
| **Local Docker** | Development • Quick iteration • Single-node evaluation | [Local Setup](https://docs.nvidia.com/nemo/evaluator/latest/get-started/quickstart/launcher.html) |
| **Slurm HPC** | Large-scale parallel evaluation • Multi-node clusters | See [Launcher Configuration](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/configuration/index.html) |
| **Lepton AI** | Cloud-native deployment • Serverless inference | See [Launcher Configuration](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/configuration/index.html) |

**Example configurations:** [packages/nemo-evaluator-launcher/examples/](packages/nemo-evaluator-launcher/examples/)

---

### Extensible Architecture

Add custom benchmarks, private datasets, and custom exporters to existing evaluation workflows.

**Extension Points:**
- **Custom Benchmarks**: Define new tasks with Framework Definition Files
- **Private Datasets**: Integrate proprietary evaluation data
- **Custom Exporters**: Connect to existing MLOps tooling
- **Adapter System**: Modify request/response processing with interceptors

```yaml
# Framework Definition File example
framework:
  name: my_custom_eval
  description: Custom evaluation for domain-specific tasks
  
evaluations:
  - name: domain_specific_task
    description: Evaluate domain-specific capabilities
    defaults:
      config:
        params:
          task: domain_task
          temperature: 0.0
```

**Learn more:** [Extension Guide](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/extending/index.html)

---

### Advanced Adapter System

Sophisticated request/response processing with interceptor architecture for caching, logging, and custom transformations.

**Available Interceptors:**
- **System Message**: Enforce consistent system prompts
- **Request/Response Logging**: Capture evaluation traces with configurable limits
- **Caching**: Reduce API costs with intelligent caching
- **Reasoning Cleanup**: Strip intermediate reasoning tokens
- **Progress Tracking**: Monitor evaluation progress in real-time

```yaml
# Adapter configuration example
target:
  api_endpoint:
    url: "http://localhost:8080/v1/completions/"
    model_id: "my-model"
    adapter_config:
      interceptors:
        - name: system_message
          config:
            system_message: "You are a helpful AI assistant."
        - name: caching
          config:
            cache_dir: "./evaluation_cache"
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
```

**Learn more:** [Core Library Documentation](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/index.html)

---

## Architecture

NeMo Evaluator SDK consists of two main components that work together:

### 1. NeMo Evaluator (Core Engine)

The evaluation core engine that manages interaction between evaluation harnesses and models.

**Key Capabilities:**
- Standardized evaluation workflows
- Adapter system for request/response processing
- Containerized benchmarks with NGC integration
- Programmatic Python API

```python
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
```

---

### 2. NeMo Evaluator Launcher (Orchestration)

The CLI and orchestration layer for managing evaluations across different backends.

**Key Capabilities:**
- Unified CLI interface
- Multi-backend execution (local, Slurm, Lepton AI)
- Job monitoring and lifecycle management
- Result export to MLflow, Weights & Biases, Google Sheets

```bash
# Launch evaluation on Slurm cluster
nemo-evaluator-launcher run \
  --config-dir packages/nemo-evaluator-launcher/examples \
  --config-name slurm_llama_3_1_8b_instruct

# Monitor status
nemo-evaluator-launcher status <job_id>

# Export results
nemo-evaluator-launcher export <job_id> --dest mlflow
```

---

## OpenAI API Compatibility

NeMo Evaluator works with any model exposing an OpenAI-compatible API endpoint.

**Supported Endpoint Types:**
- **`completions`**: Direct text completion (`/v1/completions`) for base models
- **`chat`**: Conversational interface (`/v1/chat/completions`) for instruction-tuned models
- **`vlm`**: Vision-language model endpoints with image inputs
- **`embedding`**: Embedding generation for retrieval evaluation

**Compatible Model Serving:**
- **Hosted**: NVIDIA Build • OpenAI • Anthropic • Cohere
- **Self-Hosted**: NVIDIA NIM • vLLM • TensorRT-LLM • NeMo Framework
- **Custom**: Any service implementing OpenAI API specification

---

## Result Export

First-class integration with popular MLOps platforms for experiment tracking and result sharing.

```bash
# Export to MLflow
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to Google Sheets
nemo-evaluator-launcher export <invocation_id> --dest gsheets
```

**Learn more:** [Exporters Guide](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/exporters/index.html)

---

## Evaluate NeMo Framework Checkpoints

Seamlessly integrate with NeMo Framework for checkpoint evaluation during model training.

```python
# Start NeMo Framework container
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash --gpus all \
  nvcr.io/nvidia/nemo:${TAG}

# Deploy a NeMo checkpoint
python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "/path/to/your/checkpoint" \
  --model_id megatron_model \
  --port 8080 --host 0.0.0.0

# Evaluate the deployed model
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")
results = evaluate(target_cfg=target, eval_cfg=config)
```

**Learn more:** See the [NeMo Evaluator Core documentation](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/index.html)

---

## Learn More

| Resource | Links |
|----------|-------|
| **Documentation** | [Main Docs](https://docs.nvidia.com/nemo/evaluator/latest/) • [Key Features](https://docs.nvidia.com/nemo/evaluator/latest/about/key-features.html) • [Concepts](https://docs.nvidia.com/nemo/evaluator/latest/about/concepts/index.html) |
| **Get Started** | [Installation](https://docs.nvidia.com/nemo/evaluator/latest/get-started/install.html) • [Quickstart](https://docs.nvidia.com/nemo/evaluator/latest/get-started/quickstart/index.html) |
| **Libraries** | [Launcher](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/index.html) • [Core](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator/index.html) |
| **Configuration** | [Launcher Configuration](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/configuration/index.html) • [Executors](https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/configuration/executors/index.html) |
| **Community** | [GitHub Discussions](https://github.com/NVIDIA-NeMo/Evaluator/discussions) • [Issues](https://github.com/NVIDIA-NeMo/Evaluator/issues) |

---

## Contribute

We welcome community contributions! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting pull requests, reporting issues, and suggesting features.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
