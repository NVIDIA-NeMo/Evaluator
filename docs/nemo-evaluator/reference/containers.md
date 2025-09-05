# Eval Factory Containers

This document provides information about the various evaluation containers available in the NVIDIA NGC catalog for Eval Factory. These containers are pre-built and ready to use for different types of AI model evaluation tasks.

## Overview

Eval Factory provides a collection of specialized containers for different evaluation frameworks and tasks. Each container is optimized and tested to work seamlessly with NVIDIA hardware and software stack.

## Available Containers

### 1. Agentic Evaluation Container
**NGC Catalog**: [agentic_eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)

**Description**: Container for evaluating agentic AI models that can perform complex, multi-step tasks and demonstrate reasoning capabilities.

**Use Cases**:
- Multi-step problem solving
- Tool usage evaluation
- Reasoning chain assessment
- Agent behavior analysis

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/agentic_eval:25.07.3
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `parallelism` | `null` |
| `judge_model_type` | `openai` |
| `judge_model_args` | `null` |
| `judge_sanity_check` | `True` |
| `metric_mode` | `"f1"` |
| `data_template_path` | `null` |

### 2. RAG Retriever Evaluation Container
**NGC Catalog**: [rag_retriever_eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval)

**Description**: Container for evaluating Retrieval-Augmented Generation (RAG) systems and their retrieval capabilities.

**Use Cases**:
- Document retrieval accuracy
- Context relevance assessment
- RAG pipeline evaluation
- Information retrieval performance

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/rag_retriever_eval:25.07.3
```

### 3. Simple-Evals Container
**NGC Catalog**: [simple-evals](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)

**Description**: Container for lightweight evaluation tasks and simple model assessments.

**Use Cases**:
- Simple question-answering evaluation
- Math and reasoning capabilities
- Basic Python coding

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `4096` |
| `temperature` | `0` |
| `top_p` | `1e-05` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `60` |
| `downsampling_ratio` | `None` |
| `add_system_prompt` | `False` |
| `custom_config` | `None` |
| `judge` | `{'url': None, 'model_id': None, 'api_key': None, 'backend': 'openai', 'request_timeout': 600, 'max_retries': 16, 'temperature': 0.0, 'top_p': 0.0001, 'max_tokens': 1024, 'max_concurrent_requests': None}` |

### 4. LM-Evaluation-Harness Container
**NGC Catalog**: [lm-evaluation-harness](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)

**Description**: Container based on the Language Model Evaluation Harness framework for comprehensive language model evaluation.

**Use Cases**:
- Standard NLP benchmarks
- Language model performance evaluation
- Multi-task assessment
- Academic benchmark evaluation

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/lm-evaluation-harness:25.07.3
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `None` |
| `temperature` | `1e-07` |
| `top_p` | `0.9999999` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `tokenizer` | `None` |
| `tokenizer_backend` | `None` |
| `downsampling_ratio` | `None` |
| `tokenized_requests` | `False` |

### 5. BigCode Evaluation Harness Container
**NGC Catalog**: [bigcode-evaluation-harness](https://catalog.ngc.nvidia.com/orgs/nvidia/eval-factory/containers/bigcode-evaluation-harness)

**Description**: Container specialized for evaluating code generation models and programming language models.

**Use Cases**:
- Code generation quality assessment
- Programming problem solving
- Code completion evaluation
- Software engineering task assessment

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:25.07.3
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `512` |
| `temperature` | `1e-07` |
| `top_p` | `0.9999999` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `do_sample` | `True` |
| `n_samples` | `1` |

### 6. MT-Bench Container
**NGC Catalog**: [mtbench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)

**Description**: Container for MT-Bench evaluation framework, designed for multi-turn conversation evaluation.

**Use Cases**:
- Multi-turn dialogue evaluation
- Conversation quality assessment
- Context maintenance evaluation
- Interactive AI system testing

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/mtbench:25.07.1
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `max_new_tokens` | `1024` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `judge` | `{'url': None, 'model_id': 'gpt-4', 'api_key': None, 'request_timeout': 60, 'max_retries': 16, 'temperature': 0.0, 'top_p': 0.0001, 'max_tokens': 2048}` |

### 7. HELM Container
**NGC Catalog**: [helm](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)

**Description**: Container for the Holistic Evaluation of Language Models (HELM) framework, with a focus on MedHELM - an extensible evaluation framework for assessing LLM performance for medical tasks.

**Use Cases**:
- Medical AI model evaluation
- Clinical task assessment
- Healthcare-specific benchmarking
- Diagnostic decision-making evaluation
- Patient communication assessment
- Medical knowledge evaluation

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/helm:25.07.2
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `parallelism` | `1` |
| `data_path` | `None` |
| `num_output_tokens` | `None` |
| `subject` | `None` |
| `condition` | `None` |
| `max_length` | `None` |
| `num_train_trials` | `None` |
| `subset` | `None` |
| `gpt_judge_api_key` | `GPT_JUDGE_API_KEY` |
| `llama_judge_api_key` | `LLAMA_JUDGE_API_KEY` |
| `claude_judge_api_key` | `CLAUDE_JUDGE_API_KEY` |

### 8. ToolTalk Container
**NGC Catalog**: [tooltalk](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)

**Description**: Container for evaluating AI models' ability to use tools and APIs effectively.

**Use Cases**:
- Tool usage evaluation
- API interaction assessment
- Function calling evaluation
- External tool integration testing

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/tooltalk:25.07.1
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |

### 9. BFCL Container
**NGC Catalog**: [bfcl](https://catalog.ngc.nvidia.com/teams/eval-factory/containers/bfcl)

**Description**: Container for Berkeley Function-Calling Leaderboard evaluation framework.

**Use Cases**:
- Tool usage evaluation
- Multi-turn interactions
- Native support for function/tool calling
- Function calling evaluation

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/bfcl:25.07.3
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `parallelism` | `10` |
| `native_calling` | `False` |
| `custom_dataset` | `{'path': None, 'format': None, 'data_template_path': None}` |

### 10. Garak Container
**NGC Catalog**: [garak](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/garak)

**Description**: Container for security and robustness evaluation of AI models.

**Use Cases**:
- Security testing
- Adversarial attack evaluation
- Robustness assessment
- Safety evaluation

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/garak:25.07.1
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `max_new_tokens` | `150` |
| `temperature` | `0.1` |
| `top_p` | `0.7` |
| `parallelism` | `32` |
| `probes` | `None` |

### 11. Safety Harness Container
**NGC Catalog**: [safety-harness](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/safety-harness)

**Description**: Container for comprehensive safety evaluation of AI models.

**Use Cases**:
- Safety alignment evaluation
- Harmful content detection
- Bias and fairness assessment
- Ethical AI evaluation

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/safety-harness:25.07.3
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `6144` |
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `parallelism` | `8` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `judge` | `{'url': None, 'model_id': None, 'api_key': None, 'parallelism': 32, 'request_timeout': 60, 'max_retries': 16}` |

### 12. VLMEvalKit Container
**NGC Catalog**: [vlmevalkit](https://catalog.ngc.nvidia.com/orgs/nvidia/eval-factory/containers/vlmevalkit)

**Description**: Container for Vision-Language Model evaluation toolkit.

**Use Cases**:
- Multimodal model evaluation
- Image-text understanding assessment
- Visual reasoning evaluation
- Cross-modal performance testing

**Pull Command**:
```bash
docker pull nvcr.io/nvidia/eval-factory/vlmevalkit:25.07.1
```

**Defaults**:

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `2048` |
| `temperature` | `0` |
| `top_p` | `None` |
| `parallelism` | `4` |
| `max_retries` | `5` |
| `request_timeout` | `60` |

## Execution and API Usage

For detailed information on how to execute tasks and use the NeMo Evaluator API to run evaluations with these containers, see the [Using Containers](../workflows/using_containers.md).

## General Usage

### Prerequisites
- Docker or NVIDIA Container Toolkit
- NVIDIA GPU (for GPU-accelerated evaluation)
- Sufficient disk space for models and datasets

### Basic Container Usage
```bash
# Pull a container
docker pull nvcr.io/nvidia/eval-factory/<container-name>:latest

# Run a container with GPU support
docker run --gpus all -it nvcr.io/nvidia/eval-factory/<container-name>:latest

# Run with volume mounts for data persistence
docker run --gpus all -v /path/to/data:/workspace/data -it nvcr.io/nvidia/eval-factory/<container-name>:latest
```

## Getting Help

For specific container usage and configuration:
1. Check the individual NGC catalog pages for detailed documentation
2. Refer to the container's internal help documentation

## Support

For technical support and questions:
- Check the NGC catalog pages for container-specific information
- Contact NVIDIA support for enterprise customers
