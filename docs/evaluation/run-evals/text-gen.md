(text-gen)=

# Text Generation Evaluation

Understanding the core concepts and configuration for text generation evaluation in NeMo Eval.


## Overview

Text generation evaluation is the primary method for assessing LLM capabilities where models produce natural language responses to prompts. This approach evaluates the quality, accuracy, and appropriateness of generated text across various tasks and domains.

## Evaluation Approach

In text generation evaluation:

1. **Prompt Construction**: Models receive carefully crafted prompts (questions, instructions, or text to continue)
2. **Response Generation**: Models generate natural language responses using their trained parameters
3. **Response Assessment**: Generated text is evaluated for correctness, quality, or adherence to specific criteria
4. **Metric Calculation**: Numerical scores are computed based on evaluation criteria

This differs from **log-probability evaluation** where models assign confidence scores to predefined choices.
For log-probability methods, refer to ["Log-Probability Evaluation"](logprobs.md).

## Choose Your Approach

::::{tab-set}
:::{tab-item} NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run text generation evaluations with unified CLI:

```bash
# List available text generation tasks
nv-eval ls tasks

# Run MMLU Pro evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o evaluation.tasks='["mmlu_pro"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}

# Run multiple text generation benchmarks
nv-eval run \
    --config-dir examples \
    --config-name local_text_generation_suite \
    -o evaluation.tasks='["mmlu_pro", "arc_challenge", "hellaswag", "truthfulqa"]'
```

:::

:::{tab-item} ⚙️ Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams
)

# Configure text generation evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=None,  # Full dataset
        temperature=0.0,     # Deterministic
        max_new_tokens=512,
        top_p=0.95
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct", 
        type="chat",
        api_key="your_api_key"
    )
)

result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
```

:::

:::{tab-item} 🐳 Containers Directly
:sync: containers

For specialized container workflows:

```bash
# Pull and run text generation container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here

# Run evaluation
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=100'
```

:::
::::

## Discovering Available Tasks

Use the launcher CLI to discover all available text generation tasks:

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter by text generation category (if supported)
nv-eval ls tasks --filter text_generation

# Get detailed information about a specific task (if supported)
nv-eval ls tasks --task mmlu_pro
```

## Text Generation Task Categories

### Academic Benchmarks

**Purpose**: Assess general knowledge and reasoning across academic domains

**Key Tasks**:
- `mmlu` - Massive Multitask Language Understanding (57 academic subjects)
- `arc_challenge` - AI2 Reasoning Challenge (science questions)
- `hellaswag` - Commonsense reasoning and situation modeling
- `truthfulqa` - Factual accuracy and truthfulness

**Evaluation Method**: Multiple-choice or short-answer text generation

### Instruction Following

**Purpose**: Evaluate ability to follow complex instructions and formatting requirements

**Key Tasks**:
- `ifeval` - Instruction Following Evaluation
- `mmlu_instruct` - MMLU with instruction formatting
- `gpqa_diamond_cot` - Graduate-level Q&A with chain-of-thought

**Evaluation Method**: Generated responses assessed against specific instruction criteria

### Mathematical Reasoning

**Purpose**: Test mathematical problem-solving and multi-step reasoning

**Key Tasks**:
- `gsm8k` - Grade school math word problems
- `mgsm` - Multilingual grade school math
- `math` - Competition-level mathematics

**Evaluation Method**: Final answer extraction and numerical comparison

### Multilingual Evaluation

**Purpose**: Assess capabilities across different languages

**Key Tasks**:
- `arc_multilingual` - ARC in multiple languages
- `hellaswag_multilingual` - Cross-lingual commonsense reasoning
- `mgsm` - Math problems in various languages

**Evaluation Method**: Language-specific text generation and assessment

## Task Naming and Framework Specification

### Standard Task Names

Use simple task names when only one framework provides the task:

```python
# Unambiguous task names
config = EvaluationConfig(type="mmlu")
config = EvaluationConfig(type="gsm8k")
config = EvaluationConfig(type="arc_challenge")
```

### Framework-Qualified Names

When multiple frameworks provide the same task, specify the framework explicitly:

```python
# Explicit framework specification
config = EvaluationConfig(type="lm-evaluation-harness.mmlu")
config = EvaluationConfig(type="simple-evals.mmlu")
```

### Finding Framework for Tasks

Resolve task naming conflicts using the framework discovery utility:

```python
from nemo_eval.utils.base import find_framework

# Find which framework provides a task
framework = find_framework("mmlu")
print(f"MMLU provided by: {framework}")
```

## Evaluation Configuration

### Basic Configuration Structure

Text generation evaluations use the NVIDIA Eval Commons framework:

```python
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, ConfigParams
)

# Configure target endpoint
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation parameters
params = ConfigParams(
    temperature=0,      # Deterministic generation
    top_p=1.0,         # No nucleus sampling
    limit_samples=100, # Evaluate subset for testing
    parallelism=1      # Single-threaded requests
)

# Configure evaluation task
config = EvaluationConfig(
    type="mmlu",
    params=params,
    output_dir="./evaluation_results"
)

# Execute evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
```

### Endpoint Types

**Completions Endpoint** (`/v1/completions/`):
- Direct text completion without conversation formatting
- Used for: Academic benchmarks, reasoning tasks, base model evaluation
- Model processes prompts as-is without applying chat templates

**Chat Endpoint** (`/v1/chat/completions/`):
- Conversational interface with role-based message formatting
- Used for: Instruction following, chat benchmarks, instruction-tuned models
- Requires models with defined chat templates

### Configuration Parameters

For comprehensive parameter reference including all available settings, optimization patterns, and framework-specific options, refer to [Evaluation Configuration Parameters](parameters.md).

**Key Parameters for Text Generation**:
- `temperature=0` for deterministic, reproducible results
- `max_new_tokens` to control response length
- `limit_samples` for subset evaluation during testing
- `parallelism` to balance speed with server capacity

## Prerequisites

Before configuring text generation evaluations:

1. **Deployed Model**: Model accessible via OpenAI-compatible API endpoints
2. **Evaluation Framework**: `nvidia-lm-eval` or additional frameworks installed  
3. **Authentication**: HuggingFace token for gated datasets (if required)
4. **Resources**: Adequate compute for generation and evaluation processing