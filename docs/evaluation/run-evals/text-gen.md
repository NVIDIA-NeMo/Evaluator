(eval-run-text)=

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

## Discovering Available Tasks

Use the `list_available_evaluations` function to discover evaluation tasks in your environment:

```python
from nemo_eval.utils.base import list_available_evaluations

# Get dictionary of available tasks by framework
available_tasks = list_available_evaluations()
print(available_tasks)
```

Example output:
```python
{
    'core_evals.lm_evaluation_harness': [
        'mmlu', 'mmlu_instruct', 'gsm8k', 'arc_challenge',
        'hellaswag', 'truthfulqa', 'ifeval', 'mmlu_pro'
        # ... more tasks
    ]
}
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
from nvidia_eval_commons.core.evaluate import evaluate
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

## Next Steps

- **End-to-End Tutorial**: Follow [MMLU Tutorial](../tutorials/mmlu.ipynb) for complete workflow
- **Parameter Configuration**: See [Evaluation Configuration Parameters](parameters.md) for comprehensive settings reference
- **Custom Task Setup**: Learn [Custom Task Configuration](custom-tasks.md) for specialized evaluations
- **Log-Probability Methods**: Explore [Log-Probability Evaluation](logprobs.md) for confidence-based assessment
- **Comprehensive Benchmarks**: Browse [Benchmark Catalog](benchmarks.md) for all available tasks
