(text-gen)=

# Text Generation Evaluation

<!-- active person first intro sentence -->

Text generation evaluation is the primary method for assessing LLM capabilities where models produce natural language responses to prompts. This approach evaluates the quality, accuracy, and appropriateness of generated text across various tasks and domains.

## Before You Start

Ensure you have:

1. TODO

---

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

:::{tab-item}  Core API
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

:::{tab-item}  Containers Directly
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

```{list-table} Text Generation Benchmarks Overview
:header-rows: 1
:widths: 25 25 25 25

* - Area
  - Purpose
  - Key Tasks
  - Evaluation Method
* - Academic Benchmarks
  - Assess general knowledge and reasoning across academic domains
  - - `mmlu`
    - `arc_challenge`
    - `hellaswag`
    - `truthfulqa`
  - Multiple-choice or short-answer text generation
* - Instruction Following
  - Evaluate ability to follow complex instructions and formatting requirements
  - - `ifeval`
    - `mmlu_instruct`
    - `gpqa_diamond_cot`
  - Generated responses assessed against instruction criteria
* - Mathematical Reasoning
  - Test mathematical problem-solving and multi-step reasoning
  - - `gsm8k`
    - `mgsm`
    - `math`
  - Final answer extraction and numerical comparison
* - Multilingual Evaluation
  - Assess capabilities across different languages
  - - `arc_multilingual`
    - `hellaswag_multilingual`
    - `mgsm`
  - Language-specific text generation and assessment
```

## Task Naming and Framework Specification

::::{tab-set}
:::{tab-item} Standard Names
:sync: standard

Use simple task names when only one framework provides the task:

```python
# Unambiguous task names
config = EvaluationConfig(type="mmlu")
config = EvaluationConfig(type="gsm8k")
config = EvaluationConfig(type="arc_challenge")
```

These tasks have unique names across all evaluation frameworks, so no qualification is needed.

:::

:::{tab-item} Framework-Qualified Names
:sync: qualified

When multiple frameworks provide the same task, specify the framework explicitly:

```python
# Explicit framework specification
config = EvaluationConfig(type="lm-evaluation-harness.mmlu")
config = EvaluationConfig(type="simple-evals.mmlu")
```

Use this approach when:
- Multiple frameworks implement the same benchmark
- You need specific framework behavior or scoring
- Avoiding ambiguity in task resolution

:::

:::{tab-item} Framework Discovery
:sync: discovery

Resolve task naming conflicts using the framework discovery utility:

```python
from nemo_eval.utils.base import find_framework

# Find which framework provides a task
framework = find_framework("mmlu")
print(f"MMLU provided by: {framework}")
```

This utility helps you:
- Identify which framework implements a task
- Resolve naming conflicts programmatically
- Understand available task sources

:::
::::

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

For comprehensive parameter reference including all available settings, optimization patterns, and framework-specific options, refer to [Evaluation Configuration Parameters](../parameters.md).

**Key Parameters for Text Generation**:
- `temperature=0` for deterministic, reproducible results
- `max_new_tokens` to control response length
- `limit_samples` for subset evaluation during testing
- `parallelism` to balance speed with server capacity