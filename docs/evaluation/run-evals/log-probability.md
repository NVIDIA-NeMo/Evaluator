(log-probability)=

# Log-Probability Evaluation

Assess model confidence and uncertainty by analyzing the probabilities assigned to tokens without requiring text generation. This approach is particularly effective for multiple-choice scenarios and base model evaluation.

## Overview

Log-probability evaluation quantifies a model's "surprise" or uncertainty when processing text sequences by calculating the sum of log-probabilities assigned to each token. This method provides direct insight into model confidence and eliminates the need for complex instruction-following.

**Key Benefits**:

- **Direct confidence measurement** through probability analysis
- **No text generation required** - faster evaluation
- **Ideal for base models** - no instruction-following needed
- **Reproducible results** - deterministic probability calculations

## Before You Start

Ensure you have:

- **Completions Endpoint**: Log-probability tasks require completions endpoints (not chat)
- **Model Tokenizer**: Access to tokenizer files for client-side tokenization
- **API Access**: Valid API key for your model endpoint
- **Authentication**: Hugging Face token for gated datasets and tokenizers

---

## Choose Your Approach

::::{tab-set}
:::{tab-item} NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run log-probability evaluations with unified CLI:

```bash
# List available log-probability tasks
nv-eval ls tasks | grep -E "(arc|hellaswag|winogrande|truthfulqa)"

# Run ARC Challenge evaluation with existing endpoint
# Note: Configure tokenizer parameters in your YAML config file
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=http://0.0.0.0:8080/v1/completions \
    -o target.api_endpoint.type=completions \
    -o target.api_endpoint.model_id=megatron_model
```

:::

:::{tab-item} Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, ConfigParams
)

# Configure log-probability evaluation
eval_config = EvaluationConfig(
    type="adlr_arc_challenge_llama",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=100,  # Remove for full dataset
        parallelism=4,      # Concurrent requests
        request_timeout=120,
        extra={
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
            "trust_remote_code": True
        }
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type=EndpointType.COMPLETIONS,
        model_id="megatron_model"
    )
)

result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
# Access accuracy from nested result structure
task_name = "adlr_arc_challenge_llama"
accuracy = result.tasks[task_name].metrics['acc'].scores['acc'].value
print(f"ARC Challenge Accuracy: {accuracy:.1%}")
```

:::

:::{tab-item} Containers Directly
:sync: containers

For specialized container workflows:

```bash
# Pull and run LM Evaluation Harness container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/lm-evaluation-harness:{{ docker_compose_latest }} bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here
export HF_TOKEN=your_hf_token_here

# Run log-probability evaluation using eval-factory (nemo-evaluator CLI)
eval-factory run_eval \
    --eval_type adlr_arc_challenge_llama \
    --model_id megatron_model \
    --model_url http://0.0.0.0:8080/v1/completions \
    --model_type completions \
    --output_dir /tmp/results \
    --overrides "config.params.extra.tokenizer=/path/to/tokenizer,config.params.extra.tokenizer_backend=huggingface,config.params.limit_samples=100"
```

:::
::::

## Installation

Install the LM Evaluation Harness for local development:

```bash
# Core evaluation framework (pre-installed in NeMo container)
pip install nvidia-lm-eval

# Verify installation
python -c "from nemo_evaluator import show_available_tasks; print('NeMo Evaluator installed')"
```

## Discovering Available Tasks

Use the launcher CLI to discover all available log-probability tasks:

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter for log-probability tasks
nv-eval ls tasks | grep -E "(arc|hellaswag|winogrande|truthfulqa)"

# Get detailed information about a specific task (if supported)
nv-eval ls tasks --task adlr_arc_challenge_llama
```

## How Log-Probability Evaluation Works

In log-probability evaluation:

1. **Combined Input**: The model receives text containing both question and potential answer
2. **Token Probability Calculation**: Model assigns log-probabilities to each token in the sequence
3. **Answer-Specific Analysis**: Only tokens belonging to the answer portion are analyzed
4. **Confidence Assessment**: Higher probability sums indicate higher model confidence in that answer
5. **Multiple Choice Selection**: For multiple-choice tasks, the answer with highest probability sum is selected

This approach eliminates the need for complex instruction-following and provides direct insight into model uncertainty.

## Available Tasks

### Reasoning and Knowledge Tasks

| Task | Description | Samples | Metrics |
|------|-------------|---------|---------|
| `adlr_arc_challenge_llama` | AI2 Reasoning Challenge (hard subset) | 1,172 | accuracy, accuracy_norm |
| `arc_multilingual` | ARC in multiple languages | ~1,000 per language | accuracy per language |

### Common Sense Tasks

| Task | Description | Samples | Evaluation Focus |
|------|-------------|---------|------------------|
| `hellaswag` | Commonsense reasoning about situations | 10,042 | Situation modeling |
| `hellaswag_multilingual` | HellaSwag in multiple languages | ~10,000 per language | Cross-lingual reasoning |
| `winogrande` | Pronoun resolution reasoning | 1,267 | Coreference resolution |
| `commonsense_qa` | Common sense question answering | 1,221 | Everyday reasoning |

### Reading Comprehension Tasks

| Task | Description | Samples | Special Requirements |
|------|-------------|---------|---------------------|
| `openbookqa` | Open-book science questions | 500 | Science knowledge |
| `piqa` | Physical interaction Q&A | 1,838 | Physical reasoning |

:::{note}
For tasks not listed in the pre-configured set, you can access additional LM Evaluation Harness tasks using the framework-qualified format: `lm-evaluation-harness.<task_name>` (e.g., `lm-evaluation-harness.lambada_openai`). Refer to {ref}`eval-custom-tasks` for more details.
:::

### Factual Knowledge Tasks

| Task | Description | Samples | Assessment Type |
|------|-------------|---------|-----------------|
| `adlr_truthfulqa_mc2` | Truthfulness in question answering | 817 | Factual accuracy vs. plausibility |
| `social_iqa` | Social interaction reasoning | 1,954 | Social understanding |

**Key Requirement**: All log-probability tasks require completions endpoints and tokenizer configuration.

## Configuration Requirements

### Required Parameters (in `extra` dict)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `tokenizer` | `str` | Path to model tokenizer | `"/path/to/nemo_tokenizer"` |
| `tokenizer_backend` | `str` | Tokenizer implementation | `"huggingface"`, `"sentencepiece"` |

### Optional Parameters (in `extra` dict)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `trust_remote_code` | `bool` | Allow remote code in tokenizer | `False` |
| `add_bos_token` | `bool` | Add beginning-of-sequence token | Model default |
| `add_eos_token` | `bool` | Add end-of-sequence token | Model default |

## Complete Example: ARC Challenge

### Step 1: Deploy Model

**Deploy your model**: Choose from {ref}`launcher-orchestrated-deployment` (recommended) or {ref}`bring-your-own-endpoint`.

For manual deployment, refer to the {ref}`bring-your-own-endpoint-manual` guide for instructions on deploying with vLLM, TensorRT-LLM, or other serving frameworks.

### Step 2: Wait for Server Readiness

```python
import requests
import time

# Example health check - adjust endpoint path based on your deployment
# vLLM/SGLang/NIM: use /health
# Custom deployments: check your framework's health endpoint
base_url = "http://0.0.0.0:8080"
max_retries = 60
for _ in range(max_retries):
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("Server ready")
            break
    except requests.exceptions.RequestException:
        pass
    time.sleep(10)
else:
    raise RuntimeError("Server not ready after waiting")
```

### Step 3: Configure and Run Evaluation

```python
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint, ConfigParams, EndpointType, EvaluationConfig, EvaluationTarget
)
from nemo_evaluator.core.evaluate import evaluate

# Configure target
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type=EndpointType.COMPLETIONS,
        model_id="megatron_model"
    )
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="adlr_arc_challenge_llama",
    output_dir="/results/arc_challenge",
    params=ConfigParams(
        limit_samples=100,  # Subset for testing
        parallelism=4,      # Concurrent requests
        extra={
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
            "trust_remote_code": True
        }
    )
)

# Run evaluation
results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
print(f"Accuracy: {results.tasks['adlr_arc_challenge_llama'].metrics['acc'].scores['acc'].value:.3f}")
```

## Multi-Task Evaluation

Run multiple log-probability tasks in sequence:

```python
# Define tasks to evaluate
log_prob_tasks = [
    "adlr_arc_challenge_llama",
    "hellaswag", 
    "winogrande",
    "adlr_truthfulqa_mc2"
]

# Base configuration
base_params = ConfigParams(
    limit_samples=100,  # Subset for quick evaluation
    parallelism=4,
    extra={
        "tokenizer": "/checkpoints/model/context/nemo_tokenizer", 
        "tokenizer_backend": "huggingface",
        "trust_remote_code": True
    }
)

# Run evaluations
results = {}
for task in log_prob_tasks:
    print(f"Running {task}...")
    
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/{task}",
        params=base_params
    )
    
    task_results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
    results[task] = task_results.tasks[task]
    
    print(f"{task} completed")

# Summary report
print("\nLog-Probability Evaluation Summary:")
for task_name, task_result in results.items():
    # Access accuracy metric from task results
    if 'acc' in task_result.metrics:
        acc = task_result.metrics['acc'].scores['acc'].value
        print(f"{task_name:15}: {acc:.3f}")
    elif 'exact_match' in task_result.metrics:
        em = task_result.metrics['exact_match'].scores['exact_match'].value
        print(f"{task_name:15}: {em:.3f}")
```

## Advanced Configuration

### Custom Few-Shot Settings

Some tasks benefit from few-shot examples:

```python
params = ConfigParams(
    extra={
        "tokenizer": "/path/to/tokenizer",
        "tokenizer_backend": "huggingface",
        "num_fewshot": 5,  # Number of examples
        "fewshot_delimiter": "\n\n",  # Separator
        "fewshot_seed": 42  # Reproducible selection
    }
)
```

### Language-Specific Configuration

For multilingual tasks:

```python
params = ConfigParams(
    extra={
        "tokenizer": "/path/to/tokenizer",
        "tokenizer_backend": "huggingface",
        "languages": ["en", "es", "fr", "de", "zh"],
        "max_length": 2048,  # Longer context for some languages
        "trust_remote_code": True
    }
)
```

### Performance Optimization

```python
# High-throughput configuration
params = ConfigParams(
    parallelism=16,  # High concurrency
    request_timeout=60,  # Shorter timeout
    max_retries=3,   # Retry policy
    extra={
        "tokenizer": "/path/to/tokenizer",
        "tokenizer_backend": "huggingface"
    }
)
```

## Understanding Results

### Result Structure

The `evaluate()` function returns a results object with task metrics:

```python
# Example results structure - EvaluationResult with nested dataclasses
# results.tasks = Dict[str, TaskResult]
# TaskResult.metrics = Dict[str, MetricResult]
# MetricResult.scores = Dict[str, Score]
# Score.value = float

# Access specific metrics
task_result = results.tasks['adlr_arc_challenge_llama']
acc_metric = task_result.metrics['acc']
accuracy = acc_metric.scores['acc'].value
print(f"ARC Challenge Accuracy: {accuracy:.1%}")
```

### Metric Interpretation

**Accuracy (`acc`)**: Standard accuracy metric
```python
acc = results.tasks['adlr_arc_challenge_llama'].metrics['acc'].scores['acc'].value
print(f"Model answered {acc:.1%} correctly")
```

**Normalized Accuracy (`acc_norm`)**: Length-normalized scoring
```python
# Often more reliable for log-probability evaluation
norm_acc = results.tasks['adlr_arc_challenge_llama'].metrics['acc_norm'].scores['acc_norm'].value
```

**Standard Error**: Confidence intervals
```python
acc_metric = results.tasks['adlr_arc_challenge_llama'].metrics['acc']
acc = acc_metric.scores['acc'].value
stderr = acc_metric.scores['acc'].stats.stderr
print(f"Accuracy: {acc:.3f} ± {stderr:.3f}")
```

## Performance Considerations

### Recommended Settings

**Development/Testing**:
```python
ConfigParams(
    limit_samples=10,     # Quick validation
    parallelism=1,        # Conservative
    request_timeout=60,   # Standard timeout
    extra={
        "tokenizer": "/path/to/tokenizer",
        "tokenizer_backend": "huggingface"
    }
)
```

**Production Evaluation**:
```python
ConfigParams(
    limit_samples=None,   # Full dataset
    parallelism=8,        # High throughput
    request_timeout=120,  # Generous timeout
    extra={
        "tokenizer": "/path/to/tokenizer",
        "tokenizer_backend": "huggingface"
    }
)
```

### Throughput Considerations

- **Tokenizer Performance**: Client-side tokenization can be bottleneck
- **Request Batching**: Adjust parallelism based on server capacity
- **Memory Usage**: Log-probability calculations require additional GPU memory
- **Network Latency**: Higher parallelism reduces total evaluation time

## Environment Setup

### Authentication

```bash
# Required for gated datasets and tokenizers
export HF_TOKEN="your_huggingface_token"

# Required for some benchmarks
export HF_DATASETS_TRUST_REMOTE_CODE="1"

# Optional: Cache management
export HF_HOME="/path/to/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

## Common Configuration Errors

### Missing Tokenizer

:::{admonition} Problem
:class: error
Missing tokenizer for log-probability tasks

```python
# Incorrect - missing tokenizer
params = ConfigParams(extra={})
```
:::

:::{admonition} Solution
:class: tip
Always specify tokenizer for log-probability tasks

```python
# Correct
params = ConfigParams(
    extra={
        "tokenizer_backend": "huggingface",
        "tokenizer": "/path/to/nemo_tokenizer"
    }
)
```
:::

### Wrong Endpoint Type

:::{admonition} Problem
:class: error
Using chat endpoint for log-probability tasks

```python
# Incorrect - log-probability requires completions
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/chat/completions",
    type=EndpointType.CHAT
)
```
:::

:::{admonition} Solution
:class: tip
Use completions endpoint

```python
# Correct
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type=EndpointType.COMPLETIONS
)
```
:::

---

*For comprehensive parameter documentation, see {ref}`eval-parameters`. For custom task configuration, see {ref}`eval-custom-tasks`.*
