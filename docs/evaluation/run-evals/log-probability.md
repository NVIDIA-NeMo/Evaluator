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
nv-eval ls tasks | grep -E "(arc_challenge|hellaswag|winogrande|truthfulqa)"

# Run ARC Challenge evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o evaluation.tasks='["arc_challenge"]' \
    -o target.api_endpoint.url=http://0.0.0.0:8080/v1/completions \
    -o target.api_endpoint.type=completions \
    -o target.api_endpoint.model_id=megatron_model

# Run multiple log-probability benchmarks
nv-eval run \
    --config-dir examples \
    --config-name local_log_probability_suite \
    -o evaluation.tasks='["arc_challenge", "hellaswag", "winogrande", "truthfulqa"]'
```

:::

:::{tab-item} Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, ConfigParams
)

# Configure log-probability evaluation
eval_config = EvaluationConfig(
    type="arc_challenge",
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
print(f"ARC Challenge Accuracy: {result.tasks['arc_challenge']['acc']:.1%}")
```

:::

:::{tab-item} Containers Directly
:sync: containers

For specialized container workflows:

```bash
# Pull and run LM Evaluation Harness container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/lm-evaluation-harness:25.07.3 bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here
export HF_TOKEN=your_hf_token_here

# Run log-probability evaluation
eval-factory run_eval \
    --eval_type arc_challenge \
    --model_id megatron_model \
    --model_url http://0.0.0.0:8080/v1/completions \
    --model_type completions \
    --output_dir /tmp/results \
    --overrides 'config.params.extra.tokenizer=/path/to/tokenizer,config.params.extra.tokenizer_backend=huggingface,config.params.limit_samples=100'
```

:::
::::

## Installation

Install the LM Evaluation Harness for local development:

```bash
# Core evaluation framework (pre-installed in NeMo container)
pip install nvidia-lm-eval==25.7.1

# Verify installation
python -c "from nemo_eval.utils.base import list_available_evaluations; print('NeMo Eval installed')"
```

## Discovering Available Tasks

Use the launcher CLI to discover all available log-probability tasks:

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter for log-probability tasks
nv-eval ls tasks | grep -E "(arc_challenge|hellaswag|winogrande|truthfulqa|bbh|musr)"

# Get detailed information about a specific task (if supported)
nv-eval ls tasks --task arc_challenge
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
| `arc_challenge` | AI2 Reasoning Challenge (hard subset) | 1,172 | accuracy, accuracy_norm |
| `arc_multilingual` | ARC in multiple languages | ~1,000 per language | accuracy per language |
| `bbh` | Big-Bench Hard reasoning | 6,511 | exact_match |
| `musr` | Multistep reasoning tasks | 980 | accuracy |

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
| `lambada_openai` | Reading comprehension prediction | 5,153 | Word prediction accuracy |
| `openbookqa` | Open-book science questions | 500 | Science knowledge |
| `piqa` | Physical interaction Q&A | 1,838 | Physical reasoning |

### Factual Knowledge Tasks

| Task | Description | Samples | Assessment Type |
|------|-------------|---------|-----------------|
| `truthfulqa` | Truthfulness in question answering | 817 | Factual accuracy vs. plausibility |
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

**Deploy your model**: Choose from [Launcher-Orchestrated Deployment](../../deployment/launcher-orchestrated/index.md) (recommended) or [Bring-Your-Own-Endpoint](../../deployment/bring-your-own-endpoint/index.md) with [PyTriton](../../deployment/bring-your-own-endpoint/pytriton.md) or [Ray Serve](../../deployment/bring-your-own-endpoint/ray-serve.md).

For a quick deployment example:

```python
from nemo_eval.api import deploy

# Deploy NeMo checkpoint with PyTriton backend
deploy(
    nemo_checkpoint="/checkpoints/llama-3_2-1b-instruct_v2.0",
    serving_backend="pytriton",
    server_port=8080,
    num_gpus=1,
    max_input_len=4096,
    max_batch_size=8
)
```

### Step 2: Wait for Server Readiness

```python
from nemo_eval.utils.base import wait_for_fastapi_server

# Verify server and model are ready
server_ready = wait_for_fastapi_server(
    base_url="http://0.0.0.0:8080",
    model_name="megatron_model",
    max_retries=600
)
assert server_ready, "Server not ready"
```

### Step 3: Configure and Run Evaluation

```python
from nvidia_eval_commons.api.api_dataclasses import (
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
    type="arc_challenge",
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
print(f"Accuracy: {results.tasks['arc_challenge']['acc']:.3f}")
```

## Multi-Task Evaluation

Run multiple log-probability tasks in sequence:

```python
# Define tasks to evaluate
log_prob_tasks = [
    "arc_challenge",
    "hellaswag", 
    "winogrande",
    "truthfulqa"
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
    
    print(f"{task} completed: {task_results.tasks[task]}")

# Summary report
print("\nLog-Probability Evaluation Summary:")
for task, metrics in results.items():
    acc = metrics.get('acc', metrics.get('exact_match', 'N/A'))
    print(f"{task:15}: {acc:.3f}")
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
        "tokenizer_backend": "huggingface",
        "batch_size": 8,  # Request batching if supported
        "cache_requests": True  # Enable caching
    }
)
```

## Understanding Results

### Result Structure

The `evaluate()` function returns a results object with task metrics:

```python
# Example results structure
results.tasks = {
    'arc_challenge': {
        'acc': 0.847,
        'acc_stderr': 0.016,
        'acc_norm': 0.851,
        'acc_norm_stderr': 0.016
    }
}

# Access specific metrics
accuracy = results.tasks['arc_challenge']['acc']
print(f"ARC Challenge Accuracy: {accuracy:.1%}")
```

### Metric Interpretation

**Accuracy (`acc`)**: Standard accuracy metric
```python
print(f"Model answered {results.tasks['arc_challenge']['acc']:.1%} correctly")
```

**Normalized Accuracy (`acc_norm`)**: Length-normalized scoring
```python
# Often more reliable for log-probability evaluation
norm_acc = results.tasks['arc_challenge']['acc_norm']
```

**Standard Error**: Confidence intervals
```python
acc = results.tasks['arc_challenge']['acc']
stderr = results.tasks['arc_challenge']['acc_stderr'] 
print(f"Accuracy: {acc:.3f} ± {stderr:.3f}")
```

### Baseline Performance Ranges

| Task | 1B Model | 7B Model | 70B Model |
|------|----------|----------|-----------|
| ARC Challenge | 0.35-0.45 | 0.55-0.65 | 0.70-0.80 |
| HellaSwag | 0.45-0.55 | 0.70-0.80 | 0.85-0.90 |
| Winogrande | 0.55-0.65 | 0.70-0.75 | 0.80-0.85 |
| TruthfulQA | 0.30-0.40 | 0.35-0.45 | 0.45-0.55 |

*Note: Actual performance depends on training data, architecture, and fine-tuning.*

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
        "tokenizer_backend": "huggingface",
        "batch_size": 4   # If supported by harness
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

**Problem**: Missing tokenizer for log-probability tasks
```python
# Incorrect - missing tokenizer
params = ConfigParams(extra={})
```

**Solution**: Always specify tokenizer for log-probability tasks
```python
# Correct
params = ConfigParams(
    extra={
        "tokenizer_backend": "huggingface",
        "tokenizer": "/path/to/nemo_tokenizer"
    }
)
```

### Wrong Endpoint Type

**Problem**: Using chat endpoint for log-probability tasks
```python
# Incorrect - log-probability requires completions
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/chat/completions",
    type=EndpointType.CHAT
)
```

**Solution**: Use completions endpoint
```python
# Correct
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type=EndpointType.COMPLETIONS
)
```

---

*For comprehensive parameter documentation, see [Evaluation Configuration Parameters](../parameters.md). For custom task configuration, see [Custom Task Configuration](../custom-tasks.md).*
