(log-probability-evaluation)=

# Log-Probability Evaluations

Log-probability evaluations assess model confidence and uncertainty by analyzing the probabilities assigned to tokens without requiring text generation. This approach is particularly effective for multiple-choice scenarios and base model evaluation.

## Prerequisites

- Set up or select an existing evaluation target using either [PyTriton](../../deployment/pytriton.md) or [Ray Serve](../../deployment/ray-serve.md)
- Access to model tokenizer files (required for client-side tokenization)
- Hugging Face token for gated datasets and tokenizers
- Completions endpoint deployment (log-probability tasks cannot use chat endpoints)

---

## How Log-Probability Evaluation Works

In log-probability evaluation:

1. **Combined Input**: The model receives text containing both question and potential answer
2. **Token Probability Calculation**: Model assigns log-probabilities to each token in the sequence
3. **Answer-Specific Analysis**: Only tokens belonging to the answer portion are analyzed
4. **Confidence Assessment**: Higher probability sums indicate higher model confidence in that answer
5. **Multiple Choice Selection**: For multiple-choice tasks, the answer with highest probability sum is selected

This approach eliminates the need for complex instruction-following and provides direct insight into model uncertainty.

---

## Supported Tasks

```{list-table} Log-Probability Task Categories
:header-rows: 1

* - Category
  - Example Tasks
  - Description
* - Reasoning & Knowledge
  - `arc_challenge`, `arc_multilingual`
  - Science reasoning and multilingual variants
* - Common Sense
  - `hellaswag`, `hellaswag_multilingual`
  - Commonsense reasoning in multiple languages
* - Language Understanding
  - `winogrande`, `commonsense_qa`
  - Pronoun resolution and commonsense QA
* - Reading Comprehension
  - `lambada_openai`, `openbookqa`
  - Reading comprehension and open-book Q&A
* - Factual Knowledge
  - `truthfulqa`, `piqa`
  - Truthfulness assessment and physical reasoning
* - Advanced Reasoning
  - `bbh`, `musr`
  - Big-Bench Hard and multistep reasoning
* - Social Understanding
  - `social_iqa`
  - Social interaction Q&A
```

**Key Requirement**: All log-probability tasks require completions endpoints and tokenizer configuration.

---

## Configuration Requirements

### Target Configuration

Log-probability evaluations require specific endpoint configuration using NeMo Eval's Python API:

```python
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationTarget
)

# Configure the evaluation target
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type=EndpointType.COMPLETIONS,
        model_id="megatron_model"
    )
)
```

| Field | Description | Required | Notes |
|-------|-------------|----------|-------|
| `url` | Completions endpoint URL | Yes | Must end with `/v1/completions/` |
| `type` | Must be `EndpointType.COMPLETIONS` | Yes | Chat endpoints not supported |
| `model_id` | Model identifier | Yes | Usually `"megatron_model"` |

### Evaluation Configuration

```python
from nvidia_eval_commons.api.api_dataclasses import (
    ConfigParams, EvaluationConfig
)

# Configure the evaluation parameters
eval_config = EvaluationConfig(
    type="arc_challenge",
    output_dir="/results/arc_challenge",
    params=ConfigParams(
        limit_samples=100,
        parallelism=4,
        request_timeout=120,
        extra={
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
            "trust_remote_code": True
        }
    )
)
```

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

---

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

### Installation

```bash
# Core evaluation framework (pre-installed in NeMo container)
pip install nvidia-lm-eval>=25.6

# Verify installation
python -c "from nemo_eval.utils.base import list_available_evaluations; print('✅ NeMo Eval installed')"
```

---

## Complete Example: ARC Challenge

### Step 1: Deploy Model

Using the NeMo Eval deployment API:

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
from nvidia_eval_commons.core.evaluate import evaluate

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

---

## Data Format

### Input Format

Log-probability tasks use the underlying harness format. For ARC Challenge:

```json
{
  "question": "Which of the following is most likely to form a fossil?",
  "choices": ["a wooden toy", "a glass bottle", "a paper cup", "a snail shell"],
  "answerKey": "D"
}
```

### Processing Flow

1. **Question-Answer Combination**: Each choice is combined with the question
2. **Tokenization**: Client-side tokenization using provided tokenizer
3. **Log-Probability Request**: Server returns probabilities for each token
4. **Answer Isolation**: Only tokens belonging to the answer portion are analyzed
5. **Score Calculation**: Sum of log-probabilities for answer tokens

### Output Format

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

---

## Performance Optimization

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

---

## Troubleshooting

### Common Issues

#### ❌ Missing Tokenizer Configuration

**Error**: `TokenizerError: No tokenizer specified`

**Solution**:
```python
# Ensure tokenizer is specified in extra parameters
params = ConfigParams(
    extra={
        "tokenizer": "/path/to/nemo_tokenizer",
        "tokenizer_backend": "huggingface"
    }
)
```

#### ❌ Wrong Endpoint Type

**Error**: `EndpointError: Chat endpoint not supported`

**Solution**:
```python
# Use completions endpoint, not chat
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",  # Not /chat/completions/
    type=EndpointType.COMPLETIONS,
    model_id="megatron_model"
)
```

#### ❌ Tokenizer Path Issues

**Error**: `FileNotFoundError: Tokenizer not found`

**Solution**:
```bash
# Verify tokenizer path exists
ls -la /checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer/

# Check mounted paths in container
mount | grep checkpoints
```

#### ❌ Authentication Failures

**Error**: `HuggingFaceError: Authentication required`

**Solution**:
```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Verify token works
python -c "from huggingface_hub import whoami; print(whoami())"
```

### Performance Issues

#### Slow Evaluation

**Symptoms**: Low requests/second, high latency

**Solutions**:
1. Reduce `parallelism` if overwhelming server
2. Increase `request_timeout` for complex models
3. Use `limit_samples` for subset evaluation
4. Check network connectivity and GPU utilization

#### Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
1. Reduce `max_batch_size` in deployment
2. Lower `parallelism` to reduce concurrent requests
3. Use model parallelism for large models
4. Monitor GPU memory usage during evaluation

---

## Next Steps

- **Multiple Tasks**: Run [multiple log-probability tasks](lm-harness.md) in sequence
- **Custom Datasets**: Learn about [custom task configuration](../custom-tasks.md)
- **Performance Tuning**: See [configuration parameters](../parameters.md) for optimization
- **Alternative Methods**: Compare with [text generation evaluation](../text-generation/index.md)
