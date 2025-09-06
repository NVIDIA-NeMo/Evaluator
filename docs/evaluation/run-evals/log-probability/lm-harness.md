(eval-run-log-lm-harness)=

# LM Harness Log-Probability Tasks

LM Evaluation Harness provides the core log-probability evaluation tasks for academic benchmarks. These tasks assess model performance using probability analysis rather than text generation.

## Prerequisites

- Deployed model with completions endpoint
- Access to model tokenizer files  
- HuggingFace token for dataset access
- Review [log-probability evaluation overview](index.md) for concepts

---

## Available Tasks

### Reasoning & Knowledge Tasks

```{list-table} Reasoning Tasks Configuration
:header-rows: 1

* - Task
  - Description
  - Samples
  - Metrics
* - `arc_challenge`
  - AI2 Reasoning Challenge (hard subset)
  - 1,172
  - accuracy, accuracy_norm
* - `arc_multilingual`
  - ARC in multiple languages
  - ~1,000 per language
  - accuracy per language
* - `bbh`
  - Big-Bench Hard reasoning
  - 6,511
  - exact_match
* - `musr`
  - Multistep reasoning tasks
  - 980
  - accuracy
```

**Example Configuration**:
```python
eval_config = EvaluationConfig(
    type="arc_challenge",
    output_dir="/results/arc_challenge",
    params=ConfigParams(
        extra={
            "tokenizer": "/path/to/tokenizer",
            "tokenizer_backend": "huggingface"
        }
    )
)
```

### Common Sense Tasks

```{list-table} Common Sense Tasks Configuration
:header-rows: 1

* - Task
  - Description
  - Samples
  - Evaluation Focus
* - `hellaswag`
  - Commonsense reasoning about situations
  - 10,042
  - Situation modeling
* - `hellaswag_multilingual`
  - HellaSwag in multiple languages
  - ~10,000 per language
  - Cross-lingual reasoning
* - `winogrande`
  - Pronoun resolution reasoning
  - 1,267
  - Coreference resolution
* - `commonsense_qa`
  - Common sense question answering
  - 1,221
  - Everyday reasoning
```

**Multilingual Example**:
```python
eval_config = EvaluationConfig(
    type="hellaswag_multilingual",
    output_dir="/results/hellaswag_multilingual",
    params=ConfigParams(
        extra={
            "tokenizer": "/path/to/tokenizer",
            "tokenizer_backend": "huggingface",
            "languages": ["en", "es", "fr", "de"]  # Specify languages
        }
    )
)
```

### Reading Comprehension Tasks

```{list-table} Reading Comprehension Configuration
:header-rows: 1

* - Task
  - Description
  - Samples
  - Special Requirements
* - `lambada_openai`
  - Reading comprehension prediction
  - 5,153
  - Word prediction accuracy
* - `openbookqa`
  - Open-book science questions
  - 500
  - Science knowledge
* - `piqa`
  - Physical interaction Q&A
  - 1,838
  - Physical reasoning
```

**Reading Comprehension Example**:
```python
eval_config = EvaluationConfig(
    type="lambada_openai",
    output_dir="/results/lambada_openai", 
    params=ConfigParams(
        extra={
            "tokenizer": "/path/to/tokenizer",
            "tokenizer_backend": "huggingface",
            "add_bos_token": True  # Often required for reading tasks
        }
    )
)
```

### Factual Knowledge Tasks

```{list-table} Knowledge Tasks Configuration
:header-rows: 1

* - Task
  - Description
  - Samples
  - Assessment Type
* - `truthfulqa`
  - Truthfulness in question answering
  - 817
  - Factual accuracy vs. plausibility
* - `social_iqa`
  - Social interaction reasoning
  - 1,954
  - Social understanding
```

---

## Task-Specific Configuration

### ARC Challenge (Recommended Starting Point)

```python
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EndpointType, EvaluationConfig, EvaluationTarget, ConfigParams
)
from nemo_evaluator.core.evaluate import evaluate

# Basic configuration
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type=EndpointType.COMPLETIONS,
        model_id="megatron_model"
    )
)

eval_config = EvaluationConfig(
    type="arc_challenge",
    output_dir="/results/arc_challenge",
    params=ConfigParams(
        limit_samples=50,  # Start with subset
        parallelism=4,
        extra={
            "tokenizer": "/checkpoints/model/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
            "trust_remote_code": True
        }
    )
)

results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
```

### HellaSwag (Common Sense Reasoning)

```python
eval_config = EvaluationConfig(
    type="hellaswag",
    output_dir="/results/hellaswag", 
    params=ConfigParams(
        # HellaSwag benefits from larger context
        extra={
            "tokenizer": "/checkpoints/model/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
            "max_length": 2048,  # Longer context for situation modeling
            "add_bos_token": True
        }
    )
)
```

### TruthfulQA (Factual Accuracy)

```python
eval_config = EvaluationConfig(
    type="truthfulqa",
    output_dir="/results/truthfulqa",
    params=ConfigParams(
        # TruthfulQA is challenging - use careful settings
        parallelism=2,  # Conservative parallelism
        request_timeout=180,  # Longer timeout
        extra={
            "tokenizer": "/checkpoints/model/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
            # TruthfulQA-specific parameters
            "preset": "qa",  # Question-answer format
            "num_few_shot": 0  # Zero-shot evaluation recommended
        }
    )
)
```

---

## Multi-Task Evaluation

Run multiple log-probability tasks in sequence:

```python
import os
from nemo_evaluator.core.evaluate import evaluate

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
    print(f"🚀 Running {task}...")
    
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/{task}",
        params=base_params
    )
    
    task_results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
    results[task] = task_results.tasks[task]
    
    print(f"✅ {task} completed: {task_results.tasks[task]}")

# Summary report
print("\n📊 Log-Probability Evaluation Summary:")
for task, metrics in results.items():
    acc = metrics.get('acc', metrics.get('exact_match', 'N/A'))
    print(f"{task:15}: {acc:.3f}")
```

---

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

---

## Expected Results

### Baseline Performance Ranges

```{list-table} Typical Performance Ranges by Model Size
:header-rows: 1

* - Task
  - 1B Model
  - 7B Model  
  - 70B Model
* - ARC Challenge
  - 0.35-0.45
  - 0.55-0.65
  - 0.70-0.80
* - HellaSwag
  - 0.45-0.55
  - 0.70-0.80
  - 0.85-0.90
* - Winogrande
  - 0.55-0.65
  - 0.70-0.75
  - 0.80-0.85
* - TruthfulQA
  - 0.30-0.40
  - 0.35-0.45
  - 0.45-0.55
```

*Note: Actual performance depends on training data, architecture, and fine-tuning.*

### Interpreting Results

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
