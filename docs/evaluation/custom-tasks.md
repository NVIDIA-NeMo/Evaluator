(eval-custom-tasks)=

# Custom Task Evaluation

Advanced guide for evaluating models on tasks without pre-defined configurations using custom benchmark definitions and configuration patterns.


## Overview

While NeMo Evaluator provides pre-configured tasks for common benchmarks, you may need to evaluate models on:

- **Research Benchmarks**: Newly released datasets not yet integrated
- **Custom Datasets**: Proprietary or domain-specific evaluation data  
- **Task Variants**: Modified versions of existing benchmarks with different settings
- **Specialized Configurations**: Tasks requiring specific parameters or tokenizers

This guide demonstrates how to configure custom evaluations across multiple harnesses and optimization patterns.

## When to Use Custom Tasks

**Choose Custom Tasks When**:

- Your target benchmark lacks a pre-defined configuration
- You need specific few-shot settings different from defaults
- Research requires non-standard evaluation parameters
- Evaluating on proprietary or modified datasets

**Use Pre-Defined Tasks When**:

- Standard benchmarks with optimal settings (refer to [Text Generation Evaluation](run-evals/text-gen.md))
- Quick prototyping and baseline comparisons
- Following established evaluation protocols

## Task Specification Format

Custom tasks require explicit harness specification using the format:

```text
"<harness_name>.<task_name>"
```

**Examples**:

- `"lm-evaluation-harness.lambada_openai"` - LM-Eval harness task
- `"simple-evals.human_eval"` - Simple-Evals harness task  
- `"bigcode-evaluation-harness.humaneval_python"` - BigCode harness task

## lambada_openai (Log-Probability Task)

The `lambada_openai` task evaluates reading comprehension using log-probabilities.

```bash
pip install nvidia-lm-eval==25.6
```

1. Deploy your model:

   ```{literalinclude} ../scripts/snippets/deploy.py
   :language: python
   :start-after: "## Deploy"
   :linenos:
   ```

   ```bash
   python deploy.py
   ```

2. Configure and run the evaluation:

   ```{literalinclude} ../scripts/snippets/lambada.py
   :language: python
   :start-after: "## Run the evaluation"
   :linenos:
   ```

**Key Configuration Notes**:

- Uses log-probabilities for evaluation (refer to [Log-Probability Evaluation](run-evals/log-probability))
- Requires tokenizer configuration for proper probability calculation
- `limit_samples=10` used for quick testing (remove for full evaluation)

## WikiText (Perplexity Task)

Evaluate language modeling capabilities using the WikiText benchmark:

```python
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, ConfigParams, EndpointType, EvaluationConfig, EvaluationTarget
)
from nemo_evaluator.core.evaluate import evaluate

# Configure evaluation
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/", 
        type=EndpointType.COMPLETIONS, 
        model_id="megatron_model"
    )
)

eval_config = EvaluationConfig(
    type="lm-evaluation-harness.wikitext",
    params=ConfigParams(
        extra={
            "tokenizer_backend": "huggingface",
            "tokenizer": "/checkpoints/llama-3_2-1b-instruct_v2.0/context/nemo_tokenizer"
        }
    ),
    output_dir="./wikitext-results"
)

results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
```

## HumanEval (Code Generation)

Evaluate code generation capabilities using HumanEval:

```bash
# Install simple-evals framework
pip install nvidia-simple-evals==25.6
```

```python
# Configure HumanEval evaluation
eval_config = EvaluationConfig(
    type="simple-evals.human_eval",
    params=ConfigParams(
        temperature=0.2,  # Slight randomness for code diversity
        max_new_tokens=512,  # Sufficient for code solutions
        limit_samples=20,  # Test subset
        extra={
            "pass_at_k": [1, 5, 10],  # Evaluate pass@1, pass@5, pass@10
            "timeout": 10  # Code execution timeout
        }
    ),
    output_dir="./humaneval-results"
)
```

**Key Configuration Notes**:

- Uses chat endpoint for instruction-tuned models
- Requires code execution environment
- `pass_at_k` metrics measure success rates

## Multi-Language Code Evaluation

```bash
# Install BigCode evaluation harness
pip install nvidia-bigcode-eval==25.6
```

```python
# Configure multi-language code evaluation  
eval_config = EvaluationConfig(
    type="bigcode-evaluation-harness.humaneval_multilang",
    params=ConfigParams(
        temperature=0.1,
        max_new_tokens=1024,
        extra={
            "languages": ["python", "java", "cpp"],  # Target languages
            "num_workers": 4,  # Parallel execution workers
            "eval_metric": "pass_at_k"
        }
    ),
    output_dir="./multilang-results"
)
```

---

## Advanced Configuration Patterns

### Custom Few-Shot Configuration

```python
# Configure custom few-shot settings
params = ConfigParams(
    limit_samples=100,
    extra={
        "num_fewshot": 5,  # Number of examples in prompt
        "fewshot_delimiter": "\\n\\n",  # Separator between examples
        "fewshot_seed": 42,  # Reproducible example selection
        "description": "Answer the following question:",  # Custom prompt prefix
    }
)
```

### Performance Optimization

```python
# Optimize for high-throughput evaluation
params = ConfigParams(
    parallelism=16,  # Concurrent request threads
    max_retries=5,   # Retry failed requests
    request_timeout=120,  # Timeout per request (seconds)
    temperature=0,   # Deterministic for reproducibility
    extra={
        "batch_size": 8,  # Requests per batch (if supported)
        "cache_requests": True  # Enable request caching
    }
)
```

### Custom Tokenizer Configuration

```python
# Configure task-specific tokenizers
params = ConfigParams(
    extra={
        # Hugging Face tokenizer
        "tokenizer_backend": "huggingface",
        "tokenizer": "/path/to/nemo_tokenizer",
        
        # Alternative: Direct tokenizer specification
        "tokenizer_name": "meta-llama/Llama-2-7b-hf",
        "add_bos_token": True,
        "add_eos_token": False,
        
        # Trust remote code for custom tokenizers
        "trust_remote_code": True
    }
)
```

### Task-Specific Generation Settings

```python
# Configure generation for different task types

# Academic benchmarks (deterministic)
academic_params = ConfigParams(
    temperature=0,
    top_p=1.0,
    max_new_tokens=256,
    extra={"do_sample": False}
)

# Creative tasks (controlled randomness)  
creative_params = ConfigParams(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512,
    extra={"repetition_penalty": 1.1}
)

# Code generation (balanced)
code_params = ConfigParams(
    temperature=0.2,
    top_p=0.95,
    max_new_tokens=1024,
    extra={"stop_sequences": ["```", "\\n\\n"]}
)
```

## Configuration Reference

For comprehensive parameter documentation including universal settings, framework-specific options, and optimization patterns, refer to [Evaluation Configuration Parameters](parameters.md).

### Key Custom Task Considerations

When configuring custom tasks, pay special attention to:

- **Tokenizer Requirements**: Log-probability tasks require `tokenizer` and `tokenizer_backend` in `extra`
- **Framework-Specific Parameters**: Each harness supports different parameters in the `extra` dictionary
- **Performance Tuning**: Adjust `parallelism` and timeout settings based on task complexity
- **Reproducibility**: Use `temperature=0` and set `fewshot_seed` for consistent results

