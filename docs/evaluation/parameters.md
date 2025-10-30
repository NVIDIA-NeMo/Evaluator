---
orphan: true
---
(eval-parameters)=

# Evaluation Configuration Parameters

Comprehensive reference for configuring evaluation tasks in {{ product_name_short }}, covering universal parameters, framework-specific settings, and optimization patterns.

:::{admonition} Quick Navigation
:class: info

**Looking for task-specific guides?**
- {ref}`text-gen` - Text generation evaluation
- {ref}`log-probability` - Log-probability evaluation
- {ref}`code-generation` - Code generation evaluation
- {ref}`safety-security` - Safety and security evaluation

**Looking for available benchmarks?**
- {ref}`eval-benchmarks` - Browse available benchmarks by category

**Need help getting started?**
- {ref}`evaluation-overview` - Overview of evaluation workflows
- {ref}`eval-run` - Step-by-step evaluation guides
:::

## Overview

All evaluation tasks in {{ product_name_short }} use the `ConfigParams` class for configuration. This provides a consistent interface across different evaluation harnesses while allowing framework-specific customization through the `extra` parameter.

```python
from nemo_evaluator.api.api_dataclasses import ConfigParams

# Basic configuration
params = ConfigParams(
    temperature=0,
    top_p=1.0,
    max_new_tokens=256,
    limit_samples=100
)

# Advanced configuration with framework-specific parameters
params = ConfigParams(
    temperature=0,
    parallelism=8,
    extra={
        "num_fewshot": 5,
        "tokenizer": "/path/to/tokenizer",
        "custom_prompt": "Answer the question:"
    }
)
```

## Universal Parameters

These parameters are available for all evaluation tasks regardless of the underlying harness or benchmark.

### Core Generation Parameters

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Notes
* - `temperature`
  - `float`
  - Sampling randomness
  - `0` (deterministic), `0.7` (creative)
  - Use `0` for reproducible results
* - `top_p`
  - `float`
  - Nucleus sampling threshold
  - `1.0` (disabled), `0.9` (selective)
  - Controls diversity of generated text
* - `max_new_tokens`
  - `int`
  - Maximum response length
  - `256`, `512`, `1024`
  - Limits generation length
```

### Evaluation Control Parameters

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Notes
* - `limit_samples`
  - `int/float`
  - Evaluation subset size
  - `100` (count), `0.1` (10% of dataset)
  - Use for quick testing or resource limits
* - `task`
  - `str`
  - Task-specific identifier
  - `"custom_task"`
  - Used by some harnesses for task routing
```

### Performance Parameters

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Notes
* - `parallelism`
  - `int`
  - Concurrent request threads
  - `1`, `8`, `16`
  - Balance against server capacity
* - `max_retries`
  - `int`
  - Retry attempts for failed requests
  - `3`, `5`, `10`
  - Increases robustness for network issues
* - `request_timeout`
  - `int`
  - Request timeout (seconds)
  - `60`, `120`, `300`
  - Adjust for model response time
```

## Framework-Specific Parameters

Framework-specific parameters are passed through the `extra` dictionary within `ConfigParams`.

::::{dropdown} LM-Evaluation-Harness Parameters
:icon: code-square

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Use Cases
* - `num_fewshot`
  - `int`
  - Few-shot examples count
  - `0`, `5`, `25`
  - Academic benchmarks
* - `tokenizer`
  - `str`
  - Tokenizer path
  - `"/path/to/tokenizer"`
  - Log-probability tasks
* - `tokenizer_backend`
  - `str`
  - Tokenizer implementation
  - `"huggingface"`, `"sentencepiece"`
  - Custom tokenizer setups
* - `trust_remote_code`
  - `bool`
  - Allow remote code execution
  - `True`, `False`
  - For custom tokenizers
* - `add_bos_token`
  - `bool`
  - Add beginning-of-sequence token
  - `True`, `False`
  - Model-specific formatting
* - `add_eos_token`
  - `bool`
  - Add end-of-sequence token
  - `True`, `False`
  - Model-specific formatting
* - `fewshot_delimiter`
  - `str`
  - Separator between examples
  - `"\\n\\n"`, `"\\n---\\n"`
  - Custom prompt formatting
* - `fewshot_seed`
  - `int`
  - Reproducible example selection
  - `42`, `1337`
  - Ensures consistent few-shot examples
* - `description`
  - `str`
  - Custom prompt prefix
  - `"Answer the question:"`
  - Task-specific instructions
* - `bootstrap_iters`
  - `int`
  - Statistical bootstrap iterations
  - `1000`, `10000`
  - For confidence intervals
```

::::

::::{dropdown} Simple-Evals Parameters
:icon: code-square

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Use Cases
* - `pass_at_k`
  - `list[int]`
  - Code evaluation metrics
  - `[1, 5, 10]`
  - Code generation tasks
* - `timeout`
  - `int`
  - Code execution timeout
  - `5`, `10`, `30`
  - Code generation tasks
* - `max_workers`
  - `int`
  - Parallel execution workers
  - `4`, `8`, `16`
  - Code execution parallelism
* - `languages`
  - `list[str]`
  - Target programming languages
  - `["python", "java", "cpp"]`
  - Multi-language evaluation
```

::::

::::{dropdown} BigCode-Evaluation-Harness Parameters
:icon: code-square

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Use Cases
* - `num_workers`
  - `int`
  - Parallel execution workers
  - `4`, `8`, `16`
  - Code execution parallelism
* - `eval_metric`
  - `str`
  - Evaluation metric
  - `"pass_at_k"`, `"bleu"`
  - Different scoring methods
* - `languages`
  - `list[str]`
  - Programming languages
  - `["python", "javascript"]`
  - Language-specific evaluation
```

::::

::::{dropdown} Safety and Specialized Harnesses
:icon: code-square

```{list-table}
:header-rows: 1
:widths: 15 10 30 25 20

* - Parameter
  - Type
  - Description
  - Example Values
  - Use Cases
* - `probes`
  - `str`
  - Garak security probes
  - `"ansiescape.AnsiEscaped"`
  - Security evaluation
* - `detectors`
  - `str`
  - Garak security detectors
  - `"base.TriggerListDetector"`
  - Security evaluation
* - `generations`
  - `int`
  - Number of generations per prompt
  - `1`, `5`, `10`
  - Safety evaluation
```

::::

## Configuration Patterns

::::{dropdown} Academic Benchmarks (Deterministic)
:icon: code-square

```python
academic_params = ConfigParams(
    temperature=0.01,      # Near-deterministic generation (0.0 not supported by all endpoints)
    top_p=1.0,             # No nucleus sampling
    max_new_tokens=256,    # Moderate response length
    limit_samples=None,    # Full dataset evaluation
    parallelism=4,         # Conservative parallelism
    extra={
        "num_fewshot": 5,  # Standard few-shot count
        "fewshot_seed": 42 # Reproducible examples
    }
)
```

::::

::::{dropdown} Creative Tasks (Controlled Randomness)
:icon: code-square

```python
creative_params = ConfigParams(
    temperature=0.7,       # Moderate creativity
    top_p=0.9,            # Nucleus sampling
    max_new_tokens=512,   # Longer responses
    extra={
        "repetition_penalty": 1.1,  # Reduce repetition
        "do_sample": True          # Enable sampling
    }
)
```

::::

::::{dropdown} Code Generation (Balanced)
:icon: code-square

```python
code_params = ConfigParams(
    temperature=0.2,       # Slight randomness for diversity
    top_p=0.95,           # Selective sampling
    max_new_tokens=1024,  # Sufficient for code solutions
    extra={
        "pass_at_k": [1, 5, 10],      # Multiple success metrics
        "timeout": 10,                # Code execution timeout
        "stop_sequences": ["```", "\\n\\n"]  # Code block terminators
    }
)
```

::::

::::{dropdown} Log-Probability Tasks
:icon: code-square

```python
logprob_params = ConfigParams(
    # No generation parameters needed for log-probability tasks
    limit_samples=100,    # Quick testing
    extra={
        "tokenizer_backend": "huggingface",
        "tokenizer": "/path/to/nemo_tokenizer",
        "trust_remote_code": True
    }
)
```

::::

::::{dropdown} High-Throughput Evaluation
:icon: code-square

```python
performance_params = ConfigParams(
    temperature=0.01,      # Near-deterministic for speed
    parallelism=16,       # High concurrency
    max_retries=5,        # Robust retry policy
    request_timeout=120,  # Generous timeout
    limit_samples=0.1,    # 10% sample for testing
    extra={
        "batch_size": 8,          # Batch requests if supported
        "cache_requests": True    # Enable caching
    }
)
```

::::

## Parameter Selection Guidelines

### By Evaluation Type

**Text Generation Tasks**:
- Use `temperature=0.01` for near-deterministic, reproducible results (most endpoints don't support exactly 0.0)
- Set appropriate `max_new_tokens` based on expected response length
- Configure `parallelism` based on server capacity

**Log-Probability Tasks**:
- Always specify `tokenizer` and `tokenizer_backend` in `extra`
- Generation parameters (temperature, top_p) are not used
- Focus on tokenizer configuration accuracy

**Code Generation Tasks**:
- Use moderate `temperature` (0.1-0.3) for diversity without randomness
- Set higher `max_new_tokens` (1024+) for complete solutions
- Configure `timeout` and `pass_at_k` in `extra`

**Safety Evaluation**:
- Use appropriate `probes` and `detectors` in `extra`
- Consider multiple `generations` per prompt
- Use chat endpoints for instruction-following safety tests

### By Resource Constraints

**Limited Compute**:
- Reduce `parallelism` to 1-4
- Use `limit_samples` for subset evaluation
- Increase `request_timeout` for slower responses

**High-Performance Clusters**:
- Increase `parallelism` to 16-32
- Enable request batching in `extra` if supported
- Use full dataset evaluation (`limit_samples=None`)

**Development/Testing**:
- Use `limit_samples=10-100` for quick validation
- Set `temperature=0.01` for consistent results
- Enable verbose logging in `extra` if available

## Common Configuration Errors

### Tokenizer Issues

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

### Performance Issues

:::{admonition} Problem
:class: error
Excessive parallelism overwhelming server

```python
# Incorrect - too many concurrent requests
params = ConfigParams(parallelism=100)
```
:::

:::{admonition} Solution
:class: tip
Start conservative and scale up

```python
# Correct - reasonable concurrency
params = ConfigParams(parallelism=8, max_retries=3)
```
:::

### Parameter Conflicts

:::{admonition} Problem
:class: error
Mixing generation and log-probability parameters

```python
# Incorrect - generation params unused for log-probability
params = ConfigParams(
    temperature=0.7,  # Ignored for log-probability tasks
    extra={"tokenizer": "/path"}
)
```
:::

:::{admonition} Solution
:class: tip
Use appropriate parameters for task type

```python
# Correct - only relevant parameters
params = ConfigParams(
    limit_samples=100,  # Relevant for all tasks
    extra={"tokenizer": "/path"}  # Required for log-probability
)
```
:::

## Best Practices

### Development Workflow

1. **Start Small**: Use `limit_samples=10` for initial validation
2. **Test Configuration**: Verify parameters work before full evaluation
3. **Monitor Resources**: Check memory and compute usage during evaluation
4. **Document Settings**: Record successful configurations for reproducibility

### Production Evaluation

1. **Deterministic Settings**: Use `temperature=0.01` for consistent results
2. **Full Datasets**: Remove `limit_samples` for complete evaluation
3. **Robust Configuration**: Set appropriate retries and timeouts
4. **Resource Planning**: Scale `parallelism` based on available infrastructure

### Parameter Tuning

1. **Task-Appropriate**: Match parameters to evaluation methodology
2. **Incremental Changes**: Adjust one parameter at a time
3. **Baseline Comparison**: Compare against known good configurations
4. **Performance Monitoring**: Track evaluation speed and resource usage

## Next Steps

- **Basic Usage**: See {ref}`text-gen` for getting started
- **Custom Tasks**: Learn {ref}`eval-custom-tasks` for specialized evaluations
- **Troubleshooting**: Refer to {ref}`troubleshooting-index` for common issues
- **Benchmarks**: Browse {ref}`eval-benchmarks` for task-specific recommendations
