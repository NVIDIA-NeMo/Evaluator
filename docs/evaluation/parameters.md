(eval-parameters)=

# Evaluation Configuration Parameters

Comprehensive reference for configuring evaluation tasks in {{ product_name_short }}, covering universal parameters, framework-specific settings, and optimization patterns.

:::{admonition} Quick Navigation
:class: info

**Looking for available benchmarks?**
- {ref}`eval-benchmarks` - Browse available benchmarks by category

**Need help getting started?**
- {ref}`evaluation-overview` - Overview of evaluation workflows
- {ref}`eval-run` - Step-by-step evaluation guides
:::

## Overview

All evaluation tasks in {{ product_name_short }} use the {ref}`ConfigParams <modelling-inout>` class for configuration. This provides a consistent interface across different evaluation harnesses while allowing framework-specific customization through the `extra` parameter. Default configuration (including which parameters a task uses) is defined in the **Framework Definition File (FDF)** for each framework; see {ref}`framework-definition-file` for details.

:::{admonition} How to see possible parameters for a given task
:class: important

**Python API (core)** — Get default params and which params a task uses. Use `framework_name.task_name` to avoid ambiguity when the same task name exists in multiple harnesses:

```python
from nemo_evaluator.core.input import get_available_evaluations

# Returns (framework_evals_mapping, framework_defaults, all_eval_name_mapping)
framework_evals, _, _ = get_available_evaluations()

# Use framework_name.task_name (e.g. simple_evals.mmlu_pro) for a single task
framework_name, task_name = "simple_evals", "mmlu_pro"
eval_obj = framework_evals[framework_name][task_name]

# Default params for this task (ConfigParams / dict-like)
print(eval_obj.config.params)

# Command template shows which {{ config.params.* }} the task uses
print(eval_obj.command)
```

**CLI (core)** — List tasks, then show merged config (including params) for a task:

```bash
# List available tasks
nemo-evaluator ls

# Show full rendered config (including config.params) for a task without running
# Use framework_name.task_name (e.g. simple_evals.mmlu_pro) to avoid ambiguity
nemo-evaluator run_eval --eval_type simple_evals.mmlu_pro --model_id x --model_url https://example.com/v1/chat/completions --model_type chat --output_dir ./out --dry_run
```

The `--dry_run` output prints the merged configuration (YAML) and the rendered command, so you can see which parameters apply to that task.

**Launcher** — If you use the launcher, `nemo-evaluator-launcher ls task <task_name>` (or `harness.task_name`) prints task details including **Defaults** with `config.params` and `config.params.extra`. List all tasks with `nemo-evaluator-launcher ls tasks`.
:::

```python
from nemo_evaluator.api.api_dataclasses import ConfigParams

# Basic configuration
params = ConfigParams(
    temperature=0,
    top_p=1.0,
    max_new_tokens=256,
    limit_samples=100
)

# With framework-specific parameters (extra)
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

These parameters are standardized across all frameworks and share the same names and semantics. That does **not** mean every framework supports every parameter: each task’s command template only uses a subset. If you pass a parameter that the task does not use, you will see a warning like: *"Configuration contains parameter(s) that are not used in the command template"* (see `validate_params_in_command` in `nemo_evaluator.core.utils`). 

```{list-table}
:header-rows: 1
:widths: 12 14 10 28 22 14

* - Category
  - Parameter
  - Type
  - Description
  - Example Values
  - Notes
* - Sampling
  - `temperature`
  - `float`
  - Sampling randomness
  - `0` (deterministic), `0.7` (creative)
  - Use `0` for reproducible results
* - Sampling
  - `top_p`
  - `float`
  - Nucleus sampling threshold
  - `1.0` (disabled), `0.9` (selective)
  - Controls diversity of generated text
* - Sampling
  - `max_new_tokens`
  - `int`
  - Maximum response length
  - `256`, `512`, `1024`
  - Limits generation length
* - Evaluation control
  - `limit_samples`
  - `int/float`
  - Evaluation subset size
  - `100` (count), `0.1` (10% of dataset)
  - Use for quick testing or resource limits
* - Evaluation control
  - `task`
  - `str`
  - Task-specific identifier
  - `"custom_task"`
  - Used by some harnesses for task routing
* - Performance
  - `parallelism`
  - `int`
  - Concurrent request threads
  - `1`, `8`, `16`
  - Balance against server capacity
* - Performance
  - `max_retries`
  - `int`
  - Retry attempts for failed requests
  - `3`, `5`, `10`
  - Increases robustness for network issues
* - Performance
  - `request_timeout`
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

## Parameter Selection Guidelines

- Configure `parallelism` and `request_timeout` based on server capacity.
- Use `limit_samples` for subset evaluation (e.g. for debugging or quick validation).

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

## Next Steps

- **Basic Usage**: See {ref}`text-gen` for getting started
- **Custom Tasks**: Learn {ref}`eval-custom-tasks` for specialized evaluations
- **Troubleshooting**: Refer to {ref}`troubleshooting-index` for common issues
- **Benchmarks**: Browse {ref}`eval-benchmarks` for task-specific recommendations
