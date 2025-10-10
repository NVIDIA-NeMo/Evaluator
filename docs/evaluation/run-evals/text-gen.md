(text-gen)=

# Text Generation Evaluation

<!-- active person first intro sentence -->

Text generation evaluation is the primary method for assessing LLM capabilities where models produce natural language responses to prompts. This approach evaluates the quality, accuracy, and appropriateness of generated text across various tasks and domains.

## Before You Start

Ensure you have:

1. **Model Endpoint**: An OpenAI-compatible API endpoint for your model (completions or chat)
2. **API Access**: Valid API key if your endpoint requires authentication
3. **Installed Packages**: NeMo Evaluator or access to evaluation containers
4. **Sufficient Resources**: Adequate compute for your chosen benchmarks

### Pre-Flight Check

Verify your setup before running full evaluation:

```{literalinclude} ../_snippets/prerequisites/endpoint_check.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::{tip}
**Run this script directly**: `python docs/evaluation/_snippets/prerequisites/endpoint_check.py`
:::

---

## Evaluation Approach

In text generation evaluation:

1. **Prompt Construction**: Models receive carefully crafted prompts (questions, instructions, or text to continue)
2. **Response Generation**: Models generate natural language responses using their trained parameters
3. **Response Assessment**: Generated text is evaluated for correctness, quality, or adherence to specific criteria
4. **Metric Calculation**: Numerical scores are computed based on evaluation criteria

This differs from **log-probability evaluation** where models assign confidence scores to predefined choices.
For log-probability methods, see the [Log-Probability Evaluation guide](../run-evals/log-probability).

## Choose Your Approach

::::{tab-set}
:::{tab-item} NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run text generation evaluations with unified CLI:

```bash
# List available text generation tasks
nemo-evaluator-launcher ls tasks

# Run MMLU Pro evaluation
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["mmlu_pro"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}

# Run multiple text generation benchmarks
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_text_generation_suite \
    -o 'evaluation.tasks=["mmlu_pro", "arc_challenge", "hellaswag", "truthfulqa"]'
```

:::

:::{tab-item}  Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams, EndpointType
)

# Configure text generation evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=None,  # Full dataset
        temperature=0.01,    # Near-deterministic for reproducibility
        max_new_tokens=512,
        top_p=0.95
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct", 
        type=EndpointType.CHAT,
        api_key="MY_API_KEY"  # Environment variable name containing your API key
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
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here

# Run evaluation
nemo-evaluator run_eval \
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

```{literalinclude} ../_snippets/commands/list_tasks.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

Run these commands to discover the complete list of available benchmarks across all installed frameworks.

## Text Generation Task Categories

```{list-table} Text Generation Benchmarks Overview
:header-rows: 1
:widths: 25 25 25 25

* - Area
  - Purpose
  - Example Tasks
  - Evaluation Method
* - Academic Benchmarks
  - Assess general knowledge and reasoning across academic domains
  - - `mmlu`
    - `mmlu_pro`
    - `arc_challenge`
    - `hellaswag`
    - `truthfulqa`
  - Multiple-choice or short-answer text generation
* - Instruction Following
  - Evaluate ability to follow complex instructions and formatting requirements
  - - `ifeval`
    - `gpqa_diamond`
  - Generated responses assessed against instruction criteria
* - Mathematical Reasoning
  - Test mathematical problem-solving and multi-step reasoning
  - - `gsm8k`
    - `math`
  - Final answer extraction and numerical comparison
* - Multilingual Evaluation
  - Assess capabilities across different languages
  - - `mgsm` (multilingual GSM8K)
  - Language-specific text generation and assessment
```

:::{note}
Task availability depends on installed frameworks. Use `nemo-evaluator-launcher ls tasks` to see the complete list for your environment.
:::

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

Resolve task naming conflicts by listing available tasks:

```python
from nemo_evaluator import show_available_tasks

# Display all tasks organized by framework
print("Available tasks by framework:")
show_available_tasks()
```

Or use the CLI for programmatic access:

```bash
# List all tasks with framework information
nemo-evaluator-launcher ls tasks

# Filter for specific tasks
nemo-evaluator-launcher ls tasks | grep mmlu
```

This helps you:
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
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, ConfigParams, EndpointType
)

# Configure target endpoint
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type=EndpointType.COMPLETIONS,
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation parameters
params = ConfigParams(
    temperature=0.01,   # Near-deterministic generation
    top_p=1.0,          # No nucleus sampling
    limit_samples=100,  # Evaluate subset for testing
    parallelism=1       # Single-threaded requests
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

**Quick Reference - Essential Parameters**:

```{literalinclude} ../_snippets/parameters/academic_minimal.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::{seealso}
**Complete Parameter Reference**

This guide shows minimal configuration for getting started. For comprehensive parameter options including:
- Framework-specific parameters (`num_fewshot`, `tokenizer`, etc.)
- Optimization patterns for different scenarios
- Troubleshooting common configuration issues
- Performance tuning guidelines

See {ref}`eval-parameters`.
:::

**Key Parameters for Text Generation**:
- `temperature`: Use 0.01 for near-deterministic, reproducible results
- `max_new_tokens`: Controls maximum response length
- `limit_samples`: Limits evaluation to a subset for testing
- `parallelism`: Balances speed with server capacity

## Understanding Results

After evaluation completes, you'll receive structured results with task-level metrics:

```{literalinclude} ../_snippets/api-examples/result_access.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

### Common Metrics

- **`acc` (Accuracy)**: Percentage of correct responses
- **`acc_norm` (Normalized Accuracy)**: Length-normalized scoring (often more reliable)
- **`exact_match`**: Exact string match percentage
- **`f1`**: F1 score for token-level overlap

Each metric includes statistics (mean, stderr) for confidence intervals.

## Multi-Task Evaluation

Evaluate across multiple academic benchmarks in a single workflow:

```{literalinclude} ../_snippets/api-examples/multi_task.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::{tip}
**Run this example**: `python docs/evaluation/_snippets/api-examples/multi_task.py`
:::

## Common Issues

::::{dropdown} "Temperature cannot be 0.0" Error
:icon: alert

Some endpoints don't support exact 0.0 temperature. Use 0.01 instead:

```python
params = ConfigParams(temperature=0.01)  # Near-deterministic
```
::::

::::{dropdown} Slow Evaluation Performance
:icon: alert

**Symptoms**: Evaluation takes too long or times out

**Solutions**:
- Increase `parallelism` (start with 4, scale to 8-16 based on endpoint capacity)
- Reduce `request_timeout` if requests hang
- Use `limit_samples` for initial testing before full runs
- Check endpoint health and availability

```python
# Optimized configuration
params = ConfigParams(
    parallelism=8,           # Higher concurrency
    request_timeout=120,     # Appropriate timeout
    limit_samples=100,       # Test subset first
    max_retries=3           # Retry failed requests
)
```
::::

::::{dropdown} API Authentication Errors
:icon: alert

**Symptoms**: 401 or 403 errors during evaluation

**Solutions**:
- Verify `api_key` parameter contains the environment variable NAME, not the key value
- Ensure the environment variable is set: `export YOUR_API_KEY="actual_key_value"`
- Check API key has necessary permissions

```bash
# Correct setup
export MY_API_KEY="nvapi-..."
```

```python
# Use environment variable name
api_endpoint=ApiEndpoint(
    api_key="MY_API_KEY"  # Name of env var, not the value
)
```
::::

::::{dropdown} Task Not Found Error
:icon: alert

**Symptoms**: Task name not recognized

**Solutions**:
- Verify task name with `nemo-evaluator-launcher ls tasks`
- Check if evaluation framework is installed
- Use framework-qualified names for ambiguous tasks (e.g., `lm-evaluation-harness.mmlu`)

```bash
# Discover available tasks
nemo-evaluator-launcher ls tasks | grep mmlu
```
::::

## Next Steps

- **Optimize Configuration**: See {ref}`eval-parameters` for advanced parameter tuning
- **Custom Tasks**: Learn {ref}`eval-custom-tasks` for specialized evaluations
- **Troubleshooting**: Refer to {ref}`troubleshooting-index` for detailed issue resolution
- **Benchmarks**: Browse {ref}`eval-benchmarks` for more evaluation tasks
