(function-calling)=

# Function Calling Evaluation

Assess tool use capabilities, API calling accuracy, and structured output generation for agent-like behaviors using the Berkeley Function Calling Leaderboard (BFCL).

## Overview

Function calling evaluation measures a model's ability to:

- **Tool Discovery**: Identify appropriate functions for given tasks
- **Parameter Extraction**: Extract correct parameters from natural language
- **API Integration**: Generate proper function calls and handle responses  
- **Multi-Step Reasoning**: Chain function calls for complex workflows
- **Error Handling**: Manage invalid parameters and API failures

## Before You Start

Ensure you have:

- **Chat Model Endpoint**: Function calling requires chat-formatted OpenAI-compatible endpoints
- **API Access**: Valid API key for your model endpoint
- **Structured Output Support**: Model capable of generating JSON/function call formats

---

## Choose Your Approach

::::{tab-set}
:::{tab-item} NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run function calling evaluations with unified CLI:

```bash
# List available function calling tasks
nemo-evaluator-launcher ls tasks | grep -E "(bfcl|function)"

# Run BFCL AST prompting evaluation
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["bfclv3_ast_prompting"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}
```
:::

:::{tab-item}  Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig,
    EvaluationTarget,
    ApiEndpoint,
    ConfigParams,
    EndpointType
)

# Configure function calling evaluation
eval_config = EvaluationConfig(
    type="bfclv3_ast_prompting",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,    # Remove for full dataset
        temperature=0.1,     # Low temperature for precise function calls
        max_new_tokens=512,  # Adequate for function call generation
        top_p=0.9
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct", 
        type=EndpointType.CHAT,
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
# Pull and run BFCL evaluation container
docker run --rm -it nvcr.io/nvidia/eval-factory/bfcl:{{ docker_compose_latest }} bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here

# Run function calling evaluation
nemo-evaluator run_eval \
    --eval_type bfclv3_ast_prompting \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=10,config.params.temperature=0.1'
```
:::
::::

## Installation

Install the BFCL evaluation package for local development:

```bash
pip install nvidia-bfcl==25.7.1
```

## Discovering Available Tasks

Use the launcher CLI to discover all available function calling tasks:

```bash
# List all available benchmarks
nemo-evaluator-launcher ls tasks

# Filter for function calling tasks
nemo-evaluator-launcher ls tasks | grep -E "(bfcl|function)"
```

## Available Function Calling Tasks

BFCL provides comprehensive function calling benchmarks:

| Task | Description | Complexity | Format |
|------|-------------|------------|---------|
| `bfclv3_ast_prompting` | AST-based function calling with structured output | Intermediate | Structured |
| `bfclv2_ast_prompting` | BFCL v2 AST-based function calling (legacy) | Intermediate | Structured |

## Basic Function Calling Evaluation

The most comprehensive BFCL task is AST-based function calling that evaluates structured function calling. Use any of the three approaches above to run BFCL evaluations.

### Understanding Function Calling Format

BFCL evaluates models on their ability to generate proper function calls:

**Input Example**:
```text
What's the weather like in San Francisco and New York?

Available functions:
- get_weather(city: str, units: str = "celsius") -> dict
```

**Expected Output**:
```json
[
    {"name": "get_weather", "arguments": {"city": "San Francisco"}},
    {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

## Advanced Configuration

### Custom Evaluation Parameters

```python
# Optimized settings for function calling
eval_params = ConfigParams(
    limit_samples=100,
    parallelism=2,              # Conservative for complex reasoning
    temperature=0.1,            # Low temperature for precise function calls
    max_new_tokens=512,         # Adequate for function call generation
    top_p=0.9                   # Focused sampling for accuracy
)

eval_config = EvaluationConfig(
    type="bfclv3_ast_prompting",
    output_dir="/results/bfcl_optimized/",
    params=eval_params
)
```

### Multi-Task Function Calling Evaluation

Evaluate multiple BFCL versions:

```python
function_calling_tasks = [
    "bfclv2_ast_prompting",  # BFCL v2
    "bfclv3_ast_prompting"   # BFCL v3 (latest)
]

results = {}

for task in function_calling_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/{task}/",
        params=ConfigParams(
            limit_samples=50,
            temperature=0.0,    # Deterministic for consistency
            parallelism=1       # Sequential for complex reasoning
        )
    )
    
    result = evaluate(
        target_cfg=target_config, 
        eval_cfg=eval_config
    )
    results[task] = result
    
    # Access metrics from EvaluationResult object
    print(f"Completed {task} evaluation")
    print(f"Results: {result}")
```

## Understanding Metrics

### Results Structure

The `evaluate()` function returns an `EvaluationResult` object containing task-level and metric-level results:

```python
from nemo_evaluator.core.evaluate import evaluate

result = evaluate(eval_cfg=eval_config, target_cfg=target_config)

# Access task results
if result.tasks:
    for task_name, task_result in result.tasks.items():
        print(f"Task: {task_name}")
        for metric_name, metric_result in task_result.metrics.items():
            for score_name, score in metric_result.scores.items():
                print(f"  {metric_name}.{score_name}: {score.value}")
```

### Interpreting BFCL Scores

BFCL evaluations measure function calling accuracy across various dimensions. The specific metrics depend on the BFCL version and configuration. Check the `results.yml` output file for detailed metric breakdowns.

---

*For more function calling tasks and advanced configurations, see the [BFCL package documentation](https://pypi.org/project/nvidia-bfcl/).*
